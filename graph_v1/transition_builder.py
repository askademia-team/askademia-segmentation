from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from .builder import _cleanup_graph_edges, load_lecture_entries, merge_graph_nodes
from .llm import LLMClient
from .models import Graph, GraphEdge, GraphNode, SegmentationLayers, TransitionCandidate
from .retrieval import ensure_node_embeddings
from .clustering import build_semantic_cluster_hierarchy
from .validate import print_validation_summary, validate_graph


def spans_in_range(spans, start_ts: float, end_ts: float):
    return [s for s in spans if s.timestamp >= start_ts and s.timestamp < end_ts]


def dedupe_candidates(cands: List[TransitionCandidate], tolerance: float = 6.0) -> List[TransitionCandidate]:
    out: List[TransitionCandidate] = []
    for c in sorted(cands, key=lambda x: x.timestamp):
        if not out or abs(c.timestamp - out[-1].timestamp) > tolerance:
            out.append(c)
        else:
            if c.shift_score > out[-1].shift_score:
                out[-1] = c
    return out


def zscore(scores: List[float], value: float) -> float:
    if len(scores) < 3:
        return value
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / max(1, len(scores) - 1)
    std = math.sqrt(max(1e-9, var))
    return (value - mean) / std


def normalize_shift_scores(candidates: List[TransitionCandidate], recent_scores: List[float]) -> List[TransitionCandidate]:
    out: List[TransitionCandidate] = []
    for c in candidates:
        z = zscore(recent_scores, c.shift_score)
        calibrated = max(0.0, min(1.0, 0.5 + 0.2 * z))
        out.append(
            TransitionCandidate(
                timestamp=c.timestamp,
                shift_score=calibrated,
                shift_type=c.shift_type,
                confidence=c.confidence,
                rationale=c.rationale,
                source_window_id=c.source_window_id,
            )
        )
    return out


def classify_shift_type(score: float, major_thr: float, minor_thr: float) -> str:
    if score >= major_thr:
        return "major"
    if score >= minor_thr:
        return "minor"
    return "none"


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _rebuild_hierarchy(graph: Graph) -> None:
    fine_nodes = sorted([n for n in graph.nodes if n.layer == "fine"], key=lambda n: (n.start_ts, n.end_ts, n.id))
    coarse_nodes = sorted([n for n in graph.nodes if n.layer == "coarse"], key=lambda n: (n.start_ts, n.end_ts, n.id))
    if not fine_nodes or not coarse_nodes:
        return

    # Reset parent assignments and remove all existing structural edges.
    for fn in fine_nodes:
        fn.parent_coarse_id = None
    graph.edges = [e for e in graph.edges if e.type != "part_of"]

    assignments: Dict[str, str] = {}
    coarse_children: Dict[str, List[str]] = {cn.id: [] for cn in coarse_nodes}

    for fn in fine_nodes:
        containing = [cn for cn in coarse_nodes if cn.start_ts <= fn.start_ts and fn.end_ts <= cn.end_ts]
        if containing:
            chosen = min(containing, key=lambda cn: ((cn.end_ts - cn.start_ts), abs(((cn.start_ts + cn.end_ts) / 2.0) - ((fn.start_ts + fn.end_ts) / 2.0)), cn.id))
        else:
            # Fallback: choose the coarse node with the strongest temporal overlap;
            # if no overlap remains after merges, choose the nearest midpoint.
            scored = []
            fn_mid = (fn.start_ts + fn.end_ts) / 2.0
            for cn in coarse_nodes:
                ov = _overlap(fn.start_ts, fn.end_ts, cn.start_ts, cn.end_ts)
                cn_mid = (cn.start_ts + cn.end_ts) / 2.0
                scored.append((ov, -abs(cn_mid - fn_mid), -(cn.end_ts - cn.start_ts), cn))
            chosen = max(scored, key=lambda row: (row[0], row[1], row[2], row[3].id))[3]

        fn.parent_coarse_id = chosen.id
        chosen.start_ts = min(chosen.start_ts, fn.start_ts)
        chosen.end_ts = max(chosen.end_ts, fn.end_ts)
        assignments[fn.id] = chosen.id
        coarse_children[chosen.id].append(fn.id)

    # Drop coarse nodes that ended up empty after reassignment/merges.
    keep_coarse_ids = {cid for cid, children in coarse_children.items() if children}
    graph.nodes = [n for n in graph.nodes if n.layer != "coarse" or n.id in keep_coarse_ids]

    # Recreate part_of edges deterministically.
    existing_edge_nums = []
    for e in graph.edges:
        if e.id.startswith("edge_"):
            try:
                existing_edge_nums.append(int(e.id.split("_", 1)[1]))
            except Exception:
                pass
    next_edge_num = max(existing_edge_nums, default=0) + 1
    for fn in fine_nodes:
        parent_id = fn.parent_coarse_id
        if not parent_id or parent_id not in keep_coarse_ids:
            continue
        graph.edges.append(
            GraphEdge(
                id=f"edge_{next_edge_num}",
                from_id=fn.id,
                to_id=parent_id,
                type="part_of",
                reason="fine segment assigned to containing coarse topic segment",
                evidence_span_ids=fn.source_span_ids[:3],
                edge_confidence=0.9,
                evidence_count=min(3, len(fn.source_span_ids)),
            )
        )
        next_edge_num += 1

    _cleanup_graph_edges(graph)


def build_graph_transition_scored(
    audio_path: Path,
    video_path: Path | None,
    output_path: Path,
    window_sec: int = 75,
    overlap_sec: int = 15,
    merge_every_n_windows: int = 1,
    min_segment_sec: float = 25.0,
    max_segment_sec: float = 180.0,
    major_threshold: float = 0.70,
    minor_threshold: float = 0.35,
    verbose: bool = False,
) -> Tuple[Graph, List[Dict], SegmentationLayers, Dict[str, List[Dict]]]:
    lecture_id, spans = load_lecture_entries(audio_path, video_path)
    if not spans:
        g = Graph(lecture_id=lecture_id, nodes=[], edges=[], spans=[], metadata={})
        return g, [], SegmentationLayers(), {"accepted": [], "rejected": []}

    llm = LLMClient()

    start_ts = spans[0].timestamp
    end_ts = spans[-1].timestamp
    step = max(1, window_sec - overlap_sec)

    windows = []
    cur = start_ts
    w_idx = 0
    while cur <= end_ts:
        w_start = cur
        w_end = min(end_ts, cur + window_sec)
        window_spans = spans_in_range(spans, w_start, w_end)
        if window_spans:
            windows.append((f"w_{w_idx}", w_start, w_end, window_spans))
            w_idx += 1
        cur += step

    graph = Graph(
        lecture_id=lecture_id,
        spans=spans,
        metadata={
            "mode": "transition_scored_hierarchical",
            "window_sec": window_sec,
            "overlap_sec": overlap_sec,
            "merge_every_n_windows": merge_every_n_windows,
            "min_segment_sec": min_segment_sec,
            "max_segment_sec": max_segment_sec,
            "major_threshold": major_threshold,
            "minor_threshold": minor_threshold,
            "source_audio": str(audio_path),
            "source_video": str(video_path) if video_path else None,
            "llm_available": llm.available(),
        },
    )

    node_counter = 0
    edge_counter = 0
    merges: List[Dict] = []
    carry_start_ts = start_ts
    recent_segments: List[Dict] = []
    recent_shift_scores: List[float] = []
    prev_transitions: List[float] = []

    fine_boundaries: List[float] = []
    coarse_boundaries: List[float] = []
    accepted_labels: List[Dict] = []
    rejected_labels: List[Dict] = []

    total = len(windows)
    for i, (window_id, w_start, w_end, w_spans) in enumerate(windows):
        pct = (i + 1) / max(1, total) * 100
        print(f"[window {i+1}/{total} | {pct:5.1f}%] {w_start:7.1f}s -> {w_end:7.1f}s | nodes={len(graph.nodes)} edges={len(graph.edges)}")

        seg_start = min(carry_start_ts, w_start)
        sub_spans = spans_in_range(spans, seg_start, w_end)
        if not sub_spans:
            continue

        valid_ts = sorted({s.timestamp for s in sub_spans})
        candidate_ts = [t for t in valid_ts if (t - carry_start_ts) >= min_segment_sec]
        text = "\n".join(f"[{s.timestamp:.1f}s][{s.modality}] {s.text}" for s in sub_spans)

        raw_candidates: List[TransitionCandidate] = []
        if candidate_ts:
            raw_candidates = llm.detect_transition_candidates(
                window_text=text,
                window_id=window_id,
                valid_timestamps=candidate_ts,
                previous_transitions=prev_transitions,
                prior_segments=recent_segments[-2:],
            )

        candidates = dedupe_candidates(raw_candidates)
        candidates = normalize_shift_scores(candidates, recent_shift_scores[-30:])
        candidates = [c for c in candidates if c.timestamp > carry_start_ts]

        # periodic retrospective calibration: reweight with recent context every 6 windows
        if (i + 1) % 6 == 0 and candidates:
            candidates = sorted(candidates, key=lambda c: c.shift_score, reverse=True)

        if (w_end - carry_start_ts) >= max_segment_sec and not candidates:
            nearest = min(valid_ts, key=lambda x: abs(x - (carry_start_ts + max_segment_sec)))
            candidates = [
                TransitionCandidate(
                    timestamp=nearest,
                    shift_score=max(0.5, minor_threshold),
                    shift_type="minor",
                    confidence=0.4,
                    rationale="forced_split_max_segment",
                    source_window_id=window_id,
                )
            ]

        # pick up to 2 best candidates by score
        chosen = sorted(candidates, key=lambda c: c.shift_score, reverse=True)[:2]
        chosen = sorted(chosen, key=lambda c: c.timestamp)

        created_nodes = 0
        created_edges = 0
        dropped_too_short = 0
        dropped_no_spans = 0
        carry_before = carry_start_ts

        for c in chosen:
            ctype = classify_shift_type(c.shift_score, major_threshold, minor_threshold)
            c = TransitionCandidate(
                timestamp=c.timestamp,
                shift_score=c.shift_score,
                shift_type=ctype,  # type: ignore[arg-type]
                confidence=c.confidence,
                rationale=c.rationale,
                source_window_id=c.source_window_id,
            )
            if c.shift_type == "none":
                rejected_labels.append({"candidate": asdict(c), "reason": "below_minor_threshold"})
                continue
            if (c.timestamp - carry_start_ts) < min_segment_sec:
                dropped_too_short += 1
                rejected_labels.append({"candidate": asdict(c), "reason": "too_short"})
                continue

            seg_spans = spans_in_range(spans, carry_start_ts, c.timestamp)
            if not seg_spans:
                dropped_no_spans += 1
                rejected_labels.append({"candidate": asdict(c), "reason": "no_spans"})
                continue

            seg_text = "\n".join(f"[{s.timestamp:.1f}s][{s.modality}] {s.text}" for s in seg_spans)
            source_ids = [s.span_id for s in seg_spans]
            node_payload = llm.summarize_segment_to_node(seg_text, source_ids, carry_start_ts, c.timestamp)

            node_counter += 1
            node_id = f"node_{node_counter}"
            graph.nodes.append(
                GraphNode(
                    id=node_id,
                    lecture_id=lecture_id,
                    title=node_payload["title"],
                    summary=node_payload["summary"],
                    explanation=node_payload["explanation"],
                    source_span_ids=node_payload["source_span_ids"],
                    start_ts=carry_start_ts,
                    end_ts=c.timestamp,
                    layer="fine",
                    parent_coarse_id=None,
                )
            )
            created_nodes += 1
            if verbose:
                print(f"    - {node_payload['title']} ({c.shift_type}, score={c.shift_score:.2f})")

            # weak label accepted
            accepted_labels.append({"candidate": asdict(c), "node_id": node_id})
            recent_shift_scores.append(c.shift_score)
            fine_boundaries.append(c.timestamp)
            if c.shift_type == "major":
                coarse_boundaries.append(c.timestamp)

            existing = [{"id": n.id, "title": n.title, "summary": n.summary} for n in graph.nodes[:-1] if n.layer == "fine"]
            proposed_edges = llm.propose_edges_for_new_node(
                {
                    "id": node_id,
                    "title": node_payload["title"],
                    "summary": node_payload["summary"],
                    "explanation": node_payload["explanation"],
                },
                existing,
                source_ids,
            )
            valid_ids = {n.id for n in graph.nodes}
            for pe in proposed_edges:
                tgt = pe["to_existing"]
                if tgt not in valid_ids:
                    continue
                edge_counter += 1
                graph.edges.append(
                    GraphEdge(
                        id=f"edge_{edge_counter}",
                        from_id=node_id,
                        to_id=tgt,
                        type=pe["type"],
                        reason=pe["reason"],
                        evidence_span_ids=pe["evidence_span_ids"],
                        edge_confidence=0.6,
                        evidence_count=len(pe["evidence_span_ids"]),
                    )
                )
                created_edges += 1

            carry_start_ts = c.timestamp

        prev_transitions = [c.timestamp for c in chosen]
        recent_segments.extend(
            [
                {
                    "title": n.title,
                    "summary": n.summary,
                    "start_ts": n.start_ts,
                    "end_ts": n.end_ts,
                }
                for n in graph.nodes[-created_nodes:]
            ]
        )

        print(
            "  + transitions:"
            f" raw={len(raw_candidates)} deduped={len(candidates)} chosen={len(chosen)} "
            f"created_nodes={created_nodes} created_edges={created_edges} "
            f"dropped_too_short={dropped_too_short} dropped_no_spans={dropped_no_spans} "
            f"carry={carry_before:.1f}->{carry_start_ts:.1f}"
        )

        if (i + 1) % merge_every_n_windows == 0:
            new_merges = merge_graph_nodes(graph, llm)
            merges.extend(new_merges)
            if new_merges:
                print(f"  * merge_pass: merged={len(new_merges)}")
                for m in new_merges:
                    print(f"    - {m['merged_from']} -> {m['merged_into']} ({m['reason']})")

    # flush final fine segment
    if end_ts - carry_start_ts >= min_segment_sec:
        seg_spans = spans_in_range(spans, carry_start_ts, end_ts + 1e-6)
        if seg_spans:
            seg_text = "\n".join(f"[{s.timestamp:.1f}s][{s.modality}] {s.text}" for s in seg_spans)
            source_ids = [s.span_id for s in seg_spans]
            node_payload = llm.summarize_segment_to_node(seg_text, source_ids, carry_start_ts, end_ts)
            node_counter += 1
            node_id = f"node_{node_counter}"
            graph.nodes.append(
                GraphNode(
                    id=node_id,
                    lecture_id=lecture_id,
                    title=node_payload["title"],
                    summary=node_payload["summary"],
                    explanation=node_payload["explanation"],
                    source_span_ids=node_payload["source_span_ids"],
                    start_ts=carry_start_ts,
                    end_ts=end_ts,
                    layer="fine",
                    parent_coarse_id=None,
                )
            )

    # coarse segmentation materialization from major boundaries
    all_coarse = sorted({b for b in coarse_boundaries if start_ts < b < end_ts})
    coarse_segments = [start_ts] + all_coarse + [end_ts]
    for idx in range(len(coarse_segments) - 1):
        cs, ce = coarse_segments[idx], coarse_segments[idx + 1]
        if ce - cs < min_segment_sec:
            continue
        seg_spans = spans_in_range(spans, cs, ce)
        if not seg_spans:
            continue
        seg_text = "\n".join(f"[{s.timestamp:.1f}s][{s.modality}] {s.text}" for s in seg_spans)
        source_ids = [s.span_id for s in seg_spans]
        node_payload = llm.summarize_segment_to_node(seg_text, source_ids, cs, ce)
        node_counter += 1
        coarse_id = f"node_{node_counter}"
        graph.nodes.append(
            GraphNode(
                id=coarse_id,
                lecture_id=lecture_id,
                title=node_payload["title"],
                summary=node_payload["summary"],
                explanation=node_payload["explanation"],
                source_span_ids=source_ids,
                start_ts=cs,
                end_ts=ce,
                layer="coarse",
                parent_coarse_id=None,
            )
        )

        # link contained fine nodes via part_of
        fine_nodes = [n for n in graph.nodes if n.layer == "fine" and n.start_ts >= cs and n.end_ts <= ce]
        for fn in fine_nodes:
            fn.parent_coarse_id = coarse_id
            edge_counter += 1
            graph.edges.append(
                GraphEdge(
                    id=f"edge_{edge_counter}",
                    from_id=fn.id,
                    to_id=coarse_id,
                    type="part_of",
                    reason="fine segment is contained within coarse major topic segment",
                    evidence_span_ids=fn.source_span_ids[:3],
                    edge_confidence=0.9,
                    evidence_count=min(3, len(fn.source_span_ids)),
                )
            )

    # final cleanup merge/edges
    final_merges = merge_graph_nodes(graph, llm)
    merges.extend(final_merges)
    if final_merges:
        print(f"  * final_merge_pass: merged={len(final_merges)}")
        for m in final_merges:
            print(f"    - {m['merged_from']} -> {m['merged_into']} ({m['reason']})")

    # Final merges can invalidate coarse containment. Rebuild structural hierarchy now.
    _rebuild_hierarchy(graph)

    calibration_state = {
        "major_threshold": major_threshold,
        "minor_threshold": minor_threshold,
        "recent_score_mean": (sum(recent_shift_scores) / len(recent_shift_scores)) if recent_shift_scores else 0.0,
        "recent_score_count": len(recent_shift_scores),
    }
    layers = SegmentationLayers(
        coarse_boundaries=all_coarse,
        fine_boundaries=sorted(set(fine_boundaries)),
        calibration_state=calibration_state,
    )

    graph.metadata["segmentation_layers"] = asdict(layers)
    graph.metadata["weak_label_counts"] = {"accepted": len(accepted_labels), "rejected": len(rejected_labels)}

    build_semantic_cluster_hierarchy(graph, llm, max_cluster_layers=2, max_clusters=8)
    ensure_node_embeddings(graph, llm)
    validation = validate_graph(graph, min_segment_sec=min_segment_sec)
    print_validation_summary(validation)
    graph.metadata["validation"] = {
        "ok": validation["ok"],
        "error_count": validation["error_count"],
        "warning_count": validation["warning_count"],
    }

    print(f"[done] nodes={len(graph.nodes)} edges={len(graph.edges)} merges={len(merges)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2)

    merge_path = output_path.with_name(output_path.stem + "_merge_log.json")
    with merge_path.open("w", encoding="utf-8") as f:
        json.dump(merges, f, indent=2)

    weak_labels = {"accepted": accepted_labels, "rejected": rejected_labels}
    weak_path = output_path.with_name(output_path.stem + "_weak_labels.json")
    with weak_path.open("w", encoding="utf-8") as f:
        json.dump(weak_labels, f, indent=2)

    validation_path = output_path.with_name(output_path.stem + "_validation.json")
    with validation_path.open("w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    return graph, merges, layers, weak_labels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hierarchical transition-scored graph builder")
    p.add_argument("--audio", required=True)
    p.add_argument("--video")
    p.add_argument("--output", required=True)
    p.add_argument("--window-sec", type=int, default=75)
    p.add_argument("--overlap-sec", type=int, default=15)
    p.add_argument("--merge-every-n", type=int, default=1)
    p.add_argument("--min-segment-sec", type=float, default=25.0)
    p.add_argument("--max-segment-sec", type=float, default=180.0)
    p.add_argument("--major-threshold", type=float, default=0.70)
    p.add_argument("--minor-threshold", type=float, default=0.35)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_graph_transition_scored(
        audio_path=Path(args.audio),
        video_path=Path(args.video) if args.video else None,
        output_path=Path(args.output),
        window_sec=args.window_sec,
        overlap_sec=args.overlap_sec,
        merge_every_n_windows=args.merge_every_n,
        min_segment_sec=args.min_segment_sec,
        max_segment_sec=args.max_segment_sec,
        major_threshold=args.major_threshold,
        minor_threshold=args.minor_threshold,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
