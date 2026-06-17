from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from graph_v1.builder import _cleanup_graph_edges, lexical_similarity, load_lecture_entries
from graph_v1.models import Graph, GraphEdge, GraphNode, SegmentationLayers, TransitionCandidate
from graph_v1.retrieval import ensure_node_embeddings
from graph_v1.validate import print_validation_summary

from .llm import GraphV2LLMClient
from .validate import validate_graph_v2


def spans_in_range(spans, start_ts: float, end_ts: float):
    return [s for s in spans if s.timestamp >= start_ts and s.timestamp < end_ts]


def dedupe_candidates(cands: List[TransitionCandidate], tolerance: float = 8.0) -> List[TransitionCandidate]:
    out: List[TransitionCandidate] = []
    for c in sorted(cands, key=lambda x: x.timestamp):
        if not out or abs(c.timestamp - out[-1].timestamp) > tolerance:
            out.append(c)
        elif c.shift_score > out[-1].shift_score:
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


def choose_candidates(candidates: List[TransitionCandidate], max_candidates: int = 2) -> List[TransitionCandidate]:
    chosen = sorted(candidates, key=lambda c: c.shift_score, reverse=True)[:max_candidates]
    return sorted(chosen, key=lambda c: c.timestamp)


def _reassign_edge_ids(graph: Graph) -> None:
    for idx, edge in enumerate(graph.edges, start=1):
        edge.id = f"edge_{idx}"


def _semantic_edge_allowed_v2(graph: Graph, edge: GraphEdge) -> bool:
    node_by_id = {n.id: n for n in graph.nodes}
    from_node = node_by_id.get(edge.from_id)
    to_node = node_by_id.get(edge.to_id)
    if not from_node or not to_node:
        return False
    if edge.type == "part_of":
        return from_node.layer == "fine" and to_node.layer == "coarse" and from_node.parent_coarse_id == to_node.id
    if from_node.layer != to_node.layer:
        return False
    if from_node.layer == "fine" and from_node.parent_coarse_id != to_node.parent_coarse_id:
        return False
    return True


def cleanup_graph_v2_edges(graph: Graph) -> None:
    _cleanup_graph_edges(graph)
    graph.edges = [e for e in graph.edges if _semantic_edge_allowed_v2(graph, e)]
    _cleanup_graph_edges(graph)
    _reassign_edge_ids(graph)


def merge_nodes_local(
    graph: Graph,
    llm: GraphV2LLMClient,
    node_ids: Sequence[str],
    *,
    threshold: float = 0.38,
    merge_scope: str,
) -> List[Dict[str, Any]]:
    merges: List[Dict[str, Any]] = []
    node_by_id = {n.id: n for n in graph.nodes}
    nodes = [node_by_id[nid] for nid in node_ids if nid in node_by_id]
    nodes = sorted(nodes, key=lambda n: (n.start_ts, n.end_ts, n.id))
    if len(nodes) < 2:
        return []

    remap: Dict[str, str] = {}
    id_to_pos = {n.id: idx for idx, n in enumerate(nodes)}
    candidate_payload = [
        {
            "id": n.id,
            "title": n.title,
            "summary": n.summary,
            "explanation": n.explanation,
            "start_ts": n.start_ts,
            "end_ts": n.end_ts,
        }
        for n in nodes[-12:]
    ]
    for group in llm.propose_adjacent_merge_groups(candidate_payload):
        ids = group["node_ids"]
        positions = sorted(id_to_pos.get(i, -10**9) for i in ids)
        if not positions or any(p < 0 for p in positions):
            continue
        if positions != list(range(positions[0], positions[-1] + 1)):
            continue
        anchor = ids[0]
        anchor_node = node_by_id.get(anchor)
        if not anchor_node:
            continue
        for mid in ids[1:]:
            src = remap.get(mid, mid)
            if src == anchor or src in remap:
                continue
            src_node = node_by_id.get(src)
            if not src_node:
                continue
            if anchor_node.layer != src_node.layer:
                continue
            if anchor_node.layer == "fine" and anchor_node.parent_coarse_id != src_node.parent_coarse_id:
                continue
            sim_group = lexical_similarity(
                f"{anchor_node.title} {anchor_node.summary} {anchor_node.explanation}",
                f"{src_node.title} {src_node.summary} {src_node.explanation}",
            )
            if sim_group < 0.33:
                continue
            remap[src] = anchor
            anchor_node.source_span_ids = sorted(set(anchor_node.source_span_ids + src_node.source_span_ids))
            anchor_node.start_ts = min(anchor_node.start_ts, src_node.start_ts)
            anchor_node.end_ts = max(anchor_node.end_ts, src_node.end_ts)
            anchor_node.embedding = None
            merges.append({
                "scope": merge_scope,
                "merged_from": src,
                "merged_into": anchor,
                "reason": f"llm_group(sim={sim_group:.3f}): {group['reason']}",
            })

    for i in range(len(nodes) - 1):
        ni = nodes[i]
        nj = nodes[i + 1]
        if ni.id in remap or nj.id in remap:
            continue
        if ni.layer != nj.layer:
            continue
        if ni.layer == "fine" and ni.parent_coarse_id != nj.parent_coarse_id:
            continue
        sim = lexical_similarity(
            f"{ni.title} {ni.summary} {ni.explanation}",
            f"{nj.title} {nj.summary} {nj.explanation}",
        )
        if sim < threshold:
            continue
        if not llm.choose_merge_candidate(asdict(ni), asdict(nj)):
            continue
        remap[nj.id] = ni.id
        ni.source_span_ids = sorted(set(ni.source_span_ids + nj.source_span_ids))
        ni.start_ts = min(ni.start_ts, nj.start_ts)
        ni.end_ts = max(ni.end_ts, nj.end_ts)
        ni.embedding = None
        merges.append({"scope": merge_scope, "merged_from": nj.id, "merged_into": ni.id, "reason": f"similarity={sim:.3f}"})

    if remap:
        graph.nodes = [n for n in graph.nodes if n.id not in remap]
        for e in graph.edges:
            e.from_id = remap.get(e.from_id, e.from_id)
            e.to_id = remap.get(e.to_id, e.to_id)
        cleanup_graph_v2_edges(graph)
    return merges


def _node_payload(node: GraphNode) -> Dict[str, Any]:
    return {
        "id": node.id,
        "title": node.title,
        "summary": node.summary,
        "explanation": node.explanation,
        "start_ts": node.start_ts,
        "end_ts": node.end_ts,
    }


def _segment_pass(
    *,
    spans,
    graph: Graph,
    llm: GraphV2LLMClient,
    node_counter: int,
    edge_counter: int,
    phase: str,
    coarse_node: GraphNode | None = None,
    coarse_target_nodes: int = 12,
    fine_target_nodes: int = 5,
    window_sec: int,
    overlap_sec: int,
    min_segment_sec: float,
    max_segment_sec: float,
    merge_every_n_windows: int,
    verbose: bool,
) -> Tuple[List[GraphNode], List[GraphEdge], int, int, List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
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
            windows.append((f"{phase}_w_{w_idx}", w_start, w_end, window_spans))
            w_idx += 1
        cur += step

    created_nodes: List[GraphNode] = []
    created_edges: List[GraphEdge] = []
    merges: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    carry_start_ts = start_ts
    recent_nodes: List[Dict[str, Any]] = []
    recent_shift_scores: List[float] = []
    prev_transitions: List[float] = []

    for i, (window_id, w_start, w_end, _) in enumerate(windows):
        total = len(windows)
        pct = (i + 1) / max(1, total) * 100
        print(f"[{phase} {i+1}/{total} | {pct:5.1f}%] {w_start:7.1f}s -> {w_end:7.1f}s | nodes={len(created_nodes)}")
        seg_start = min(carry_start_ts, w_start)
        sub_spans = spans_in_range(spans, seg_start, w_end)
        if not sub_spans:
            continue
        valid_ts = sorted({s.timestamp for s in sub_spans})
        candidate_ts = [t for t in valid_ts if (t - carry_start_ts) >= min_segment_sec]
        text = "\n".join(f"[{s.timestamp:.1f}s][{s.modality}] {s.text}" for s in sub_spans)
        raw_candidates: List[TransitionCandidate] = []
        if candidate_ts:
            if phase == "coarse":
                raw_candidates = llm.detect_coarse_transition_candidates(
                    window_text=text,
                    window_id=window_id,
                    valid_timestamps=candidate_ts,
                    previous_transitions=prev_transitions,
                    recent_nodes=recent_nodes[-2:],
                    coarse_target_nodes=coarse_target_nodes,
                )
            else:
                raw_candidates = llm.detect_fine_transition_candidates(
                    window_text=text,
                    window_id=window_id,
                    valid_timestamps=candidate_ts,
                    coarse_node=_node_payload(coarse_node),
                    recent_nodes=recent_nodes[-2:],
                    fine_target_nodes=fine_target_nodes,
                )
        candidates = dedupe_candidates(raw_candidates)
        candidates = normalize_shift_scores(candidates, recent_shift_scores[-30:])
        candidates = [c for c in candidates if c.timestamp > carry_start_ts]
        if (w_end - carry_start_ts) >= max_segment_sec and not candidates and valid_ts:
            nearest = min(valid_ts, key=lambda x: abs(x - (carry_start_ts + max_segment_sec)))
            candidates = [
                TransitionCandidate(
                    timestamp=nearest,
                    shift_score=0.62 if phase == "coarse" else 0.52,
                    shift_type="major" if phase == "coarse" else "minor",
                    confidence=0.35,
                    rationale="forced_split_max_segment",
                    source_window_id=window_id,
                )
            ]
        chosen = choose_candidates(candidates, max_candidates=2)
        phase_created_count = 0
        dropped_too_short = 0
        for c in chosen:
            if (c.timestamp - carry_start_ts) < min_segment_sec:
                dropped_too_short += 1
                rejected.append({"candidate": asdict(c), "reason": "too_short", "phase": phase})
                continue
            seg_spans = spans_in_range(spans, carry_start_ts, c.timestamp)
            if not seg_spans:
                rejected.append({"candidate": asdict(c), "reason": "no_spans", "phase": phase})
                continue
            seg_text = "\n".join(f"[{s.timestamp:.1f}s][{s.modality}] {s.text}" for s in seg_spans)
            source_ids = [s.span_id for s in seg_spans]
            payload = llm.summarize_segment_to_node(seg_text, source_ids, carry_start_ts, c.timestamp)
            node_counter += 1
            node = GraphNode(
                id=f"node_{node_counter}",
                lecture_id=graph.lecture_id,
                title=payload["title"],
                summary=payload["summary"],
                explanation=payload["explanation"],
                source_span_ids=payload["source_span_ids"],
                start_ts=carry_start_ts,
                end_ts=c.timestamp,
                layer=phase,
                parent_coarse_id=coarse_node.id if phase == "fine" and coarse_node else None,
            )
            created_nodes.append(node)
            graph.nodes.append(node)
            phase_created_count += 1
            recent_nodes.append({
                "id": node.id,
                "title": node.title,
                "summary": node.summary,
                "start_ts": node.start_ts,
                "end_ts": node.end_ts,
            })
            recent_shift_scores.append(c.shift_score)
            accepted.append({"candidate": asdict(c), "node_id": node.id, "phase": phase, "coarse_id": coarse_node.id if coarse_node else None})
            if verbose:
                print(f"    - {node.title} ({c.shift_type}, score={c.shift_score:.2f})")

            if phase == "coarse":
                existing = [_node_payload(n) for n in created_nodes[:-1] if n.layer == "coarse"]
            else:
                existing = [_node_payload(n) for n in created_nodes[:-1] if n.layer == "fine"]
            proposed_edges = llm.propose_same_layer_edges(
                new_node=_node_payload(node),
                existing_nodes=existing,
                valid_span_ids=source_ids,
                layer=phase,
                coarse_context=_node_payload(coarse_node) if coarse_node else None,
            )
            valid_ids = {n.id for n in graph.nodes}
            for pe in proposed_edges:
                tgt = pe["to_existing"]
                if tgt not in valid_ids:
                    continue
                edge_counter += 1
                edge = GraphEdge(
                    id=f"edge_{edge_counter}",
                    from_id=node.id,
                    to_id=tgt,
                    type=pe["type"],
                    reason=pe["reason"],
                    evidence_span_ids=pe["evidence_span_ids"],
                    edge_confidence=0.6,
                    evidence_count=len(pe["evidence_span_ids"]),
                )
                created_edges.append(edge)
                graph.edges.append(edge)
            if phase == "fine" and coarse_node is not None:
                edge_counter += 1
                edge = GraphEdge(
                    id=f"edge_{edge_counter}",
                    from_id=node.id,
                    to_id=coarse_node.id,
                    type="part_of",
                    reason="fine node belongs to this coarse topic",
                    evidence_span_ids=node.source_span_ids[:3],
                    edge_confidence=0.9,
                    evidence_count=min(3, len(node.source_span_ids)),
                )
                created_edges.append(edge)
                graph.edges.append(edge)
            carry_start_ts = c.timestamp

        prev_transitions = [c.timestamp for c in chosen]
        print(
            f"  + transitions: raw={len(raw_candidates)} deduped={len(candidates)} chosen={len(chosen)} "
            f"created_nodes={phase_created_count} dropped_too_short={dropped_too_short} carry={seg_start:.1f}->{carry_start_ts:.1f}"
        )
        if phase_created_count and (i + 1) % merge_every_n_windows == 0:
            phase_node_ids = [n.id for n in created_nodes if n.layer == phase]
            if phase == "fine" and coarse_node is not None:
                phase_node_ids = [n.id for n in created_nodes if n.layer == "fine" and n.parent_coarse_id == coarse_node.id]
            new_merges = merge_nodes_local(
                graph,
                llm,
                phase_node_ids,
                merge_scope=f"{phase}:{coarse_node.id if coarse_node else 'global'}",
            )
            merges.extend(new_merges)
            if new_merges:
                print(f"  * merge_pass: merged={len(new_merges)}")

    if end_ts - carry_start_ts >= min_segment_sec:
        seg_spans = spans_in_range(spans, carry_start_ts, end_ts + 1e-6)
        if seg_spans:
            seg_text = "\n".join(f"[{s.timestamp:.1f}s][{s.modality}] {s.text}" for s in seg_spans)
            source_ids = [s.span_id for s in seg_spans]
            payload = llm.summarize_segment_to_node(seg_text, source_ids, carry_start_ts, end_ts)
            node_counter += 1
            node = GraphNode(
                id=f"node_{node_counter}",
                lecture_id=graph.lecture_id,
                title=payload["title"],
                summary=payload["summary"],
                explanation=payload["explanation"],
                source_span_ids=payload["source_span_ids"],
                start_ts=carry_start_ts,
                end_ts=end_ts,
                layer=phase,
                parent_coarse_id=coarse_node.id if phase == "fine" and coarse_node else None,
            )
            created_nodes.append(node)
            graph.nodes.append(node)
            if phase == "fine" and coarse_node is not None:
                edge_counter += 1
                edge = GraphEdge(
                    id=f"edge_{edge_counter}",
                    from_id=node.id,
                    to_id=coarse_node.id,
                    type="part_of",
                    reason="fine node belongs to this coarse topic",
                    evidence_span_ids=node.source_span_ids[:3],
                    edge_confidence=0.9,
                    evidence_count=min(3, len(node.source_span_ids)),
                )
                created_edges.append(edge)
                graph.edges.append(edge)

    if phase == "fine" and coarse_node is not None:
        phase_node_ids = [n.id for n in graph.nodes if n.layer == "fine" and n.parent_coarse_id == coarse_node.id]
        final_merges = merge_nodes_local(graph, llm, phase_node_ids, merge_scope=f"fine:{coarse_node.id}")
        merges.extend(final_merges)
    cleanup_graph_v2_edges(graph)
    return created_nodes, created_edges, node_counter, edge_counter, merges, {"accepted": accepted, "rejected": rejected}


def build_graph_v2(
    audio_path: Path,
    video_path: Path | None,
    output_path: Path,
    coarse_window_sec: int = 240,
    coarse_overlap_sec: int = 60,
    fine_window_sec: int = 90,
    fine_overlap_sec: int = 20,
    merge_every_n_windows: int = 1,
    coarse_min_segment_sec: float = 180.0,
    coarse_max_segment_sec: float = 900.0,
    fine_min_segment_sec: float = 25.0,
    fine_max_segment_sec: float = 240.0,
    coarse_target_nodes: int = 12,
    fine_target_nodes: int = 5,
    verbose: bool = False,
) -> Tuple[Graph, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    lecture_id, spans = load_lecture_entries(audio_path, video_path)
    graph = Graph(
        lecture_id=lecture_id,
        spans=spans,
        metadata={
            "mode": "coarse_first_hierarchical_v2",
            "source_audio": str(audio_path),
            "source_video": str(video_path) if video_path else None,
            "coarse_params": {
                "window_sec": coarse_window_sec,
                "overlap_sec": coarse_overlap_sec,
                "min_segment_sec": coarse_min_segment_sec,
                "max_segment_sec": coarse_max_segment_sec,
                "target_nodes": coarse_target_nodes,
            },
            "fine_params": {
                "window_sec": fine_window_sec,
                "overlap_sec": fine_overlap_sec,
                "min_segment_sec": fine_min_segment_sec,
                "max_segment_sec": fine_max_segment_sec,
                "target_nodes": fine_target_nodes,
            },
        },
    )
    if not spans:
        return graph, [], {}, {}

    llm = GraphV2LLMClient()
    merges: List[Dict[str, Any]] = []
    weak_labels: Dict[str, Any] = {"coarse": {"accepted": [], "rejected": []}, "fine": {}}
    node_counter = 0
    edge_counter = 0

    coarse_nodes, _, node_counter, edge_counter, coarse_merges, coarse_labels = _segment_pass(
        spans=spans,
        graph=graph,
        llm=llm,
        node_counter=node_counter,
        edge_counter=edge_counter,
        phase="coarse",
        coarse_target_nodes=coarse_target_nodes,
        fine_target_nodes=fine_target_nodes,
        window_sec=coarse_window_sec,
        overlap_sec=coarse_overlap_sec,
        min_segment_sec=coarse_min_segment_sec,
        max_segment_sec=coarse_max_segment_sec,
        merge_every_n_windows=merge_every_n_windows,
        verbose=verbose,
    )
    merges.extend(coarse_merges)
    weak_labels["coarse"] = coarse_labels

    coarse_nodes = sorted([n for n in graph.nodes if n.layer == "coarse"], key=lambda n: (n.start_ts, n.end_ts, n.id))
    for coarse_node in coarse_nodes:
        coarse_spans = spans_in_range(spans, coarse_node.start_ts, coarse_node.end_ts + 1e-6)
        fine_nodes, _, node_counter, edge_counter, fine_merges, fine_labels = _segment_pass(
            spans=coarse_spans,
            graph=graph,
            llm=llm,
            node_counter=node_counter,
            edge_counter=edge_counter,
            phase="fine",
            coarse_node=coarse_node,
            coarse_target_nodes=coarse_target_nodes,
            fine_target_nodes=fine_target_nodes,
            window_sec=fine_window_sec,
            overlap_sec=fine_overlap_sec,
            min_segment_sec=fine_min_segment_sec,
            max_segment_sec=fine_max_segment_sec,
            merge_every_n_windows=merge_every_n_windows,
            verbose=verbose,
        )
        merges.extend(fine_merges)
        weak_labels["fine"][coarse_node.id] = fine_labels
        if not [n for n in graph.nodes if n.layer == "fine" and n.parent_coarse_id == coarse_node.id]:
            # fallback child when fine pass finds no split-worthy subtopics
            node_counter += 1
            payload = llm.summarize_segment_to_node(
                "\n".join(f"[{s.timestamp:.1f}s][{s.modality}] {s.text}" for s in coarse_spans),
                [s.span_id for s in coarse_spans],
                coarse_node.start_ts,
                coarse_node.end_ts,
            )
            node = GraphNode(
                id=f"node_{node_counter}",
                lecture_id=lecture_id,
                title=payload["title"],
                summary=payload["summary"],
                explanation=payload["explanation"],
                source_span_ids=payload["source_span_ids"],
                start_ts=coarse_node.start_ts,
                end_ts=coarse_node.end_ts,
                layer="fine",
                parent_coarse_id=coarse_node.id,
            )
            graph.nodes.append(node)
            edge_counter += 1
            graph.edges.append(
                GraphEdge(
                    id=f"edge_{edge_counter}",
                    from_id=node.id,
                    to_id=coarse_node.id,
                    type="part_of",
                    reason="fallback single fine node for coarse topic",
                    evidence_span_ids=node.source_span_ids[:3],
                    edge_confidence=0.9,
                    evidence_count=min(3, len(node.source_span_ids)),
                )
            )

    cleanup_graph_v2_edges(graph)
    ensure_node_embeddings(graph, llm)
    validation = validate_graph_v2(graph, fine_min_segment_sec=fine_min_segment_sec, coarse_min_segment_sec=coarse_min_segment_sec)
    print_validation_summary(validation)
    graph.metadata["validation"] = {
        "ok": validation["ok"],
        "error_count": validation["error_count"],
        "warning_count": validation["warning_count"],
    }
    graph.metadata["weak_label_counts"] = {
        "coarse": {
            "accepted": len(weak_labels["coarse"]["accepted"]),
            "rejected": len(weak_labels["coarse"]["rejected"]),
        },
        "fine": {
            coarse_id: {"accepted": len(labels["accepted"]), "rejected": len(labels["rejected"])}
            for coarse_id, labels in weak_labels["fine"].items()
        },
    }
    layers = SegmentationLayers(
        coarse_boundaries=sorted({n.end_ts for n in graph.nodes if n.layer == "coarse"})[:-1],
        fine_boundaries=sorted({n.end_ts for n in graph.nodes if n.layer == "fine"})[:-1],
        calibration_state={
            "coarse_target_nodes": coarse_target_nodes,
            "fine_target_nodes": fine_target_nodes,
        },
    )
    graph.metadata["segmentation_layers"] = asdict(layers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2)

    merge_path = output_path.with_name("graph_v2_merge_log.json")
    merge_path.write_text(json.dumps(merges, indent=2), encoding="utf-8")
    weak_path = output_path.with_name("graph_v2_weak_labels.json")
    weak_path.write_text(json.dumps(weak_labels, indent=2), encoding="utf-8")
    validation_path = output_path.with_name("graph_v2_validation.json")
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    render_graph_visuals(graph, output_path.parent / "visuals")
    print(f"[done] nodes={len(graph.nodes)} edges={len(graph.edges)} merges={len(merges)}")
    return graph, merges, validation, weak_labels


def render_graph_visuals(graph: Graph, visuals_dir: Path) -> None:
    visuals_dir.mkdir(parents=True, exist_ok=True)
    graph_to_dot(graph, visuals_dir / "graph_v2.dot")
    if shutil.which("dot"):
        subprocess.run([shutil.which("dot") or "dot", "-Tpng", str(visuals_dir / "graph_v2.dot"), "-o", str(visuals_dir / "graph_v2.png")], check=False)
        graph_to_dot(Graph(lecture_id=graph.lecture_id, nodes=[n for n in graph.nodes if n.layer == "coarse"], edges=[e for e in graph.edges if e.from_id in {n.id for n in graph.nodes if n.layer == 'coarse'} and e.to_id in {n.id for n in graph.nodes if n.layer == 'coarse'}], spans=[], metadata={}), visuals_dir / "graph_v2_coarse.dot")
        subprocess.run([shutil.which("dot") or "dot", "-Tpng", str(visuals_dir / "graph_v2_coarse.dot"), "-o", str(visuals_dir / "graph_v2_coarse.png")], check=False)


def graph_to_dot(graph: Graph, path: Path) -> None:
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  graph [bgcolor="white", splines=true, overlap=false];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9, color="#666666"];',
    ]
    for n in graph.nodes:
        fill = "#eaf4ff" if n.layer == "coarse" else "#eef8ea"
        border = "#5b8def" if n.layer == "coarse" else "#5a9b5a"
        label = f"{n.id}\\n{n.title}\\n[{int(n.start_ts)}-{int(n.end_ts)}s]\\n{n.layer}"
        lines.append(f'  "{esc(n.id)}" [label="{esc(label)}", fillcolor="{fill}", color="{border}"];')
    for e in graph.edges:
        color = {
            "part_of": "#2d6cdf",
            "requires": "#d97706",
            "related_to": "#6b7280",
            "example_of": "#059669",
            "references": "#7c3aed",
        }.get(e.type, "#666666")
        lines.append(f'  "{esc(e.from_id)}" -> "{esc(e.to_id)}" [label="{esc(e.type)}", color="{color}"];')
    lines.append("}")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Graph V2 coarse-first lecture graph")
    p.add_argument("--audio", required=True)
    p.add_argument("--video")
    p.add_argument("--output", required=True)
    p.add_argument("--coarse-window-sec", type=int, default=240)
    p.add_argument("--coarse-overlap-sec", type=int, default=60)
    p.add_argument("--fine-window-sec", type=int, default=90)
    p.add_argument("--fine-overlap-sec", type=int, default=20)
    p.add_argument("--merge-every-n", type=int, default=1)
    p.add_argument("--coarse-min-segment-sec", type=float, default=180.0)
    p.add_argument("--coarse-max-segment-sec", type=float, default=900.0)
    p.add_argument("--fine-min-segment-sec", type=float, default=25.0)
    p.add_argument("--fine-max-segment-sec", type=float, default=240.0)
    p.add_argument("--coarse-target-nodes", type=int, default=12)
    p.add_argument("--fine-target-nodes", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_graph_v2(
        audio_path=Path(args.audio),
        video_path=Path(args.video) if args.video else None,
        output_path=Path(args.output),
        coarse_window_sec=args.coarse_window_sec,
        coarse_overlap_sec=args.coarse_overlap_sec,
        fine_window_sec=args.fine_window_sec,
        fine_overlap_sec=args.fine_overlap_sec,
        merge_every_n_windows=args.merge_every_n,
        coarse_min_segment_sec=args.coarse_min_segment_sec,
        coarse_max_segment_sec=args.coarse_max_segment_sec,
        fine_min_segment_sec=args.fine_min_segment_sec,
        fine_max_segment_sec=args.fine_max_segment_sec,
        coarse_target_nodes=args.coarse_target_nodes,
        fine_target_nodes=args.fine_target_nodes,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
