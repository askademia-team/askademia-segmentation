from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .llm import LLMClient
from .models import Graph, GraphEdge, GraphNode, Span, Window, validate_edge_type
from .retrieval import ensure_node_embeddings
from .validate import print_validation_summary, validate_graph


def load_lecture_entries(audio_path: Path, video_path: Path | None = None) -> Tuple[str, List[Span]]:
    with audio_path.open("r", encoding="utf-8") as f:
        audio = json.load(f)

    lecture_id = str(audio.get("lecture_id") or audio.get("display_name") or audio_path.stem)
    spans: List[Span] = []
    for i, e in enumerate(audio.get("entries", [])):
        spans.append(
            Span(
                span_id=f"a_{i}",
                lecture_id=lecture_id,
                timestamp=float(e.get("timestamp", 0.0)),
                text=str(e.get("text", "")),
                modality="audio",
            )
        )

    if video_path and video_path.exists():
        with video_path.open("r", encoding="utf-8") as f:
            video = json.load(f)
        for i, e in enumerate(video.get("entries", [])):
            spans.append(
                Span(
                    span_id=f"v_{i}",
                    lecture_id=lecture_id,
                    timestamp=float(e.get("timestamp", 0.0)),
                    text=str(e.get("text", "")),
                    modality="video",
                )
            )

    spans.sort(key=lambda s: s.timestamp)
    return lecture_id, spans


def clean_text(text: str) -> str:
    text = "".join(ch if ch == "\n" or ord(ch) >= 32 else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_windows(spans: List[Span], lecture_id: str, window_sec: int = 75, overlap_sec: int = 15) -> List[Window]:
    if not spans:
        return []
    if window_sec < 60 or window_sec > 90:
        raise ValueError("window_sec must be between 60 and 90")
    step = max(1, window_sec - overlap_sec)
    start_ts = spans[0].timestamp
    end_ts = spans[-1].timestamp

    windows: List[Window] = []
    idx = 0
    current = start_ts
    while current <= end_ts:
        w_start = current
        w_end = current + window_sec
        subset = [s for s in spans if s.timestamp >= w_start and s.timestamp < w_end]
        if subset:
            text = "\n".join(f"[{s.timestamp:.1f}s][{s.modality}] {clean_text(s.text)}" for s in subset)
            windows.append(
                Window(
                    window_id=f"w_{idx}",
                    lecture_id=lecture_id,
                    start_ts=w_start,
                    end_ts=min(w_end, end_ts),
                    span_ids=[s.span_id for s in subset],
                    text=text,
                )
            )
            idx += 1
        current += step
    return windows


def lexical_similarity(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"[a-z0-9]+", a.lower()))
    b_tokens = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))


def _semantic_edge_allowed(from_node: GraphNode, to_node: GraphNode, edge_type: str) -> bool:
    if edge_type == "requires":
        return from_node.start_ts >= to_node.start_ts - 1.0
    if edge_type == "example_of":
        return (from_node.end_ts - from_node.start_ts) <= (to_node.end_ts - to_node.start_ts) + 60.0
    return True


def ensure_window_span_coverage(
    graph: Graph,
    window: Window,
    node_counter: int,
    similarity_threshold: float = 0.08,
) -> Tuple[int, bool]:
    """
    Ensure every span in the window is mapped to at least one node.
    Returns (possibly updated node_counter, created_bridge_node_flag).
    """
    window_span_ids = set(window.span_ids)
    if not window_span_ids:
        return node_counter, False

    covered = set()
    for n in graph.nodes:
        for sid in n.source_span_ids:
            if sid in window_span_ids:
                covered.add(sid)
    unmapped = window_span_ids - covered
    if not unmapped:
        return node_counter, False

    # Prefer attaching to a nearby/relevant existing node rather than creating a new concept node.
    candidate_nodes = graph.nodes[-30:] if graph.nodes else []
    best_node = None
    best_score = -1.0
    for n in candidate_nodes:
        score = lexical_similarity(window.text, f"{n.title} {n.summary} {n.explanation}")
        if score > best_score:
            best_score = score
            best_node = n

    if best_node is not None and best_score >= similarity_threshold:
        best_node.source_span_ids = sorted(set(best_node.source_span_ids + list(unmapped)))
        best_node.start_ts = min(best_node.start_ts, window.start_ts)
        best_node.end_ts = max(best_node.end_ts, window.end_ts)
        return node_counter, False

    # If we cannot reasonably anchor to an existing node, create a low-priority bridge node
    # so no span data is dropped.
    node_counter += 1
    bridge_id = f"node_{node_counter}"
    graph.nodes.append(
        GraphNode(
            id=bridge_id,
            lecture_id=graph.lecture_id,
            title="Context Continuation",
            summary="Supporting lecture context attached to nearby concepts.",
            explanation=(
                "This node preserves window content that did not form a standalone concept node "
                "but should remain retrievable as contextual evidence."
            ),
            source_span_ids=sorted(unmapped),
            start_ts=window.start_ts,
            end_ts=window.end_ts,
        )
    )
    return node_counter, True


def merge_graph_nodes(graph: Graph, llm: LLMClient, threshold: float = 0.38) -> List[Dict[str, Any]]:
    merges: List[Dict[str, Any]] = []
    nodes = sorted(graph.nodes, key=lambda n: (n.start_ts, n.end_ts, n.id))
    remap: Dict[str, str] = {}
    # Strict adjacency: only consecutive nodes in time-sorted order are merge-eligible.
    id_to_pos = {n.id: idx for idx, n in enumerate(nodes)}

    # LLM-driven merge groups over recent adjacent nodes.
    candidate = nodes[-12:]
    candidate_payload = [
        {
            "id": n.id,
            "title": n.title,
            "summary": n.summary,
            "explanation": n.explanation,
            "start_ts": n.start_ts,
            "end_ts": n.end_ts,
        }
        for n in candidate
    ]
    merge_groups = llm.propose_adjacent_merge_groups(candidate_payload)
    node_by_id = {n.id: n for n in nodes}
    for g in merge_groups:
        ids = g["node_ids"]
        # Conservative cap: avoid large aggressive cluster merges.
        if len(ids) > 3:
            continue
        # Enforce contiguous block in temporal order (allows A-B-A style with middle digression).
        positions = sorted(id_to_pos.get(i, -10**9) for i in ids)
        if not positions or any(p < 0 for p in positions):
            continue
        if positions != list(range(positions[0], positions[-1] + 1)):
            continue
        anchor = ids[0]
        if anchor in remap:
            anchor = remap[anchor]
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
            if anchor_node.node_kind != src_node.node_kind:
                continue
            # Conservative semantic gate for LLM-group merges.
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
            if anchor_node.layer == "fine" and src_node.parent_coarse_id and not anchor_node.parent_coarse_id:
                anchor_node.parent_coarse_id = src_node.parent_coarse_id
            anchor_node.embedding = None
            merges.append(
                {
                    "merged_from": src,
                    "merged_into": anchor,
                    "reason": f"llm_group(sim={sim_group:.3f}): {g['reason']}",
                }
            )

    for i in range(len(nodes)):
        ni = nodes[i]
        if ni.id in remap:
            continue
        # Strictly only the immediate next node in temporal order.
        for j in [i + 1]:
            if j >= len(nodes):
                continue
            nj = nodes[j]
            if nj.id in remap:
                continue
            if ni.layer != nj.layer:
                continue
            if ni.node_kind != nj.node_kind:
                continue
            sim = lexical_similarity(
                f"{ni.title} {ni.summary} {ni.explanation}",
                f"{nj.title} {nj.summary} {nj.explanation}",
            )
            if sim < threshold:
                continue
            should_merge = llm.choose_merge_candidate(asdict(ni), asdict(nj))
            if not should_merge:
                continue
            remap[nj.id] = ni.id
            ni.source_span_ids = sorted(set(ni.source_span_ids + nj.source_span_ids))
            ni.start_ts = min(ni.start_ts, nj.start_ts)
            ni.end_ts = max(ni.end_ts, nj.end_ts)
            ni.embedding = None
            merges.append({"merged_from": nj.id, "merged_into": ni.id, "reason": f"similarity={sim:.3f}"})

    if remap:
        graph.nodes = [n for n in nodes if n.id not in remap]
        for e in graph.edges:
            e.from_id = remap.get(e.from_id, e.from_id)
            e.to_id = remap.get(e.to_id, e.to_id)
    _cleanup_graph_edges(graph)

    return merges


def _cleanup_graph_edges(graph: Graph) -> None:
    """
    Remove broken/low-signal edges and dedupe edge fanout after merges.
    """
    node_by_id = {n.id: n for n in graph.nodes}
    valid_node_ids = set(node_by_id)
    cleaned: List[GraphEdge] = []
    for e in graph.edges:
        if e.from_id not in valid_node_ids or e.to_id not in valid_node_ids:
            continue
        if e.from_id == e.to_id:
            continue
        from_node = node_by_id[e.from_id]
        to_node = node_by_id[e.to_id]

        # Structural hierarchy rule:
        # - part_of is only allowed from a fine node to the coarse node that contains it
        # - all other semantic edges must stay within the same layer
        if e.type == "part_of":
            if from_node.layer != "fine" or to_node.layer != "coarse":
                continue
            if from_node.node_kind != "lecture" or to_node.node_kind != "lecture":
                continue
            if from_node.parent_coarse_id and from_node.parent_coarse_id != to_node.id:
                continue
            if not (to_node.start_ts <= from_node.start_ts and from_node.end_ts <= to_node.end_ts):
                continue
        else:
            if from_node.layer != to_node.layer:
                continue
            if from_node.node_kind != to_node.node_kind:
                continue
            if not _semantic_edge_allowed(from_node, to_node, e.type):
                continue
        cleaned.append(e)

    dedup: Dict[Tuple[str, str, str], GraphEdge] = {}
    for e in cleaned:
        key = (e.from_id, e.to_id, e.type)
        if key not in dedup:
            dedup[key] = e
            continue
        # Keep the edge with more evidence spans; merge evidence for stability.
        chosen = dedup[key]
        merged_evidence = sorted(set(chosen.evidence_span_ids + e.evidence_span_ids))
        if len(e.evidence_span_ids) > len(chosen.evidence_span_ids):
            e.evidence_span_ids = merged_evidence
            dedup[key] = e
        else:
            chosen.evidence_span_ids = merged_evidence

    graph.edges = list(dedup.values())


def build_graph_for_lecture(
    audio_path: Path,
    video_path: Path | None,
    output_path: Path,
    window_sec: int = 75,
    overlap_sec: int = 15,
    merge_every_n_windows: int = 10,
    show_progress: bool = True,
    verbose: bool = False,
) -> Tuple[Graph, List[Dict[str, Any]]]:
    lecture_id, spans = load_lecture_entries(audio_path, video_path)
    windows = build_windows(spans, lecture_id, window_sec=window_sec, overlap_sec=overlap_sec)
    llm = LLMClient()

    graph = Graph(
        lecture_id=lecture_id,
        spans=spans,
        metadata={
            "window_sec": window_sec,
            "overlap_sec": overlap_sec,
            "merge_every_n_windows": merge_every_n_windows,
            "source_audio": str(audio_path),
            "source_video": str(video_path) if video_path else None,
            "llm_available": llm.available(),
        },
    )

    merges: List[Dict[str, Any]] = []
    node_counter = 0
    edge_counter = 0

    total_windows = len(windows)
    for w_idx, w in enumerate(windows):
        if show_progress:
            pct = ((w_idx + 1) / max(1, total_windows)) * 100
            print(
                f"[window {w_idx + 1}/{total_windows} | {pct:5.1f}%] "
                f"{w.start_ts:7.1f}s -> {w.end_ts:7.1f}s | "
                f"nodes={len(graph.nodes)} edges={len(graph.edges)}"
            )
        existing_node_payload = [{"id": n.id, "title": n.title, "summary": n.summary} for n in graph.nodes]
        extracted = llm.extract_nodes_and_edges(
            window_text=w.text,
            window_id=w.window_id,
            span_ids=w.span_ids,
            existing_nodes=existing_node_payload,
            max_nodes=3,
        )

        temp_to_real: Dict[str, str] = {}
        created_node_titles: List[str] = []
        for n in extracted.get("nodes", []):
            node_counter += 1
            node_id = f"node_{node_counter}"
            temp_to_real[n["temp_id"]] = node_id
            if n.get("start_ts", 0.0) <= 0:
                start_ts = w.start_ts
                end_ts = w.end_ts
            else:
                start_ts = max(w.start_ts, float(n["start_ts"]))
                end_ts = min(w.end_ts, float(n["end_ts"]))
            graph.nodes.append(
                GraphNode(
                    id=node_id,
                    lecture_id=lecture_id,
                    title=n["title"],
                    summary=n["summary"],
                    explanation=n["explanation"],
                    source_span_ids=n["source_span_ids"],
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
            )
            created_node_titles.append(n["title"])

        valid_node_ids = {n.id for n in graph.nodes}
        for e in extracted.get("edges", []):
            from_id = temp_to_real.get(e["from_temp_or_existing"], e["from_temp_or_existing"])
            to_id = temp_to_real.get(e["to_temp_or_existing"], e["to_temp_or_existing"])
            if from_id not in valid_node_ids or to_id not in valid_node_ids:
                continue
            if not validate_edge_type(e["type"]):
                continue
            edge_counter += 1
            graph.edges.append(
                GraphEdge(
                    id=f"edge_{edge_counter}",
                    from_id=from_id,
                    to_id=to_id,
                    type=e["type"],
                    reason=e["reason"],
                    evidence_span_ids=e["evidence_span_ids"],
                )
            )

        if show_progress:
            print(
                f"  + created_nodes={len(created_node_titles)} "
                f"created_edges={len(extracted.get('edges', []))}"
            )
            if verbose and created_node_titles:
                for title in created_node_titles:
                    print(f"    - {title}")

        # Coverage guarantee: every span in this window must map to some node.
        prev_node_count = len(graph.nodes)
        node_counter, created_bridge = ensure_window_span_coverage(graph, w, node_counter)
        if show_progress:
            mapped_now = len(graph.nodes) - prev_node_count
            if created_bridge:
                print("  * coverage_backfill: created_bridge_node=1")
            elif mapped_now == 0:
                print("  * coverage_backfill: attached_unmapped_spans_to_existing_node")

        if (w_idx + 1) % merge_every_n_windows == 0:
            new_merges = merge_graph_nodes(graph, llm)
            merges.extend(new_merges)
            if show_progress and new_merges:
                print(f"  * merge_pass: merged={len(new_merges)}")

    final_merges = merge_graph_nodes(graph, llm)
    merges.extend(final_merges)
    if show_progress:
        print(
            f"[done] windows={total_windows} nodes={len(graph.nodes)} "
            f"edges={len(graph.edges)} total_merges={len(merges)}"
        )

    ensure_node_embeddings(graph, llm)
    validation = validate_graph(graph)
    print_validation_summary(validation)
    graph.metadata["validation"] = {
        "ok": validation["ok"],
        "error_count": validation["error_count"],
        "warning_count": validation["warning_count"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2)

    merge_log = output_path.with_name(output_path.stem + "_merge_log.json")
    with merge_log.open("w", encoding="utf-8") as f:
        json.dump(merges, f, indent=2)

    validation_log = output_path.with_name(output_path.stem + "_validation.json")
    with validation_log.open("w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2)

    return graph, merges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build single-lecture concept graph (V1).")
    parser.add_argument("--audio", required=True, help="Path to audio lecture JSON file")
    parser.add_argument("--video", required=False, help="Path to video lecture JSON file")
    parser.add_argument("--output", required=True, help="Output graph JSON path")
    parser.add_argument("--window-sec", type=int, default=75)
    parser.add_argument("--overlap-sec", type=int, default=15)
    parser.add_argument("--merge-every-n", type=int, default=10)
    parser.add_argument("--no-progress", action="store_true", help="Disable progress logging")
    parser.add_argument("--verbose", action="store_true", help="Print created node titles per window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph, merges = build_graph_for_lecture(
        audio_path=Path(args.audio),
        video_path=Path(args.video) if args.video else None,
        output_path=Path(args.output),
        window_sec=args.window_sec,
        overlap_sec=args.overlap_sec,
        merge_every_n_windows=args.merge_every_n,
        show_progress=not args.no_progress,
        verbose=args.verbose,
    )
    print(
        json.dumps(
            {
                "lecture_id": graph.lecture_id,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "spans": len(graph.spans),
                "merges": len(merges),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
