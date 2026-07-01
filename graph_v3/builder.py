from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from statistics import quantiles
from typing import Any, Dict, List, Sequence, Tuple

from graph_v1.models import Graph, GraphEdge, GraphNode, SegmentationLayers, Span

from .llm import GraphV3LLMClient
from .validate import validate_graph_v3, print_validation_summary_v3


def load_lecture_entries(audio_path: Path, video_path: Path | None = None) -> Tuple[str, List[Span]]:
    with audio_path.open("r", encoding="utf-8") as f:
        audio = json.load(f)

    lecture_id = str(audio.get("lecture_id") or audio.get("display_name") or audio_path.stem)
    spans: List[Span] = []
    for i, entry in enumerate(audio.get("entries", [])):
        spans.append(
            Span(
                span_id=f"a_{i}",
                lecture_id=lecture_id,
                timestamp=float(entry.get("timestamp", 0.0)),
                text=str(entry.get("text", "")),
                modality="audio",
            )
        )

    if video_path and video_path.exists():
        with video_path.open("r", encoding="utf-8") as f:
            video = json.load(f)
        for i, entry in enumerate(video.get("entries", [])):
            spans.append(
                Span(
                    span_id=f"v_{i}",
                    lecture_id=lecture_id,
                    timestamp=float(entry.get("timestamp", 0.0)),
                    text=str(entry.get("text", "")),
                    modality="video",
                )
            )

    spans.sort(key=lambda s: (s.timestamp, s.span_id))
    return lecture_id, spans


def spans_in_range(spans: Sequence[Span], start_ts: float, end_ts: float, *, include_end: bool = False) -> List[Span]:
    if include_end:
        return [s for s in spans if s.timestamp >= start_ts and s.timestamp <= end_ts]
    return [s for s in spans if s.timestamp >= start_ts and s.timestamp < end_ts]


def _tokenize(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _lexical_shift(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / max(1, len(ta | tb))
    return 1.0 - overlap


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def _mean_vec(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    dims = len(vectors[0])
    sums = [0.0] * dims
    for vec in vectors:
        if len(vec) != dims:
            continue
        for i, val in enumerate(vec):
            sums[i] += float(val)
    denom = float(max(1, len(vectors)))
    return [v / denom for v in sums]


def _smooth_scores(scores: List[float], width: int) -> List[float]:
    if width <= 1 or not scores:
        return list(scores)
    out: List[float] = []
    radius = max(0, width // 2)
    for i in range(len(scores)):
        lo = max(0, i - radius)
        hi = min(len(scores), i + radius + 1)
        window = scores[lo:hi]
        out.append(sum(window) / max(1, len(window)))
    return out


def _build_windows(spans: Sequence[Span], *, window_sec: int, overlap_sec: int, phase: str) -> List[Tuple[str, float, float, List[Span]]]:
    if not spans:
        return []
    start_ts = spans[0].timestamp
    end_ts = spans[-1].timestamp
    step = max(1, window_sec - overlap_sec)
    windows: List[Tuple[str, float, float, List[Span]]] = []
    cur = start_ts
    idx = 0
    while cur <= end_ts:
        w_start = cur
        w_end = min(end_ts, cur + window_sec)
        subset = spans_in_range(spans, w_start, w_end)
        if subset:
            windows.append((f"{phase}_w_{idx}", w_start, w_end, subset))
            idx += 1
        cur += step
    return windows


def _compute_window_candidates(
    *,
    spans: Sequence[Span],
    llm: GraphV3LLMClient,
    window_id: str,
    smooth_width: int,
    candidate_peak_percentile: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if len(spans) < 4:
        return [], []

    texts = [f"[{s.modality}] {s.text}" for s in spans]
    embs = llm.embed_texts(texts)

    rows: List[Dict[str, Any]] = []
    for i in range(len(spans) - 1):
        left_lo = max(0, i - 2)
        left_hi = i + 1
        right_lo = i + 1
        right_hi = min(len(spans), i + 4)

        left_emb = _mean_vec(embs[left_lo:left_hi])
        right_emb = _mean_vec(embs[right_lo:right_hi])
        emb_shift = 1.0 - _cosine(left_emb, right_emb)

        left_text = " ".join(texts[left_lo:left_hi])
        right_text = " ".join(texts[right_lo:right_hi])
        lex_shift = _lexical_shift(left_text, right_text)

        raw_score = 0.72 * emb_shift + 0.28 * lex_shift
        rows.append(
            {
                "window_id": window_id,
                "timestamp": spans[i + 1].timestamp,
                "raw_score": raw_score,
                "embedding_shift": emb_shift,
                "lexical_shift": lex_shift,
                "left_text": left_text,
                "right_text": right_text,
            }
        )

    smoothed = _smooth_scores([r["raw_score"] for r in rows], smooth_width)
    for row, score in zip(rows, smoothed):
        row["smoothed_score"] = score

    values = [r["smoothed_score"] for r in rows]
    if len(values) >= 3:
        try:
            # n=100 gives percentile granularity by rank approximation.
            pct_grid = quantiles(values, n=100)
            pct = max(1, min(99, int(candidate_peak_percentile)))
            threshold = pct_grid[pct - 1]
        except Exception:
            idx = int(max(0, min(len(values) - 1, round((candidate_peak_percentile / 100.0) * (len(values) - 1)))))
            threshold = sorted(values)[idx]
    else:
        threshold = max(values) if values else 0.0

    candidates: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        prev_score = rows[i - 1]["smoothed_score"] if i > 0 else -1.0
        next_score = rows[i + 1]["smoothed_score"] if i + 1 < len(rows) else -1.0
        if row["smoothed_score"] < threshold:
            continue
        if row["smoothed_score"] + 1e-9 < prev_score:
            continue
        if row["smoothed_score"] + 1e-9 < next_score:
            continue
        candidates.append(
            {
                "window_id": window_id,
                "timestamp": row["timestamp"],
                "score": row["smoothed_score"],
                "embedding_shift": row["embedding_shift"],
                "lexical_shift": row["lexical_shift"],
                "left_text": row["left_text"],
                "right_text": row["right_text"],
                "reason": "local_peak",
            }
        )

    return rows, candidates


def _dedupe_candidates(candidates: Sequence[Dict[str, Any]], tolerance_sec: float = 2.0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: float(c["timestamp"])):
        if not out:
            out.append(dict(cand))
            continue
        if abs(float(cand["timestamp"]) - float(out[-1]["timestamp"])) <= tolerance_sec:
            if float(cand["score"]) > float(out[-1]["score"]):
                out[-1] = dict(cand)
        else:
            out.append(dict(cand))
    return out


def _uniform_fallback_boundaries(
    *,
    spans: Sequence[Span],
    start_ts: float,
    end_ts: float,
    boundary_count: int,
) -> List[float]:
    if boundary_count <= 0 or not spans:
        return []
    if end_ts <= start_ts:
        return []
    valid = sorted({s.timestamp for s in spans if start_ts < s.timestamp < end_ts})
    if not valid:
        return []

    out: List[float] = []
    for i in range(1, boundary_count + 1):
        target = start_ts + ((end_ts - start_ts) * (i / float(boundary_count + 1)))
        nearest = min(valid, key=lambda t: abs(t - target))
        out.append(nearest)
    return sorted(set(out))


def _segments_from_boundaries(start_ts: float, end_ts: float, boundaries: Sequence[float]) -> List[Tuple[float, float]]:
    ordered = [start_ts] + sorted([b for b in boundaries if start_ts < b < end_ts]) + [end_ts]
    segments: List[Tuple[float, float]] = []
    for left, right in zip(ordered, ordered[1:]):
        if right > left:
            segments.append((left, right))
    return segments


def _select_boundaries_constrained(
    *,
    start_ts: float,
    end_ts: float,
    candidates: Sequence[Dict[str, Any]],
    spans: Sequence[Span],
    min_segment_sec: float,
    max_segment_sec: float,
    target_nodes: int,
    duration_penalty: float,
) -> Tuple[List[float], Dict[str, Any]]:
    duration = max(0.0, end_ts - start_ts)
    if duration <= 0:
        return [], {"strategy": "empty", "selected_boundary_count": 0}

    min_segments = max(1, int(math.ceil(duration / max(1.0, max_segment_sec))))
    max_segments = max(1, int(math.floor(duration / max(1.0, min_segment_sec))))
    if min_segments > max_segments:
        min_segments = max_segments

    target_segments = max(min_segments, min(max_segments, int(max(1, target_nodes))))
    target_boundaries = max(0, target_segments - 1)

    cand_sorted = sorted([c for c in candidates if start_ts < float(c["timestamp"]) < end_ts], key=lambda c: float(c["timestamp"]))
    if not cand_sorted:
        fallback = _uniform_fallback_boundaries(
            spans=spans,
            start_ts=start_ts,
            end_ts=end_ts,
            boundary_count=target_boundaries,
        )
        return fallback, {
            "strategy": "uniform_fallback_no_candidates",
            "target_segments": target_segments,
            "selected_boundary_count": len(fallback),
        }

    times = [start_ts] + [float(c["timestamp"]) for c in cand_sorted] + [end_ts]
    is_boundary = [False] + [True] * len(cand_sorted) + [False]
    boundary_scores = [0.0] + [float(c["score"]) for c in cand_sorted] + [0.0]

    m = len(times) - 2
    boundary_min = max(0, min_segments - 1)
    boundary_max = min(m, max_segments - 1)
    target_seg_dur = duration / float(max(1, target_segments))

    neg_inf = -10**18
    dp: List[List[float]] = [[neg_inf for _ in range(len(times))] for _ in range(boundary_max + 1)]
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    dp[0][0] = 0.0

    for used in range(boundary_max + 1):
        for prev_idx in range(len(times) - 1):
            cur_score = dp[used][prev_idx]
            if cur_score <= neg_inf / 2:
                continue
            for cur_idx in range(prev_idx + 1, len(times)):
                seg_dur = times[cur_idx] - times[prev_idx]
                if seg_dur < min_segment_sec - 1e-6 or seg_dur > max_segment_sec + 1e-6:
                    continue
                if cur_idx == len(times) - 1:
                    new_used = used
                else:
                    if used >= boundary_max:
                        continue
                    new_used = used + 1
                dur_cost = duration_penalty * abs(seg_dur - target_seg_dur) / max(1.0, target_seg_dur)
                step_score = -dur_cost + (boundary_scores[cur_idx] if cur_idx != len(times) - 1 else 0.0)
                nxt = cur_score + step_score
                if nxt > dp[new_used][cur_idx]:
                    dp[new_used][cur_idx] = nxt
                    parent[(new_used, cur_idx)] = (used, prev_idx)

    best_state: Tuple[int, int] | None = None
    best_value = neg_inf
    end_idx = len(times) - 1
    for used in range(boundary_min, boundary_max + 1):
        score = dp[used][end_idx]
        if score <= neg_inf / 2:
            continue
        target_penalty = 0.45 * abs(used - target_boundaries)
        total = score - target_penalty
        if total > best_value:
            best_value = total
            best_state = (used, end_idx)

    if best_state is None:
        fallback = _uniform_fallback_boundaries(
            spans=spans,
            start_ts=start_ts,
            end_ts=end_ts,
            boundary_count=target_boundaries,
        )
        return fallback, {
            "strategy": "uniform_fallback_no_feasible_dp",
            "target_segments": target_segments,
            "selected_boundary_count": len(fallback),
        }

    used, idx = best_state
    chosen_indices: List[int] = []
    while (used, idx) in parent:
        prev_used, prev_idx = parent[(used, idx)]
        if idx != end_idx and is_boundary[idx]:
            chosen_indices.append(idx)
        used, idx = prev_used, prev_idx
        if idx == 0 and used == 0:
            break

    selected = sorted(times[i] for i in chosen_indices)
    return selected, {
        "strategy": "dp_constrained",
        "target_segments": target_segments,
        "selected_boundary_count": len(selected),
        "candidate_count": len(cand_sorted),
    }


def _repair_segments(
    *,
    segments: List[Tuple[float, float]],
    container_start: float,
    container_end: float,
    min_segment_sec: float,
    max_segment_sec: float,
) -> Tuple[List[Tuple[float, float]], List[Dict[str, Any]]]:
    if not segments:
        return [(container_start, container_end)], [{"action": "create_single_segment"}]

    repaired = sorted(segments, key=lambda x: x[0])
    actions: List[Dict[str, Any]] = []

    first = repaired[0]
    if abs(first[0] - container_start) > 1e-6:
        actions.append({"action": "snap_first_start", "from": first[0], "to": container_start})
        repaired[0] = (container_start, first[1])

    last = repaired[-1]
    if abs(last[1] - container_end) > 1e-6:
        actions.append({"action": "snap_last_end", "from": last[1], "to": container_end})
        repaired[-1] = (last[0], container_end)

    # force exact adjacency to remove gaps/overlaps
    for i in range(1, len(repaired)):
        prev = repaired[i - 1]
        cur = repaired[i]
        if abs(prev[1] - cur[0]) > 1e-6:
            boundary = (prev[1] + cur[0]) / 2.0
            actions.append(
                {
                    "action": "midpoint_realign_boundary",
                    "index": i,
                    "prev_end": prev[1],
                    "cur_start": cur[0],
                    "new_boundary": boundary,
                }
            )
            repaired[i - 1] = (prev[0], boundary)
            repaired[i] = (boundary, cur[1])

    # merge short segments
    changed = True
    while changed and len(repaired) > 1:
        changed = False
        for i, (start, end) in enumerate(list(repaired)):
            if end - start >= min_segment_sec - 1e-6:
                continue
            if i == 0:
                new_segment = (repaired[0][0], repaired[1][1])
                actions.append({"action": "merge_short_with_next", "index": 0, "duration": end - start})
                repaired = [new_segment] + repaired[2:]
            elif i == len(repaired) - 1:
                new_segment = (repaired[-2][0], repaired[-1][1])
                actions.append({"action": "merge_short_with_prev", "index": i, "duration": end - start})
                repaired = repaired[:-2] + [new_segment]
            else:
                left_dur = repaired[i][1] - repaired[i - 1][0]
                right_dur = repaired[i + 1][1] - repaired[i][0]
                if left_dur <= right_dur:
                    new_segment = (repaired[i - 1][0], repaired[i][1])
                    actions.append({"action": "merge_short_with_prev", "index": i, "duration": end - start})
                    repaired = repaired[: i - 1] + [new_segment] + repaired[i + 1 :]
                else:
                    new_segment = (repaired[i][0], repaired[i + 1][1])
                    actions.append({"action": "merge_short_with_next", "index": i, "duration": end - start})
                    repaired = repaired[:i] + [new_segment] + repaired[i + 2 :]
            changed = True
            break

    # split long segments
    out: List[Tuple[float, float]] = []
    for idx, (start, end) in enumerate(repaired):
        dur = end - start
        if dur <= max_segment_sec + 1e-6:
            out.append((start, end))
            continue
        pieces = max(2, int(math.ceil(dur / max(1.0, max_segment_sec))))
        actions.append({"action": "split_long_segment", "index": idx, "duration": dur, "pieces": pieces})
        prev = start
        for p in range(1, pieces + 1):
            nxt = end if p == pieces else (start + (dur * (p / pieces)))
            out.append((prev, nxt))
            prev = nxt

    # final adjacency cleanup
    out = sorted(out, key=lambda x: x[0])
    if out:
        out[0] = (container_start, out[0][1])
        out[-1] = (out[-1][0], container_end)
    for i in range(1, len(out)):
        out[i - 1] = (out[i - 1][0], out[i][0])

    return out, actions


def _next_edge_id(graph: Graph) -> int:
    nums: List[int] = []
    for edge in graph.edges:
        if edge.id.startswith("edge_"):
            try:
                nums.append(int(edge.id.split("_", 1)[1]))
            except Exception:
                pass
    return max(nums, default=0) + 1


def _cleanup_edges_v3(graph: Graph) -> None:
    node_by_id = {n.id: n for n in graph.nodes}
    kept: List[GraphEdge] = []
    for edge in graph.edges:
        src = node_by_id.get(edge.from_id)
        dst = node_by_id.get(edge.to_id)
        if not src or not dst:
            continue
        if edge.from_id == edge.to_id:
            continue

        if edge.type == "part_of":
            if src.layer != "fine" or dst.layer != "coarse":
                continue
            if src.parent_coarse_id != dst.id:
                continue
            if src.start_ts < dst.start_ts - 1e-6 or src.end_ts > dst.end_ts + 1e-6:
                continue
        else:
            if src.layer != dst.layer:
                continue
            if src.layer == "fine" and src.parent_coarse_id != dst.parent_coarse_id:
                continue
            if edge.edge_confidence < 0.65:
                continue
            if not edge.evidence_span_ids:
                continue
        kept.append(edge)

    dedup: Dict[Tuple[str, str, str], GraphEdge] = {}
    for edge in kept:
        key = (edge.from_id, edge.to_id, edge.type)
        cur = dedup.get(key)
        if cur is None or edge.edge_confidence > cur.edge_confidence:
            dedup[key] = edge
        else:
            merged = sorted(set(cur.evidence_span_ids + edge.evidence_span_ids))
            cur.evidence_span_ids = merged
            cur.evidence_count = len(merged)

    graph.edges = list(dedup.values())
    for i, edge in enumerate(graph.edges, start=1):
        edge.id = f"edge_{i}"


def _create_nodes_for_segments(
    *,
    graph: Graph,
    llm: GraphV3LLMClient,
    spans: Sequence[Span],
    segments: Sequence[Tuple[float, float]],
    phase: str,
    coarse_parent: GraphNode | None,
    edge_trace: List[Dict[str, Any]],
) -> List[GraphNode]:
    created: List[GraphNode] = []
    next_node_num = max((int(n.id.split("_", 1)[1]) for n in graph.nodes if n.id.startswith("node_") and n.id.split("_", 1)[1].isdigit()), default=0) + 1
    next_edge_num = _next_edge_id(graph)

    for start_ts, end_ts in segments:
        seg_spans = spans_in_range(spans, start_ts, end_ts, include_end=True)
        if not seg_spans:
            continue
        source_ids = [s.span_id for s in seg_spans]
        seg_text = "\n".join(f"[{s.timestamp:.1f}s][{s.modality}] {s.text}" for s in seg_spans)
        payload = llm.summarize_segment_to_node(
            segment_text=seg_text,
            source_span_ids=source_ids,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        node = GraphNode(
            id=f"node_{next_node_num}",
            lecture_id=graph.lecture_id,
            title=payload["title"],
            summary=payload["summary"],
            explanation=payload["explanation"],
            source_span_ids=payload["source_span_ids"],
            start_ts=start_ts,
            end_ts=end_ts,
            layer=phase,
            parent_coarse_id=coarse_parent.id if coarse_parent and phase == "fine" else None,
        )
        next_node_num += 1
        graph.nodes.append(node)
        created.append(node)

        if phase == "fine" and coarse_parent is not None:
            graph.edges.append(
                GraphEdge(
                    id=f"edge_{next_edge_num}",
                    from_id=node.id,
                    to_id=coarse_parent.id,
                    type="part_of",
                    reason="fine node belongs to coarse parent",
                    evidence_span_ids=node.source_span_ids[:3],
                    edge_confidence=0.95,
                    evidence_count=min(3, len(node.source_span_ids)),
                )
            )
            next_edge_num += 1

        existing_same_layer = [
            {
                "id": n.id,
                "title": n.title,
                "summary": n.summary,
                "explanation": n.explanation,
                "parent_coarse_id": n.parent_coarse_id,
            }
            for n in graph.nodes
            if n.id != node.id and n.layer == phase and (phase != "fine" or n.parent_coarse_id == node.parent_coarse_id)
        ]
        edge_candidates = llm.propose_same_layer_edges(
            phase=phase,
            new_node={
                "id": node.id,
                "title": node.title,
                "summary": node.summary,
                "explanation": node.explanation,
            },
            existing_nodes=existing_same_layer,
            valid_span_ids=node.source_span_ids,
        )
        valid_ids = {n.id for n in graph.nodes}
        for cand in edge_candidates:
            tgt = str(cand.get("to_existing") or "")
            if tgt not in valid_ids:
                continue
            conf = float(cand.get("confidence", 0.0))
            if conf < 0.65:
                continue
            evidence = [sid for sid in cand.get("evidence_span_ids", []) if sid in set(node.source_span_ids)]
            if not evidence:
                continue
            edge = GraphEdge(
                id=f"edge_{next_edge_num}",
                from_id=node.id,
                to_id=tgt,
                type=str(cand.get("type") or "related_to"),
                reason=str(cand.get("reason") or "v3 semantic edge")[:240],
                evidence_span_ids=evidence[:3],
                edge_confidence=conf,
                evidence_count=len(evidence[:3]),
            )
            graph.edges.append(edge)
            edge_trace.append({"phase": phase, "from": node.id, "to": tgt, "type": edge.type, "confidence": conf})
            next_edge_num += 1

    return created


def _run_phase(
    *,
    graph: Graph,
    llm: GraphV3LLMClient,
    spans: Sequence[Span],
    phase: str,
    coarse_parent: GraphNode | None,
    window_sec: int,
    overlap_sec: int,
    min_segment_sec: float,
    max_segment_sec: float,
    target_nodes: int,
    smooth_width: int,
    candidate_peak_percentile: int,
    llm_gate_threshold: float,
    duration_penalty: float,
    verbose: bool,
    edge_trace: List[Dict[str, Any]],
) -> Tuple[List[GraphNode], Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    if not spans:
        return [], {"phase": phase, "windows": []}, {"accepted": [], "rejected": []}

    container_start = spans[0].timestamp
    container_end = spans[-1].timestamp
    windows = _build_windows(spans, window_sec=window_sec, overlap_sec=overlap_sec, phase=phase)

    score_rows: List[Dict[str, Any]] = []
    raw_candidates: List[Dict[str, Any]] = []
    for window_id, _, _, win_spans in windows:
        rows, cands = _compute_window_candidates(
            spans=win_spans,
            llm=llm,
            window_id=window_id,
            smooth_width=smooth_width,
            candidate_peak_percentile=candidate_peak_percentile,
        )
        score_rows.extend(rows)
        raw_candidates.extend(cands)

    deduped = _dedupe_candidates(raw_candidates)
    gated: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for cand in deduped:
        gate = llm.gate_transition(
            phase=phase,
            candidate_ts=float(cand["timestamp"]),
            candidate_score=float(cand["score"]),
            left_text=str(cand.get("left_text") or ""),
            right_text=str(cand.get("right_text") or ""),
        )
        gated_score = float(cand["score"]) * (0.65 + 0.35 * float(gate["confidence"]))
        row = {
            **cand,
            "gate": gate,
            "gated_score": gated_score,
            "phase": phase,
            "coarse_parent": coarse_parent.id if coarse_parent else None,
        }
        gated.append(row)
        if not gate["keep"]:
            rejected.append({"candidate": row, "reason": f"gate_reject:{gate['rationale_tag']}"})
            continue
        if float(gate["confidence"]) < llm_gate_threshold:
            rejected.append({"candidate": row, "reason": "gate_confidence_below_threshold"})
            continue
        if float(cand["timestamp"]) - container_start < min_segment_sec:
            rejected.append({"candidate": row, "reason": "too_close_to_start"})
            continue
        if container_end - float(cand["timestamp"]) < min_segment_sec:
            rejected.append({"candidate": row, "reason": "too_close_to_end"})
            continue
        accepted.append({"candidate": row, "reason": "accepted"})

    selected_boundaries, selector_meta = _select_boundaries_constrained(
        start_ts=container_start,
        end_ts=container_end,
        candidates=[
            {
                "timestamp": x["candidate"]["timestamp"],
                "score": x["candidate"]["gated_score"],
            }
            for x in accepted
        ],
        spans=spans,
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
        target_nodes=target_nodes,
        duration_penalty=duration_penalty,
    )

    segments_before = _segments_from_boundaries(container_start, container_end, selected_boundaries)
    repaired_segments, repair_actions = _repair_segments(
        segments=segments_before,
        container_start=container_start,
        container_end=container_end,
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
    )

    created_nodes = _create_nodes_for_segments(
        graph=graph,
        llm=llm,
        spans=spans,
        segments=repaired_segments,
        phase=phase,
        coarse_parent=coarse_parent,
        edge_trace=edge_trace,
    )

    if verbose:
        label = f"[{phase}]" if not coarse_parent else f"[{phase}:{coarse_parent.id}]"
        print(
            f"{label} windows={len(windows)} raw_candidates={len(raw_candidates)} deduped={len(deduped)} "
            f"accepted={len(accepted)} selected={len(selected_boundaries)} nodes={len(created_nodes)}"
        )

    trace = {
        "phase": phase,
        "coarse_parent": coarse_parent.id if coarse_parent else None,
        "window_count": len(windows),
        "window_scores": score_rows,
        "raw_candidates": raw_candidates,
        "deduped_candidates": deduped,
        "gated_candidates": gated,
        "selector": selector_meta,
        "selected_boundaries": selected_boundaries,
        "segments_before_repair": segments_before,
        "segments_after_repair": repaired_segments,
        "repair_actions": repair_actions,
    }
    weak = {"accepted": accepted, "rejected": rejected}
    return created_nodes, trace, weak


def _repair_fine_within_coarse(graph: Graph) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    coarse_nodes = sorted([n for n in graph.nodes if n.layer == "coarse"], key=lambda n: (n.start_ts, n.end_ts, n.id))
    next_edge_num = _next_edge_id(graph)
    next_node_num = max((int(n.id.split("_", 1)[1]) for n in graph.nodes if n.id.startswith("node_") and n.id.split("_", 1)[1].isdigit()), default=0) + 1

    for coarse in coarse_nodes:
        children = sorted(
            [n for n in graph.nodes if n.layer == "fine" and n.parent_coarse_id == coarse.id],
            key=lambda n: (n.start_ts, n.end_ts, n.id),
        )
        if not children:
            fallback = GraphNode(
                id=f"node_{next_node_num}",
                lecture_id=graph.lecture_id,
                title=f"{coarse.title} (single segment)",
                summary=coarse.summary,
                explanation=coarse.explanation,
                source_span_ids=coarse.source_span_ids,
                start_ts=coarse.start_ts,
                end_ts=coarse.end_ts,
                layer="fine",
                parent_coarse_id=coarse.id,
            )
            next_node_num += 1
            graph.nodes.append(fallback)
            graph.edges.append(
                GraphEdge(
                    id=f"edge_{next_edge_num}",
                    from_id=fallback.id,
                    to_id=coarse.id,
                    type="part_of",
                    reason="fallback fine node for empty coarse",
                    evidence_span_ids=fallback.source_span_ids[:3],
                    edge_confidence=0.95,
                    evidence_count=min(3, len(fallback.source_span_ids)),
                )
            )
            next_edge_num += 1
            actions.append({"action": "create_fallback_fine", "coarse_id": coarse.id, "fine_id": fallback.id})
            continue

        children.sort(key=lambda n: (n.start_ts, n.end_ts, n.id))
        first = children[0]
        if abs(first.start_ts - coarse.start_ts) > 1e-6:
            actions.append({"action": "snap_child_start", "node_id": first.id, "from": first.start_ts, "to": coarse.start_ts})
            first.start_ts = coarse.start_ts

        for i in range(1, len(children)):
            prev = children[i - 1]
            cur = children[i]
            if abs(prev.end_ts - cur.start_ts) > 1e-6:
                boundary = (prev.end_ts + cur.start_ts) / 2.0
                actions.append(
                    {
                        "action": "realign_child_boundary",
                        "left": prev.id,
                        "right": cur.id,
                        "new_boundary": boundary,
                    }
                )
                prev.end_ts = boundary
                cur.start_ts = boundary

        last = children[-1]
        if abs(last.end_ts - coarse.end_ts) > 1e-6:
            actions.append({"action": "snap_child_end", "node_id": last.id, "from": last.end_ts, "to": coarse.end_ts})
            last.end_ts = coarse.end_ts

        for child in children:
            if child.start_ts < coarse.start_ts:
                child.start_ts = coarse.start_ts
            if child.end_ts > coarse.end_ts:
                child.end_ts = coarse.end_ts
            if child.end_ts <= child.start_ts:
                child.end_ts = min(coarse.end_ts, child.start_ts + 1.0)

    return actions


def graph_to_dot(graph: Graph, path: Path) -> None:
    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    lines = [
        "digraph G {",
        "  rankdir=LR;",
        '  graph [bgcolor="white", splines=true, overlap=false];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9, color="#666666"];',
    ]
    for node in graph.nodes:
        fill = "#eaf4ff" if node.layer == "coarse" else "#eef8ea"
        border = "#5b8def" if node.layer == "coarse" else "#5a9b5a"
        label = f"{node.id}\\n{node.title}\\n[{int(node.start_ts)}-{int(node.end_ts)}s]\\n{node.layer}"
        lines.append(f'  "{esc(node.id)}" [label="{esc(label)}", fillcolor="{fill}", color="{border}"];')
    for edge in graph.edges:
        color = {
            "part_of": "#2d6cdf",
            "requires": "#d97706",
            "related_to": "#6b7280",
            "example_of": "#059669",
            "references": "#7c3aed",
        }.get(edge.type, "#666666")
        lines.append(f'  "{esc(edge.from_id)}" -> "{esc(edge.to_id)}" [label="{esc(edge.type)}", color="{color}"];')
    lines.append("}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_graph_v3(
    *,
    audio_path: Path,
    video_path: Path | None,
    output_path: Path,
    coarse_window_sec: int = 240,
    coarse_overlap_sec: int = 60,
    fine_window_sec: int = 90,
    fine_overlap_sec: int = 20,
    coarse_min_segment_sec: float = 180.0,
    coarse_max_segment_sec: float = 900.0,
    fine_min_segment_sec: float = 25.0,
    fine_max_segment_sec: float = 240.0,
    coarse_target_nodes: int = 12,
    fine_target_nodes: int = 5,
    signal_smooth_width: int = 3,
    candidate_peak_percentile: int = 80,
    llm_gate_threshold: float = 0.55,
    duration_penalty: float = 0.25,
    prefer_remote_llm: bool = True,
    verbose: bool = False,
) -> Tuple[Graph, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if "graph_v3" not in output_path.parts:
        raise ValueError(f"graph_v3 outputs must live under a graph_v3 path, got: {output_path}")

    lecture_id, spans = load_lecture_entries(audio_path, video_path)
    llm = GraphV3LLMClient(prefer_remote=prefer_remote_llm)
    graph = Graph(
        lecture_id=lecture_id,
        spans=spans,
        metadata={
            "mode": "graph_v3_hybrid_reliable",
            "source_audio": str(audio_path),
            "source_video": str(video_path) if video_path else None,
            "params": {
                "coarse_window_sec": coarse_window_sec,
                "coarse_overlap_sec": coarse_overlap_sec,
                "fine_window_sec": fine_window_sec,
                "fine_overlap_sec": fine_overlap_sec,
                "coarse_min_segment_sec": coarse_min_segment_sec,
                "coarse_max_segment_sec": coarse_max_segment_sec,
                "fine_min_segment_sec": fine_min_segment_sec,
                "fine_max_segment_sec": fine_max_segment_sec,
                "coarse_target_nodes": coarse_target_nodes,
                "fine_target_nodes": fine_target_nodes,
                "signal_smooth_width": signal_smooth_width,
                "candidate_peak_percentile": candidate_peak_percentile,
                "llm_gate_threshold": llm_gate_threshold,
                "duration_penalty": duration_penalty,
            },
            "env_resolution": llm.env_metadata(),
        },
    )

    if not spans:
        validation = validate_graph_v3(graph, fine_min_segment_sec=fine_min_segment_sec, coarse_min_segment_sec=coarse_min_segment_sec)
        return graph, {"coarse": {}, "fine": {}, "repair": []}, validation, {"coarse": {}, "fine": {}}

    edge_trace: List[Dict[str, Any]] = []

    coarse_nodes, coarse_trace, coarse_weak = _run_phase(
        graph=graph,
        llm=llm,
        spans=spans,
        phase="coarse",
        coarse_parent=None,
        window_sec=coarse_window_sec,
        overlap_sec=coarse_overlap_sec,
        min_segment_sec=coarse_min_segment_sec,
        max_segment_sec=coarse_max_segment_sec,
        target_nodes=coarse_target_nodes,
        smooth_width=signal_smooth_width,
        candidate_peak_percentile=candidate_peak_percentile,
        llm_gate_threshold=llm_gate_threshold,
        duration_penalty=duration_penalty,
        verbose=verbose,
        edge_trace=edge_trace,
    )

    fine_trace: Dict[str, Any] = {}
    fine_weak: Dict[str, Any] = {}
    coarse_nodes = sorted([n for n in graph.nodes if n.layer == "coarse"], key=lambda n: (n.start_ts, n.end_ts, n.id))
    for coarse in coarse_nodes:
        coarse_spans = spans_in_range(spans, coarse.start_ts, coarse.end_ts, include_end=True)
        _, trace, weak = _run_phase(
            graph=graph,
            llm=llm,
            spans=coarse_spans,
            phase="fine",
            coarse_parent=coarse,
            window_sec=fine_window_sec,
            overlap_sec=fine_overlap_sec,
            min_segment_sec=fine_min_segment_sec,
            max_segment_sec=fine_max_segment_sec,
            target_nodes=fine_target_nodes,
            smooth_width=signal_smooth_width,
            candidate_peak_percentile=candidate_peak_percentile,
            llm_gate_threshold=llm_gate_threshold,
            duration_penalty=duration_penalty,
            verbose=verbose,
            edge_trace=edge_trace,
        )
        fine_trace[coarse.id] = trace
        fine_weak[coarse.id] = weak

    repair_actions = _repair_fine_within_coarse(graph)
    _cleanup_edges_v3(graph)

    validation = validate_graph_v3(
        graph,
        fine_min_segment_sec=fine_min_segment_sec,
        coarse_min_segment_sec=coarse_min_segment_sec,
    )
    print_validation_summary_v3(validation)

    graph.metadata["validation"] = {
        "ok": validation["ok"],
        "error_count": validation["error_count"],
        "warning_count": validation["warning_count"],
    }
    graph.metadata["weak_label_counts"] = {
        "coarse": {
            "accepted": len(coarse_weak.get("accepted", [])),
            "rejected": len(coarse_weak.get("rejected", [])),
        },
        "fine": {
            coarse_id: {
                "accepted": len(data.get("accepted", [])),
                "rejected": len(data.get("rejected", [])),
            }
            for coarse_id, data in fine_weak.items()
        },
    }
    layers = SegmentationLayers(
        coarse_boundaries=sorted({n.end_ts for n in graph.nodes if n.layer == "coarse"})[:-1],
        fine_boundaries=sorted({n.end_ts for n in graph.nodes if n.layer == "fine"})[:-1],
        calibration_state={
            "llm_gate_threshold": llm_gate_threshold,
            "candidate_peak_percentile": candidate_peak_percentile,
            "duration_penalty": duration_penalty,
        },
    )
    graph.metadata["segmentation_layers"] = asdict(layers)

    trace_payload = {
        "coarse": coarse_trace,
        "fine": fine_trace,
        "repair": repair_actions,
        "edge_trace": edge_trace,
    }
    weak_payload = {
        "coarse": coarse_weak,
        "fine": fine_weak,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2)

    output_path.with_name("graph_v3_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    output_path.with_name("graph_v3_weak_labels.json").write_text(json.dumps(weak_payload, indent=2), encoding="utf-8")
    output_path.with_name("graph_v3_trace.json").write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")

    visuals_dir = output_path.parent / "visuals"
    graph_to_dot(graph, visuals_dir / "graph_v3.dot")
    if shutil.which("dot"):
        subprocess.run(
            [shutil.which("dot") or "dot", "-Tpng", str(visuals_dir / "graph_v3.dot"), "-o", str(visuals_dir / "graph_v3.png")],
            check=False,
        )

    if verbose:
        print(f"[done] nodes={len(graph.nodes)} edges={len(graph.edges)}")

    return graph, trace_payload, validation, weak_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Graph V3 reliability-first lecture graph")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--video")
    parser.add_argument("--output", required=True)
    parser.add_argument("--coarse-window-sec", type=int, default=240)
    parser.add_argument("--coarse-overlap-sec", type=int, default=60)
    parser.add_argument("--fine-window-sec", type=int, default=90)
    parser.add_argument("--fine-overlap-sec", type=int, default=20)
    parser.add_argument("--coarse-min-segment-sec", type=float, default=180.0)
    parser.add_argument("--coarse-max-segment-sec", type=float, default=900.0)
    parser.add_argument("--fine-min-segment-sec", type=float, default=25.0)
    parser.add_argument("--fine-max-segment-sec", type=float, default=240.0)
    parser.add_argument("--coarse-target-nodes", type=int, default=12)
    parser.add_argument("--fine-target-nodes", type=int, default=5)
    parser.add_argument("--signal-smooth-width", type=int, default=3)
    parser.add_argument("--candidate-peak-percentile", type=int, default=80)
    parser.add_argument("--llm-gate-threshold", type=float, default=0.55)
    parser.add_argument("--duration-penalty", type=float, default=0.25)
    parser.add_argument("--no-remote-llm", action="store_true", help="Disable remote Azure/OpenAI calls and use deterministic fallbacks.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_graph_v3(
        audio_path=Path(args.audio),
        video_path=Path(args.video) if args.video else None,
        output_path=Path(args.output),
        coarse_window_sec=args.coarse_window_sec,
        coarse_overlap_sec=args.coarse_overlap_sec,
        fine_window_sec=args.fine_window_sec,
        fine_overlap_sec=args.fine_overlap_sec,
        coarse_min_segment_sec=args.coarse_min_segment_sec,
        coarse_max_segment_sec=args.coarse_max_segment_sec,
        fine_min_segment_sec=args.fine_min_segment_sec,
        fine_max_segment_sec=args.fine_max_segment_sec,
        coarse_target_nodes=args.coarse_target_nodes,
        fine_target_nodes=args.fine_target_nodes,
        signal_smooth_width=args.signal_smooth_width,
        candidate_peak_percentile=args.candidate_peak_percentile,
        llm_gate_threshold=args.llm_gate_threshold,
        duration_penalty=args.duration_penalty,
        prefer_remote_llm=not args.no_remote_llm,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
