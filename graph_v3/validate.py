from __future__ import annotations

from typing import Any, Dict, List

from graph_v1.models import Graph
from graph_v1.validate import validate_graph


def validate_graph_v3(
    graph: Graph,
    *,
    fine_min_segment_sec: float = 25.0,
    coarse_min_segment_sec: float = 180.0,
    tolerance_sec: float = 1e-3,
) -> Dict[str, Any]:
    base = validate_graph(graph, min_segment_sec=fine_min_segment_sec, coverage_tolerance_sec=tolerance_sec)
    node_by_id = {n.id: n for n in graph.nodes}
    fine_nodes = sorted([n for n in graph.nodes if n.layer == "fine"], key=lambda n: (n.start_ts, n.end_ts, n.id))
    coarse_nodes = sorted([n for n in graph.nodes if n.layer == "coarse"], key=lambda n: (n.start_ts, n.end_ts, n.id))

    errors: List[Dict[str, Any]] = list(base["errors"])
    warnings: List[Dict[str, Any]] = list(base["warnings"])

    def err(code: str, **payload: Any) -> None:
        errors.append({"code": code, **payload})

    def warn(code: str, **payload: Any) -> None:
        warnings.append({"code": code, **payload})

    children_by_coarse: Dict[str, List[Any]] = {cn.id: [] for cn in coarse_nodes}
    for fn in fine_nodes:
        if not fn.parent_coarse_id:
            err("fine_missing_parent_coarse", node_id=fn.id)
            continue
        parent = node_by_id.get(fn.parent_coarse_id)
        if not parent or parent.layer != "coarse":
            err("fine_parent_not_coarse", node_id=fn.id, parent_coarse_id=fn.parent_coarse_id)
            continue
        if fn.start_ts < parent.start_ts - tolerance_sec or fn.end_ts > parent.end_ts + tolerance_sec:
            err(
                "fine_outside_parent",
                node_id=fn.id,
                parent_coarse_id=parent.id,
                fine_start_ts=fn.start_ts,
                fine_end_ts=fn.end_ts,
                coarse_start_ts=parent.start_ts,
                coarse_end_ts=parent.end_ts,
            )
        children_by_coarse[parent.id].append(fn)

    for coarse in coarse_nodes:
        dur = coarse.end_ts - coarse.start_ts
        if dur < coarse_min_segment_sec - tolerance_sec:
            warn("short_coarse_node", node_id=coarse.id, duration=dur)

        children = sorted(children_by_coarse.get(coarse.id, []), key=lambda n: (n.start_ts, n.end_ts, n.id))
        if not children:
            err("empty_coarse_node", node_id=coarse.id)
            continue

        first = children[0]
        if first.start_ts > coarse.start_ts + tolerance_sec:
            err(
                "fine_gap_at_coarse_start",
                coarse_id=coarse.id,
                gap_start=coarse.start_ts,
                gap_end=first.start_ts,
            )
        if first.start_ts < coarse.start_ts - tolerance_sec:
            err(
                "fine_overlap_before_coarse_start",
                coarse_id=coarse.id,
                fine_id=first.id,
                fine_start=first.start_ts,
                coarse_start=coarse.start_ts,
            )

        last = children[-1]
        if last.end_ts < coarse.end_ts - tolerance_sec:
            err(
                "fine_gap_at_coarse_end",
                coarse_id=coarse.id,
                gap_start=last.end_ts,
                gap_end=coarse.end_ts,
            )
        if last.end_ts > coarse.end_ts + tolerance_sec:
            err(
                "fine_overlap_past_coarse_end",
                coarse_id=coarse.id,
                fine_id=last.id,
                fine_end=last.end_ts,
                coarse_end=coarse.end_ts,
            )

        for left, right in zip(children, children[1:]):
            if right.start_ts > left.end_ts + tolerance_sec:
                err(
                    "fine_gap_within_coarse",
                    coarse_id=coarse.id,
                    left_node_id=left.id,
                    right_node_id=right.id,
                    gap_start=left.end_ts,
                    gap_end=right.start_ts,
                )
            elif right.start_ts < left.end_ts - tolerance_sec:
                err(
                    "fine_overlap_within_coarse",
                    coarse_id=coarse.id,
                    left_node_id=left.id,
                    right_node_id=right.id,
                    overlap_start=right.start_ts,
                    overlap_end=left.end_ts,
                )

    for edge in graph.edges:
        src = node_by_id.get(edge.from_id)
        dst = node_by_id.get(edge.to_id)
        if not src or not dst:
            continue
        if edge.type == "part_of":
            if src.layer != "fine" or dst.layer != "coarse":
                err("invalid_part_of_layers_v3", edge_id=edge.id, from_layer=src.layer, to_layer=dst.layer)
            if src.parent_coarse_id != dst.id:
                err(
                    "part_of_parent_mismatch_v3",
                    edge_id=edge.id,
                    fine_id=src.id,
                    parent_coarse_id=src.parent_coarse_id,
                    edge_to_id=dst.id,
                )
            continue

        if src.layer != dst.layer:
            err(
                "cross_layer_semantic_edge_v3",
                edge_id=edge.id,
                from_layer=src.layer,
                to_layer=dst.layer,
                edge_type=edge.type,
            )
        if src.layer == "fine" and src.parent_coarse_id != dst.parent_coarse_id:
            err(
                "cross_coarse_fine_semantic_edge_v3",
                edge_id=edge.id,
                from_id=src.id,
                to_id=dst.id,
                from_parent=src.parent_coarse_id,
                to_parent=dst.parent_coarse_id,
            )
        if edge.edge_confidence < 0.65:
            warn("low_confidence_semantic_edge_v3", edge_id=edge.id, confidence=edge.edge_confidence, edge_type=edge.type)
        if not edge.evidence_span_ids:
            warn("missing_evidence_semantic_edge_v3", edge_id=edge.id, edge_type=edge.type)

    # Deduplicate exact duplicate errors to keep reports compact.
    dedup = {}
    for item in errors:
        key = tuple(sorted(item.items()))
        dedup[key] = item
    errors = list(dedup.values())

    return {
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "fine_count": len(fine_nodes),
            "coarse_count": len(coarse_nodes),
            "part_of_count": len([e for e in graph.edges if e.type == "part_of"]),
        },
    }


def print_validation_summary_v3(report: Dict[str, Any]) -> None:
    print(
        f"[validate_v3] ok={report['ok']} errors={report['error_count']} warnings={report['warning_count']} "
        f"nodes={report['stats']['node_count']} edges={report['stats']['edge_count']}"
    )
    for item in report["errors"][:10]:
        payload = {k: v for k, v in item.items() if k != "code"}
        print(f"  - error: {item['code']} {payload}")
    for item in report["warnings"][:10]:
        payload = {k: v for k, v in item.items() if k != "code"}
        print(f"  - warn: {item['code']} {payload}")

