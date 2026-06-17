from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .models import Graph, GraphNode


def validate_graph(
    graph: Graph,
    *,
    min_segment_sec: float = 25.0,
    coverage_tolerance_sec: float = 1.0,
) -> Dict[str, Any]:
    node_by_id = {n.id: n for n in graph.nodes}
    fine_nodes = sorted([n for n in graph.nodes if n.layer == "fine"], key=lambda n: (n.start_ts, n.end_ts, n.id))
    coarse_nodes = sorted([n for n in graph.nodes if n.layer == "coarse"], key=lambda n: (n.start_ts, n.end_ts, n.id))
    part_of_edges = [e for e in graph.edges if e.type == "part_of"]

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    def err(code: str, **payload: Any) -> None:
        errors.append({"code": code, **payload})

    def warn(code: str, **payload: Any) -> None:
        warnings.append({"code": code, **payload})

    for n in graph.nodes:
        if not n.source_span_ids:
            err("empty_node", node_id=n.id)
        if n.end_ts <= n.start_ts:
            err("non_positive_duration", node_id=n.id, start_ts=n.start_ts, end_ts=n.end_ts)
        if n.layer == "fine" and (n.end_ts - n.start_ts) < min_segment_sec - 1e-6:
            warn("short_fine_node", node_id=n.id, duration=n.end_ts - n.start_ts)

    part_of_by_fine: Dict[str, List[str]] = {}
    for e in part_of_edges:
        from_node = node_by_id.get(e.from_id)
        to_node = node_by_id.get(e.to_id)
        if not from_node or not to_node:
            err("dangling_part_of", edge_id=e.id, from_id=e.from_id, to_id=e.to_id)
            continue
        if from_node.layer != "fine" or to_node.layer != "coarse":
            err("invalid_part_of_layers", edge_id=e.id, from_layer=from_node.layer, to_layer=to_node.layer)
            continue
        if from_node.parent_coarse_id and from_node.parent_coarse_id != to_node.id:
            err(
                "parent_coarse_mismatch",
                edge_id=e.id,
                fine_node_id=from_node.id,
                parent_coarse_id=from_node.parent_coarse_id,
                edge_to_id=to_node.id,
            )
        if not _contains(to_node, from_node):
            err("coarse_does_not_contain_fine", edge_id=e.id, coarse_id=to_node.id, fine_id=from_node.id)
        part_of_by_fine.setdefault(from_node.id, []).append(to_node.id)

    for fn in fine_nodes:
        if not coarse_nodes:
            break
        if not fn.parent_coarse_id:
            err("fine_missing_parent_coarse", node_id=fn.id)
        parents = part_of_by_fine.get(fn.id, [])
        if len(set(parents)) != 1:
            err("fine_part_of_count_invalid", node_id=fn.id, parent_ids=parents)
        if fn.parent_coarse_id and fn.parent_coarse_id not in node_by_id:
            err("fine_parent_missing", node_id=fn.id, parent_coarse_id=fn.parent_coarse_id)

    for cn in coarse_nodes:
        contained = [fn.id for fn in fine_nodes if _contains(cn, fn)]
        if not contained:
            warn("empty_coarse_container", node_id=cn.id)

    for e in graph.edges:
        from_node = node_by_id.get(e.from_id)
        to_node = node_by_id.get(e.to_id)
        if not from_node or not to_node:
            err("dangling_edge", edge_id=e.id, from_id=e.from_id, to_id=e.to_id)
            continue
        if e.from_id == e.to_id:
            err("self_edge", edge_id=e.id, node_id=e.from_id)
        if e.type != "part_of" and from_node.layer != to_node.layer:
            err(
                "cross_layer_semantic_edge",
                edge_id=e.id,
                edge_type=e.type,
                from_layer=from_node.layer,
                to_layer=to_node.layer,
            )
        if e.type == "requires" and from_node.start_ts + coverage_tolerance_sec < to_node.start_ts:
            warn(
                "requires_direction_suspicious",
                edge_id=e.id,
                from_id=e.from_id,
                to_id=e.to_id,
                from_start_ts=from_node.start_ts,
                to_start_ts=to_node.start_ts,
            )

    coverage = validate_fine_coverage(fine_nodes, graph, coverage_tolerance_sec=coverage_tolerance_sec)
    errors.extend(coverage["errors"])
    warnings.extend(coverage["warnings"])

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
            "part_of_count": len(part_of_edges),
        },
    }


def validate_fine_coverage(
    fine_nodes: List[GraphNode],
    graph: Graph,
    *,
    coverage_tolerance_sec: float = 1.0,
) -> Dict[str, List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    if not fine_nodes or not graph.spans:
        return {"errors": errors, "warnings": warnings}

    lecture_start = min(s.timestamp for s in graph.spans)
    lecture_end = max(s.timestamp for s in graph.spans)

    if fine_nodes[0].start_ts > lecture_start + coverage_tolerance_sec:
        errors.append(
            {
                "code": "fine_coverage_gap_at_start",
                "gap_start": lecture_start,
                "gap_end": fine_nodes[0].start_ts,
            }
        )
    if fine_nodes[-1].end_ts < lecture_end - coverage_tolerance_sec:
        errors.append(
            {
                "code": "fine_coverage_gap_at_end",
                "gap_start": fine_nodes[-1].end_ts,
                "gap_end": lecture_end,
            }
        )

    for prev, cur in zip(fine_nodes, fine_nodes[1:]):
        if cur.start_ts > prev.end_ts + coverage_tolerance_sec:
            errors.append(
                {
                    "code": "fine_coverage_gap",
                    "left_node_id": prev.id,
                    "right_node_id": cur.id,
                    "gap_start": prev.end_ts,
                    "gap_end": cur.start_ts,
                }
            )
        elif cur.start_ts < prev.end_ts - coverage_tolerance_sec:
            warnings.append(
                {
                    "code": "fine_overlap",
                    "left_node_id": prev.id,
                    "right_node_id": cur.id,
                    "overlap_start": cur.start_ts,
                    "overlap_end": prev.end_ts,
                }
            )
    return {"errors": errors, "warnings": warnings}


def _contains(container: GraphNode, child: GraphNode) -> bool:
    return container.start_ts <= child.start_ts and child.end_ts <= container.end_ts


def print_validation_summary(report: Dict[str, Any]) -> None:
    print(
        f"[validate] ok={report['ok']} errors={report['error_count']} warnings={report['warning_count']} "
        f"nodes={report['stats']['node_count']} edges={report['stats']['edge_count']}"
    )
    for item in report["errors"][:10]:
        print(f"  - error: {item['code']} {asdict_payload(item)}")
    for item in report["warnings"][:10]:
        print(f"  - warn: {item['code']} {asdict_payload(item)}")


def asdict_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in item.items() if k != "code"}
