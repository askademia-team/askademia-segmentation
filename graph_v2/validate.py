from __future__ import annotations

from typing import Any, Dict, List

from graph_v1.models import Graph
from graph_v1.validate import validate_graph


def validate_graph_v2(
    graph: Graph,
    *,
    fine_min_segment_sec: float = 25.0,
    coarse_min_segment_sec: float = 180.0,
) -> Dict[str, Any]:
    base = validate_graph(graph, min_segment_sec=fine_min_segment_sec)
    node_by_id = {n.id: n for n in graph.nodes}
    fine_nodes = [n for n in graph.nodes if n.layer == "fine"]
    coarse_nodes = [n for n in graph.nodes if n.layer == "coarse"]

    errors: List[Dict[str, Any]] = list(base["errors"])
    warnings: List[Dict[str, Any]] = list(base["warnings"])

    def err(code: str, **payload: Any) -> None:
        errors.append({"code": code, **payload})

    def warn(code: str, **payload: Any) -> None:
        warnings.append({"code": code, **payload})

    coarse_children: Dict[str, List[str]] = {n.id: [] for n in coarse_nodes}
    for fine in fine_nodes:
        if not fine.parent_coarse_id:
            continue
        parent = node_by_id.get(fine.parent_coarse_id)
        if not parent or parent.layer != "coarse":
            err("fine_parent_not_coarse", node_id=fine.id, parent_coarse_id=fine.parent_coarse_id)
            continue
        if fine.start_ts < parent.start_ts or fine.end_ts > parent.end_ts:
            err(
                "fine_outside_coarse",
                node_id=fine.id,
                parent_coarse_id=parent.id,
                fine_start_ts=fine.start_ts,
                fine_end_ts=fine.end_ts,
                coarse_start_ts=parent.start_ts,
                coarse_end_ts=parent.end_ts,
            )
        coarse_children[parent.id].append(fine.id)

    for coarse in coarse_nodes:
        if (coarse.end_ts - coarse.start_ts) < coarse_min_segment_sec - 1e-6:
            warn("short_coarse_node", node_id=coarse.id, duration=coarse.end_ts - coarse.start_ts)
        if not coarse_children.get(coarse.id):
            err("empty_coarse_node", node_id=coarse.id)

    for edge in graph.edges:
        from_node = node_by_id.get(edge.from_id)
        to_node = node_by_id.get(edge.to_id)
        if not from_node or not to_node or edge.type == "part_of":
            continue
        if from_node.layer == "fine" and to_node.layer == "fine" and from_node.parent_coarse_id != to_node.parent_coarse_id:
            err(
                "cross_coarse_fine_edge",
                edge_id=edge.id,
                from_id=edge.from_id,
                to_id=edge.to_id,
                from_parent=from_node.parent_coarse_id,
                to_parent=to_node.parent_coarse_id,
            )

    seen = set()
    dedup_errors = []
    for item in errors:
        key = tuple(sorted(item.items()))
        if key in seen:
            continue
        seen.add(key)
        dedup_errors.append(item)
    errors = dedup_errors

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
