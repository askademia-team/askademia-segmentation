from __future__ import annotations

from typing import Any, Dict, List

from .models import Graph, GraphNode


def validate_graph(
    graph: Graph,
    *,
    min_segment_sec: float = 25.0,
    coverage_tolerance_sec: float = 1.0,
) -> Dict[str, Any]:
    node_by_id = {n.id: n for n in graph.nodes}
    fine_nodes = sorted([n for n in graph.nodes if n.layer == 0], key=lambda n: (n.start_ts, n.end_ts, n.id))
    hierarchy_nodes = sorted([n for n in graph.nodes if n.layer > 0], key=lambda n: (n.layer, n.start_ts, n.end_ts, n.id))
    part_of_edges = [e for e in graph.edges if e.type == "part_of"]

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    def err(code: str, **payload: Any) -> None:
        errors.append({"code": code, **payload})

    def warn(code: str, **payload: Any) -> None:
        warnings.append({"code": code, **payload})

    for node in graph.nodes:
        if not node.source_span_ids:
            err("empty_node", node_id=node.id)
        if node.end_ts <= node.start_ts:
            err("non_positive_duration", node_id=node.id, start_ts=node.start_ts, end_ts=node.end_ts)
        if node.layer == 0 and (node.end_ts - node.start_ts) < min_segment_sec - 1e-6:
            warn("short_fine_node", node_id=node.id, duration=node.end_ts - node.start_ts)

    part_of_by_child: Dict[str, List[str]] = {}
    for edge in part_of_edges:
        from_node = node_by_id.get(edge.from_id)
        to_node = node_by_id.get(edge.to_id)
        if not from_node or not to_node:
            err("dangling_part_of", edge_id=edge.id, from_id=edge.from_id, to_id=edge.to_id)
            continue
        if to_node.layer != from_node.layer + 1:
            err("invalid_part_of_layers", edge_id=edge.id, from_layer=from_node.layer, to_layer=to_node.layer)
        if from_node.parent_node_id and from_node.parent_node_id != to_node.id:
            err(
                "parent_node_mismatch",
                edge_id=edge.id,
                child_node_id=from_node.id,
                parent_node_id=from_node.parent_node_id,
                edge_to_id=to_node.id,
            )
        if not _contains(to_node, from_node):
            err("parent_does_not_contain_child", edge_id=edge.id, parent_id=to_node.id, child_id=from_node.id)
        part_of_by_child.setdefault(from_node.id, []).append(to_node.id)

    for fine_node in fine_nodes:
        if fine_node.parent_node_id and fine_node.parent_node_id not in node_by_id:
            err("fine_parent_missing", node_id=fine_node.id, parent_node_id=fine_node.parent_node_id)
        parents = part_of_by_child.get(fine_node.id, [])
        if len(set(parents)) > 1:
            err("fine_part_of_count_invalid", node_id=fine_node.id, parent_ids=parents)

    for hierarchy_node in hierarchy_nodes:
        contained = [child.id for child in graph.nodes if child.layer < hierarchy_node.layer and _contains(hierarchy_node, child)]
        if not contained:
            warn("empty_hierarchy_container", node_id=hierarchy_node.id)

    for edge in graph.edges:
        from_node = node_by_id.get(edge.from_id)
        to_node = node_by_id.get(edge.to_id)
        if not from_node or not to_node:
            err("dangling_edge", edge_id=edge.id, from_id=edge.from_id, to_id=edge.to_id)
            continue
        if edge.from_id == edge.to_id:
            err("self_edge", edge_id=edge.id, node_id=edge.from_id)
        if edge.type != "part_of" and from_node.layer != to_node.layer:
            err(
                "cross_layer_semantic_edge",
                edge_id=edge.id,
                edge_type=edge.type,
                from_layer=from_node.layer,
                to_layer=to_node.layer,
            )
        if edge.type == "requires" and from_node.start_ts + coverage_tolerance_sec < to_node.start_ts:
            warn(
                "requires_direction_suspicious",
                edge_id=edge.id,
                from_id=edge.from_id,
                to_id=edge.to_id,
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
            "coarse_count": len(hierarchy_nodes),
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

    lecture_start = min(span.timestamp for span in graph.spans)
    lecture_end = max(span.timestamp for span in graph.spans)

    if fine_nodes[0].start_ts > lecture_start + coverage_tolerance_sec:
        errors.append({"code": "fine_coverage_gap_at_start", "gap_start": lecture_start, "gap_end": fine_nodes[0].start_ts})
    if fine_nodes[-1].end_ts < lecture_end - coverage_tolerance_sec:
        errors.append({"code": "fine_coverage_gap_at_end", "gap_start": fine_nodes[-1].end_ts, "gap_end": lecture_end})

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
        print(f"  - error: {item['code']} { {k: v for k, v in item.items() if k != 'code'} }")
    for item in report["warnings"][:10]:
        print(f"  - warn: {item['code']} { {k: v for k, v in item.items() if k != 'code'} }")
