from __future__ import annotations

import json

from graph_v1.models import Graph, GraphEdge, GraphNode, Span, TutorSessionState

from .builder import cleanup_graph_v2_edges
from .tutor import TutorRuntimeV2
from .validate import validate_graph_v2


def test_v2_cleanup_blocks_cross_coarse_fine_edges() -> None:
    spans = [Span(span_id="a_0", lecture_id="lec", timestamp=0, text="x", modality="audio")]
    nodes = [
        GraphNode(id="coarse_1", lecture_id="lec", title="A", summary="", explanation="", source_span_ids=["a_0"], start_ts=0, end_ts=100, layer="coarse"),
        GraphNode(id="coarse_2", lecture_id="lec", title="B", summary="", explanation="", source_span_ids=["a_0"], start_ts=100, end_ts=200, layer="coarse"),
        GraphNode(id="fine_1", lecture_id="lec", title="A1", summary="", explanation="", source_span_ids=["a_0"], start_ts=0, end_ts=40, layer="fine", parent_coarse_id="coarse_1"),
        GraphNode(id="fine_2", lecture_id="lec", title="B1", summary="", explanation="", source_span_ids=["a_0"], start_ts=120, end_ts=160, layer="fine", parent_coarse_id="coarse_2"),
    ]
    edges = [
        GraphEdge(id="edge_1", from_id="fine_1", to_id="fine_2", type="related_to", reason="bad", evidence_span_ids=["a_0"]),
        GraphEdge(id="edge_2", from_id="fine_1", to_id="coarse_1", type="part_of", reason="ok", evidence_span_ids=["a_0"]),
        GraphEdge(id="edge_3", from_id="fine_2", to_id="coarse_2", type="part_of", reason="ok", evidence_span_ids=["a_0"]),
    ]
    graph = Graph(lecture_id="lec", nodes=nodes, edges=edges, spans=spans, metadata={})
    cleanup_graph_v2_edges(graph)
    kept = {(e.from_id, e.to_id, e.type) for e in graph.edges}
    assert ("fine_1", "fine_2", "related_to") not in kept
    assert ("fine_1", "coarse_1", "part_of") in kept


def test_v2_validation_detects_cross_coarse_fine_edge() -> None:
    spans = [Span(span_id="a_0", lecture_id="lec", timestamp=0, text="x", modality="audio")]
    nodes = [
        GraphNode(id="coarse_1", lecture_id="lec", title="A", summary="", explanation="", source_span_ids=["a_0"], start_ts=0, end_ts=100, layer="coarse"),
        GraphNode(id="coarse_2", lecture_id="lec", title="B", summary="", explanation="", source_span_ids=["a_0"], start_ts=100, end_ts=220, layer="coarse"),
        GraphNode(id="fine_1", lecture_id="lec", title="A1", summary="", explanation="", source_span_ids=["a_0"], start_ts=0, end_ts=40, layer="fine", parent_coarse_id="coarse_1"),
        GraphNode(id="fine_2", lecture_id="lec", title="B1", summary="", explanation="", source_span_ids=["a_0"], start_ts=120, end_ts=170, layer="fine", parent_coarse_id="coarse_2"),
    ]
    edges = [
        GraphEdge(id="edge_1", from_id="fine_1", to_id="coarse_1", type="part_of", reason="ok", evidence_span_ids=["a_0"]),
        GraphEdge(id="edge_2", from_id="fine_2", to_id="coarse_2", type="part_of", reason="ok", evidence_span_ids=["a_0"]),
        GraphEdge(id="edge_3", from_id="fine_1", to_id="fine_2", type="related_to", reason="bad", evidence_span_ids=["a_0"]),
    ]
    report = validate_graph_v2(Graph(lecture_id="lec", nodes=nodes, edges=edges, spans=spans, metadata={}), fine_min_segment_sec=20, coarse_min_segment_sec=50)
    assert not report["ok"]
    assert any(err["code"] == "cross_coarse_fine_edge" for err in report["errors"])


def test_tutor_v2_prefers_selected_coarse_region() -> None:
    spans = [
        Span(span_id="a_0", lecture_id="lec", timestamp=0, text="missing data handling and null values", modality="audio"),
        Span(span_id="a_1", lecture_id="lec", timestamp=60, text="canonicalization before merging datasets", modality="audio"),
    ]
    nodes = [
        GraphNode(id="coarse_1", lecture_id="lec", title="Missing Data", summary="Handling missing values.", explanation="", source_span_ids=["a_0"], start_ts=0, end_ts=50, layer="coarse"),
        GraphNode(id="coarse_2", lecture_id="lec", title="Canonicalization", summary="Standardizing text before merges.", explanation="", source_span_ids=["a_1"], start_ts=50, end_ts=120, layer="coarse"),
        GraphNode(id="fine_1", lecture_id="lec", title="Missing Values", summary="Nulls and defaults.", explanation="", source_span_ids=["a_0"], start_ts=0, end_ts=50, layer="fine", parent_coarse_id="coarse_1"),
        GraphNode(id="fine_2", lecture_id="lec", title="Canonicalizing Names", summary="Normalize keys before joins.", explanation="Canonicalization improves merge key consistency.", source_span_ids=["a_1"], start_ts=50, end_ts=120, layer="fine", parent_coarse_id="coarse_2"),
    ]
    edges = [
        GraphEdge(id="edge_1", from_id="fine_1", to_id="coarse_1", type="part_of", reason="", evidence_span_ids=["a_0"]),
        GraphEdge(id="edge_2", from_id="fine_2", to_id="coarse_2", type="part_of", reason="", evidence_span_ids=["a_1"]),
    ]
    graph = Graph(lecture_id="lec", nodes=nodes, edges=edges, spans=spans, metadata={})
    runtime = TutorRuntimeV2(graph)
    out = runtime.run_guided_answer("How does canonicalization help before merging datasets?", TutorSessionState(session_id="s1"), top_k=5, max_hops=3)
    assert out["citations"]["nodes"], out
    assert any(node_id in {"coarse_2", "fine_2"} for node_id in out["citations"]["nodes"]), out


if __name__ == "__main__":
    test_v2_cleanup_blocks_cross_coarse_fine_edges()
    test_v2_validation_detects_cross_coarse_fine_edge()
    test_tutor_v2_prefers_selected_coarse_region()
    print(json.dumps({"ok": True}))
