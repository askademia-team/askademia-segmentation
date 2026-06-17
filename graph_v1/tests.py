from __future__ import annotations

import json
from pathlib import Path

from graph_v1.builder import _cleanup_graph_edges, build_windows
from graph_v1.llm import LLMClient
from graph_v1.models import Graph, GraphEdge, GraphNode, Span, TutorSessionState, validate_edge_type
from graph_v1.render_temporal import build_stacked_layout_dot, order_hierarchical_nodes
from graph_v1.retrieval import GraphRetriever
from graph_v1.tutor import TutorRuntime
from graph_v1.transition_builder import _rebuild_hierarchy
from graph_v1.validate import validate_graph


def test_windowing() -> None:
    spans = [Span(span_id=f"a_{i}", lecture_id="lec", timestamp=float(i * 10), text=f"t{i}", modality="audio") for i in range(30)]
    windows = build_windows(spans, "lec", window_sec=75, overlap_sec=15)
    assert windows, "windows should not be empty"
    assert windows[0].start_ts == 0.0
    if len(windows) > 1:
        assert abs(windows[1].start_ts - 60.0) < 1e-6


def test_edge_types() -> None:
    assert validate_edge_type("requires")
    assert validate_edge_type("part_of")
    assert not validate_edge_type("teaches")


def test_tutor_hops_and_citations() -> None:
    spans = [
        Span(span_id="a_0", lecture_id="lec", timestamp=0, text="Intro to data science", modality="audio"),
        Span(span_id="a_1", lecture_id="lec", timestamp=60, text="Dataframe basics", modality="audio"),
    ]
    nodes = [
        GraphNode(
            id="node_1",
            lecture_id="lec",
            title="Course intro",
            summary="Overview",
            explanation="Overview of lecture",
            source_span_ids=["a_0"],
            start_ts=0,
            end_ts=60,
        ),
        GraphNode(
            id="node_2",
            lecture_id="lec",
            title="Dataframe basics",
            summary="Columns rows",
            explanation="Basic dataframe operations",
            source_span_ids=["a_1"],
            start_ts=60,
            end_ts=120,
        ),
    ]
    edges = [
        GraphEdge(
            id="edge_1",
            from_id="node_2",
            to_id="node_1",
            type="requires",
            reason="Basics before advanced",
            evidence_span_ids=["a_1"],
        )
    ]
    graph = Graph(lecture_id="lec", nodes=nodes, edges=edges, spans=spans, metadata={})
    rt = TutorRuntime(graph)
    out = rt.run_guided_answer("What are the first concepts?", TutorSessionState(session_id="s1"), top_k=5, max_hops=4)
    assert out["citations"]["nodes"], "must cite at least one node"
    assert out["citations"]["spans"], "must cite at least one span"


def test_edge_cleanup_enforces_layer_rules() -> None:
    spans = [Span(span_id="a_0", lecture_id="lec", timestamp=0, text="x", modality="audio")]
    nodes = [
        GraphNode(
            id="fine_1",
            lecture_id="lec",
            title="Fine A",
            summary="",
            explanation="",
            source_span_ids=["a_0"],
            start_ts=10,
            end_ts=20,
            layer="fine",
            parent_coarse_id="coarse_1",
        ),
        GraphNode(
            id="fine_2",
            lecture_id="lec",
            title="Fine B",
            summary="",
            explanation="",
            source_span_ids=["a_0"],
            start_ts=20,
            end_ts=30,
            layer="fine",
            parent_coarse_id="coarse_1",
        ),
        GraphNode(
            id="coarse_1",
            lecture_id="lec",
            title="Coarse A",
            summary="",
            explanation="",
            source_span_ids=["a_0"],
            start_ts=0,
            end_ts=40,
            layer="coarse",
        ),
    ]
    edges = [
        GraphEdge(
            id="edge_1",
            from_id="fine_1",
            to_id="coarse_1",
            type="part_of",
            reason="valid structural edge",
            evidence_span_ids=["a_0"],
        ),
        GraphEdge(
            id="edge_2",
            from_id="fine_1",
            to_id="fine_2",
            type="part_of",
            reason="invalid fine-fine part_of",
            evidence_span_ids=["a_0"],
        ),
        GraphEdge(
            id="edge_3",
            from_id="fine_1",
            to_id="coarse_1",
            type="related_to",
            reason="invalid cross-layer semantic edge",
            evidence_span_ids=["a_0"],
        ),
        GraphEdge(
            id="edge_4",
            from_id="fine_1",
            to_id="fine_2",
            type="related_to",
            reason="valid same-layer semantic edge",
            evidence_span_ids=["a_0"],
        ),
    ]
    graph = Graph(lecture_id="lec", nodes=nodes, edges=edges, spans=spans, metadata={})
    _cleanup_graph_edges(graph)
    kept = {(e.from_id, e.to_id, e.type) for e in graph.edges}
    assert ("fine_1", "coarse_1", "part_of") in kept
    assert ("fine_1", "fine_2", "related_to") in kept
    assert ("fine_1", "fine_2", "part_of") not in kept
    assert ("fine_1", "coarse_1", "related_to") not in kept


def test_validator_passes_on_simple_hierarchy() -> None:
    spans = [
        Span(span_id="a_0", lecture_id="lec", timestamp=0, text="intro", modality="audio"),
        Span(span_id="a_1", lecture_id="lec", timestamp=30, text="dataframe", modality="audio"),
        Span(span_id="a_2", lecture_id="lec", timestamp=60, text="groupby", modality="audio"),
    ]
    nodes = [
        GraphNode(
            id="fine_1",
            lecture_id="lec",
            title="Dataframe basics",
            summary="",
            explanation="",
            source_span_ids=["a_0", "a_1"],
            start_ts=0,
            end_ts=45,
            layer="fine",
            parent_coarse_id="coarse_1",
        ),
        GraphNode(
            id="fine_2",
            lecture_id="lec",
            title="Groupby basics",
            summary="",
            explanation="",
            source_span_ids=["a_2"],
            start_ts=45,
            end_ts=70,
            layer="fine",
            parent_coarse_id="coarse_1",
        ),
        GraphNode(
            id="coarse_1",
            lecture_id="lec",
            title="Tabular operations",
            summary="",
            explanation="",
            source_span_ids=["a_0", "a_1", "a_2"],
            start_ts=0,
            end_ts=70,
            layer="coarse",
        ),
    ]
    edges = [
        GraphEdge(
            id="edge_1",
            from_id="fine_1",
            to_id="coarse_1",
            type="part_of",
            reason="",
            evidence_span_ids=["a_0"],
        ),
        GraphEdge(
            id="edge_2",
            from_id="fine_2",
            to_id="coarse_1",
            type="part_of",
            reason="",
            evidence_span_ids=["a_2"],
        ),
        GraphEdge(
            id="edge_3",
            from_id="fine_2",
            to_id="fine_1",
            type="requires",
            reason="groupby depends on dataframe basics",
            evidence_span_ids=["a_2"],
        ),
    ]
    report = validate_graph(Graph(lecture_id="lec", nodes=nodes, edges=edges, spans=spans, metadata={}), min_segment_sec=20)
    assert report["ok"], report


def test_retriever_prefers_semantic_match() -> None:
    spans = [
        Span(span_id="a_0", lecture_id="lec", timestamp=0, text="canonicalization standardizes names before merges", modality="audio"),
        Span(span_id="a_1", lecture_id="lec", timestamp=60, text="regular expressions match patterns in text", modality="audio"),
    ]
    nodes = [
        GraphNode(
            id="node_1",
            lecture_id="lec",
            title="Canonicalizing county names",
            summary="Standardize text values before joining data.",
            explanation="Canonicalization helps ensure matching keys align during merge operations.",
            source_span_ids=["a_0"],
            start_ts=0,
            end_ts=60,
        ),
        GraphNode(
            id="node_2",
            lecture_id="lec",
            title="Regular expressions",
            summary="Pattern matching tools for text.",
            explanation="Regex supports extraction and matching.",
            source_span_ids=["a_1"],
            start_ts=60,
            end_ts=120,
        ),
    ]
    graph = Graph(lecture_id="lec", nodes=nodes, edges=[], spans=spans, metadata={})
    retriever = GraphRetriever(graph, LLMClient())
    ranked = retriever.rank_nodes("How does canonicalization help before merging datasets?", k=2)
    assert ranked[0]["id"] == "node_1", ranked


def test_temporal_renderer_orders_children_under_parents() -> None:
    graph = Graph(
        lecture_id="lec",
        nodes=[
            GraphNode(
                id="coarse_1",
                lecture_id="lec",
                title="Coarse 1",
                summary="",
                explanation="",
                source_span_ids=[],
                start_ts=0,
                end_ts=100,
                layer="coarse",
            ),
            GraphNode(
                id="coarse_2",
                lecture_id="lec",
                title="Coarse 2",
                summary="",
                explanation="",
                source_span_ids=[],
                start_ts=100,
                end_ts=200,
                layer="coarse",
            ),
            GraphNode(
                id="fine_2",
                lecture_id="lec",
                title="Fine 2",
                summary="",
                explanation="",
                source_span_ids=[],
                start_ts=40,
                end_ts=70,
                layer="fine",
                parent_coarse_id="coarse_1",
            ),
            GraphNode(
                id="fine_1",
                lecture_id="lec",
                title="Fine 1",
                summary="",
                explanation="",
                source_span_ids=[],
                start_ts=10,
                end_ts=40,
                layer="fine",
                parent_coarse_id="coarse_1",
            ),
            GraphNode(
                id="fine_3",
                lecture_id="lec",
                title="Fine 3",
                summary="",
                explanation="",
                source_span_ids=[],
                start_ts=120,
                end_ts=160,
                layer="fine",
                parent_coarse_id="coarse_2",
            ),
        ],
        edges=[],
        spans=[],
        metadata={},
    )
    coarse_nodes, children = order_hierarchical_nodes(graph)
    assert [node.id for node in coarse_nodes] == ["coarse_1", "coarse_2"]
    assert [node.id for node in children["coarse_1"]] == ["fine_1", "fine_2"]
    assert [node.id for node in children["coarse_2"]] == ["fine_3"]


def test_temporal_renderer_stacked_layout_omits_semantic_edges() -> None:
    graph = Graph(
        lecture_id="lec",
        nodes=[
            GraphNode(
                id="coarse_1",
                lecture_id="lec",
                title="Coarse 1",
                summary="",
                explanation="",
                source_span_ids=[],
                start_ts=0,
                end_ts=100,
                layer="coarse",
            ),
            GraphNode(
                id="fine_1",
                lecture_id="lec",
                title="Fine 1",
                summary="",
                explanation="",
                source_span_ids=[],
                start_ts=10,
                end_ts=40,
                layer="fine",
                parent_coarse_id="coarse_1",
            ),
            GraphNode(
                id="fine_2",
                lecture_id="lec",
                title="Fine 2",
                summary="",
                explanation="",
                source_span_ids=[],
                start_ts=40,
                end_ts=70,
                layer="fine",
                parent_coarse_id="coarse_1",
            ),
        ],
        edges=[
            GraphEdge(
                id="edge_1",
                from_id="fine_2",
                to_id="fine_1",
                type="requires",
                reason="",
                evidence_span_ids=[],
            )
        ],
        spans=[],
        metadata={},
    )
    dot_text = build_stacked_layout_dot(graph)
    assert 'rank=same' in dot_text
    assert '"coarse_1" -> "fine_1"' in dot_text
    assert 'requires' not in dot_text


def test_rebuild_hierarchy_assigns_all_fine_nodes() -> None:
    spans = [
        Span(span_id="a_0", lecture_id="lec", timestamp=0, text="a", modality="audio"),
        Span(span_id="a_1", lecture_id="lec", timestamp=50, text="b", modality="audio"),
        Span(span_id="a_2", lecture_id="lec", timestamp=100, text="c", modality="audio"),
    ]
    nodes = [
        GraphNode(
            id="fine_1",
            lecture_id="lec",
            title="Topic A",
            summary="",
            explanation="",
            source_span_ids=["a_0"],
            start_ts=0,
            end_ts=40,
            layer="fine",
        ),
        GraphNode(
            id="fine_2",
            lecture_id="lec",
            title="Topic B",
            summary="",
            explanation="",
            source_span_ids=["a_1"],
            start_ts=40,
            end_ts=80,
            layer="fine",
        ),
        GraphNode(
            id="fine_3",
            lecture_id="lec",
            title="Topic C",
            summary="",
            explanation="",
            source_span_ids=["a_2"],
            start_ts=80,
            end_ts=120,
            layer="fine",
        ),
        GraphNode(
            id="coarse_1",
            lecture_id="lec",
            title="Block 1",
            summary="",
            explanation="",
            source_span_ids=["a_0", "a_1"],
            start_ts=0,
            end_ts=70,
            layer="coarse",
        ),
        GraphNode(
            id="coarse_2",
            lecture_id="lec",
            title="Block 2",
            summary="",
            explanation="",
            source_span_ids=["a_1", "a_2"],
            start_ts=70,
            end_ts=120,
            layer="coarse",
        ),
        GraphNode(
            id="coarse_empty",
            lecture_id="lec",
            title="Empty",
            summary="",
            explanation="",
            source_span_ids=["a_0"],
            start_ts=200,
            end_ts=240,
            layer="coarse",
        ),
    ]
    graph = Graph(lecture_id="lec", nodes=nodes, edges=[], spans=spans, metadata={})
    _rebuild_hierarchy(graph)
    fine_nodes = [n for n in graph.nodes if n.layer == "fine"]
    coarse_nodes = [n for n in graph.nodes if n.layer == "coarse"]
    assert all(n.parent_coarse_id for n in fine_nodes)
    assert "coarse_empty" not in {n.id for n in coarse_nodes}
    part_of_edges = [e for e in graph.edges if e.type == "part_of"]
    assert len(part_of_edges) == len(fine_nodes)


if __name__ == "__main__":
    test_windowing()
    test_edge_types()
    test_tutor_hops_and_citations()
    test_edge_cleanup_enforces_layer_rules()
    test_validator_passes_on_simple_hierarchy()
    test_retriever_prefers_semantic_match()
    test_rebuild_hierarchy_assigns_all_fine_nodes()
    test_temporal_renderer_orders_children_under_parents()
    test_temporal_renderer_stacked_layout_omits_semantic_edges()
    print(json.dumps({"ok": True}))
