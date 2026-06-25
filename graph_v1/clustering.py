from __future__ import annotations

import re
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from .llm import LLMClient
from .models import Graph, GraphEdge, GraphHyperedge, GraphNode


def _layer_rank(layer: str) -> int:
    if layer == "fine":
        return 0
    if layer == "coarse":
        return 1
    if layer.startswith("layer_"):
        try:
            return int(layer.split("_", 1)[1])
        except Exception:
            return 2
    return 1


def _next_hyperedge_id(graph: Graph) -> str:
    existing = [e.id for e in graph.hyperedges if e.id.startswith("hyperedge_")]
    nums = [int(re.split(r"_", eid, maxsplit=1)[1]) for eid in existing if re.split(r"_", eid, maxsplit=1)[1].isdigit()]
    return f"hyperedge_{max(nums, default=0) + 1}"


def _next_node_id(graph: Graph, prefix: str) -> str:
    existing = [n.id for n in graph.nodes if n.id.startswith(prefix)]
    nums = []
    for nid in existing:
        suffix = nid[len(prefix) :]
        if suffix.startswith("_"):
            suffix = suffix[1:]
        if suffix.isdigit():
            nums.append(int(suffix))
    return f"{prefix}_{max(nums, default=0) + 1}"


def _next_edge_id(graph: Graph) -> str:
    nums = [int(e.id.split("_", 1)[1]) for e in graph.edges if e.id.startswith("edge_") and e.id.split("_", 1)[1].isdigit()]
    return f"edge_{max(nums, default=0) + 1}"


def _summarize_cluster_payload(
    llm: LLMClient,
    cluster_id: str,
    layer_name: str,
    member_nodes: List[GraphNode],
    internal_edges: List[GraphEdge],
) -> Dict[str, str]:
    return llm.summarize_hyperedge_to_node(
        cluster_id=cluster_id,
        layer_name=layer_name,
        member_nodes=[asdict(n) for n in member_nodes],
        internal_edges=[asdict(e) for e in internal_edges],
    )


def _select_best_gmm(embeddings: np.ndarray, max_components: int = 8) -> Tuple[GaussianMixture, int, np.ndarray]:
    n = len(embeddings)
    max_components = min(max_components, n)
    if max_components < 1:
        raise ValueError("At least one embedding is required")
    best_model: Optional[GaussianMixture] = None
    best_bic = float("inf")
    best_k = 1
    best_labels = np.zeros(n, dtype=int)

    for k in range(1, max_components + 1):
        try:
            model = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
            model.fit(embeddings)
            bic = model.bic(embeddings)
        except Exception:
            continue
        labels = model.predict(embeddings)
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_k = k
            best_labels = labels

    if best_model is None:
        raise RuntimeError("Unable to fit GMM for clustering")
    return best_model, best_k, best_labels


def _embedding_matrix(nodes: List[GraphNode], llm: LLMClient) -> np.ndarray:
    texts = [f"{n.title}\n{n.summary}\n{n.explanation}" for n in nodes]
    missing = [idx for idx, n in enumerate(nodes) if not n.embedding]
    if missing:
        embeddings = llm.embed_texts([texts[i] for i in missing])
        for idx, emb in zip(missing, embeddings):
            nodes[idx].embedding = emb
    return np.array([np.array(n.embedding or llm._fallback_embedding(texts[i]), dtype=float) for i, n in enumerate(nodes)])


def _group_cluster_edges(graph: Graph, member_ids: List[str]) -> List[GraphEdge]:
    members = set(member_ids)
    return [e for e in graph.edges if e.from_id in members and e.to_id in members and e.type != "part_of"]


def _node_span_union(nodes: List[GraphNode]) -> List[str]:
    span_ids = []
    for n in nodes:
        span_ids.extend(n.source_span_ids)
    return sorted(set(span_ids))


def _node_time_bounds(nodes: List[GraphNode]) -> Tuple[float, float]:
    start_ts = min(n.start_ts for n in nodes)
    end_ts = max(n.end_ts for n in nodes)
    return start_ts, end_ts


def _should_create_clusters(n_nodes: int, n_clusters: int) -> bool:
    return n_clusters > 1 and n_clusters < n_nodes


def _cluster_layer(
    graph: Graph,
    nodes: List[GraphNode],
    llm: LLMClient,
    layer_name: str,
    max_clusters: int = 8,
) -> List[GraphNode]:
    if len(nodes) < 2:
        return []
    embeddings = _embedding_matrix(nodes, llm)
    _, best_k, labels = _select_best_gmm(embeddings, max_components=max_clusters)
    if not _should_create_clusters(len(nodes), best_k) and best_k != 1:
        # No meaningful clustering when every node becomes its own cluster.
        return []

    clusters: Dict[int, List[GraphNode]] = {}
    for node, label in zip(nodes, labels):
        clusters.setdefault(int(label), []).append(node)

    created: List[GraphNode] = []
    for cluster_idx, member_nodes in sorted(clusters.items()):
        if len(member_nodes) == 1 and best_k == len(nodes):
            # preserve singleton identity when clustering is not compressive
            continue
        cluster_id = _next_node_id(graph, f"cluster_{layer_name}")
        internal_edges = _group_cluster_edges(graph, [n.id for n in member_nodes])
        summary_payload = _summarize_cluster_payload(llm, cluster_id, layer_name, member_nodes, internal_edges)
        start_ts, end_ts = _node_time_bounds(member_nodes)
        source_span_ids = _node_span_union(member_nodes)
        cluster_node = GraphNode(
            id=cluster_id,
            lecture_id=graph.lecture_id,
            title=summary_payload["title"],
            summary=summary_payload["summary"],
            explanation=summary_payload["explanation"],
            source_span_ids=source_span_ids,
            start_ts=start_ts,
            end_ts=end_ts,
            layer=layer_name,
            parent_coarse_id=None,
            cluster_member_ids=[n.id for n in member_nodes],
        )
        graph.nodes.append(cluster_node)
        for member in member_nodes:
            member.parent_coarse_id = cluster_id
            edge_id = _next_edge_id(graph)
            graph.edges.append(
                GraphEdge(
                    id=edge_id,
                    from_id=member.id,
                    to_id=cluster_id,
                    type="part_of",
                    reason=f"Semantic cluster membership within {layer_name}",
                    evidence_span_ids=member.source_span_ids[:3],
                    edge_confidence=0.95,
                    evidence_count=min(3, len(member.source_span_ids)),
                )
            )
        graph.hyperedges.append(
            GraphHyperedge(
                id=_next_hyperedge_id(graph),
                layer=layer_name,
                cluster_node_id=cluster_id,
                member_node_ids=[n.id for n in member_nodes],
                internal_edge_ids=[e.id for e in internal_edges],
                title=summary_payload["title"],
                summary=summary_payload["summary"],
                explanation=summary_payload["explanation"],
                source_span_ids=source_span_ids,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        )
        created.append(cluster_node)
    return created


def build_semantic_cluster_hierarchy(
    graph: Graph,
    llm: LLMClient,
    max_cluster_layers: int = 2,
    max_clusters: int = 8,
) -> None:
    current_nodes = [n for n in graph.nodes if n.layer == "fine"]
    if len(current_nodes) < 2:
        return

    for layer_idx in range(1, max_cluster_layers + 1):
        layer_name = "coarse" if layer_idx == 1 else f"layer_{layer_idx}"
        next_nodes = _cluster_layer(graph, current_nodes, llm, layer_name, max_clusters=max_clusters)
        if not next_nodes:
            break
        current_nodes = next_nodes

    graph.metadata["semantic_cluster_layers"] = {
        "created_layers": [n.layer for n in graph.nodes if n.layer != "fine" and n.id.startswith("cluster_")],
        "max_cluster_layers": max_cluster_layers,
        "max_clusters": max_clusters,
    }
