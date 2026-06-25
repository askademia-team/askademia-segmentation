from __future__ import annotations

import math
import re
import networkx as nx
from dataclasses import asdict
from typing import Any, Dict, List, Sequence

from .llm import LLMClient
from .models import Graph, GraphNode, Span


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _lexical_score(query: str, text: str) -> float:
    q = _tokenize(query)
    t = _tokenize(text)
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(q))


def _cosine(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def node_retrieval_text(node: GraphNode) -> str:
    return f"{node.title}\n{node.summary}\n{node.explanation}"


def span_retrieval_text(span: Span) -> str:
    return span.text


def ensure_node_embeddings(graph: Graph, llm: LLMClient) -> None:
    missing = [n for n in graph.nodes if not n.embedding]
    if not missing:
        return
    texts = [node_retrieval_text(n) for n in missing]
    embeddings = llm.embed_texts(texts)
    for node, emb in zip(missing, embeddings):
        node.embedding = emb


class GraphRetriever:
    def __init__(self, graph: Graph, llm: LLMClient | None = None):
        self.graph = graph
        self.llm = llm or LLMClient()
        ensure_node_embeddings(self.graph, self.llm)
        self.node_by_id = {n.id: n for n in graph.nodes}
        self.span_by_id = {s.span_id: s for s in graph.spans}

    def query_embedding(self, query: str) -> List[float]:
        return self.llm.embed_texts([query])[0]

    def rank_nodes(
        self,
        query: str,
        *,
        candidate_ids: List[str] | None = None,
        k: int = 5,
        include_coarse: bool = True,
        prefer_fine: bool = True,
    ) -> List[Dict[str, Any]]:
        q_emb = self.query_embedding(query)
        candidates = [self.node_by_id[cid] for cid in candidate_ids if cid in self.node_by_id] if candidate_ids else list(self.graph.nodes)
        scored: List[Dict[str, Any]] = []
        broad_query = _is_broad_query(query)
        for node in candidates:
            if not include_coarse and node.layer != "fine":
                continue
            embedding_score = _cosine(q_emb, node.embedding)
            lexical = _lexical_score(query, node_retrieval_text(node))
            title_boost = _lexical_score(query, node.title) * 0.15
            layer_bias = 0.0
            if prefer_fine and node.layer == "fine":
                layer_bias += 0.06
            if broad_query and node.layer != "fine":
                layer_bias += 0.08
            score = 0.65 * embedding_score + 0.25 * lexical + title_boost + layer_bias
            scored.append(
                {
                    **asdict(node),
                    "score": score,
                    "embedding_score": embedding_score,
                    "lexical_score": lexical,
                }
            )
        scored.sort(key=lambda row: row["score"], reverse=True)
        return scored[:k]

    def rank_spans_for_node(self, query: str, node_id: str, *, k: int = 5) -> List[Dict[str, Any]]:
        node = self.node_by_id.get(node_id)
        if not node:
            return []
        spans = [self.span_by_id[sid] for sid in node.source_span_ids if sid in self.span_by_id]
        if not spans:
            return []
        q_emb = self.query_embedding(query)
        span_embs = self.llm.embed_texts([span_retrieval_text(s) for s in spans])
        rows = []
        for span, emb in zip(spans, span_embs):
            lexical = _lexical_score(query, span.text)
            score = 0.75 * _cosine(q_emb, emb) + 0.25 * lexical
            rows.append({**asdict(span), "score": score, "lexical_score": lexical})
        rows.sort(key=lambda row: row["score"], reverse=True)
        return rows[:k]


def _is_broad_query(query: str) -> bool:
    q = query.lower()
    return any(
        token in q
        for token in [
            "overview",
            "big picture",
            "broadly",
            "recap",
            "what did we cover",
            "summary",
            "lecture about",
        ]
    )


def _graph_component_msts(graph: Graph) -> List[Dict[str, Any]]:
    g = nx.Graph()
    for node in graph.nodes:
        g.add_node(node.id)
    for edge in graph.edges:
        if edge.from_id == edge.to_id:
            continue
        weight = float(edge.edge_confidence or 0.0)
        g.add_edge(edge.from_id, edge.to_id, weight=weight, edge_id=edge.id, edge_type=edge.type)

    components = []
    for comp_idx, component_nodes in enumerate(nx.connected_components(g), start=1):
        sub = g.subgraph(component_nodes)
        if sub.number_of_nodes() == 0:
            continue
        if sub.number_of_nodes() == 1:
            mst = sub
        else:
            mst = nx.maximum_spanning_tree(sub, weight="weight")
        edge_rows = []
        for u, v, data in mst.edges(data=True):
            edge_rows.append({
                "from_id": u,
                "to_id": v,
                "edge_id": data.get("edge_id"),
                "type": data.get("edge_type"),
                "weight": data.get("weight", 0.0),
            })
        components.append(
            {
                "component_id": f"component_{comp_idx}",
                "node_ids": list(component_nodes),
                "edge_rows": edge_rows,
            }
        )
    return components


def summarize_broad_components(graph: Graph, llm: LLMClient, query: str) -> List[Dict[str, Any]]:
    components = _graph_component_msts(graph)
    node_by_id = {n.id: n for n in graph.nodes}
    summaries = []
    for comp in components:
        nodes = [asdict(node_by_id[nid]) for nid in comp["node_ids"] if nid in node_by_id]
        if not nodes:
            continue
        component_summary = llm.summarize_mst_component(query, comp["component_id"], nodes, comp["edge_rows"])
        summaries.append(
            {
                "component_id": comp["component_id"],
                "summary": component_summary,
                "nodes": nodes,
                "edges": comp["edge_rows"],
            }
        )
    return summaries


def extract_contextual_bridge(
    graph: Graph,
    node_id: str,
    max_path_length: int = 5,
) -> Dict[str, Any]:
    """Extract contextual bridges showing why a concept matters.
    
    Uses Dijkstra's algorithm to find:
    1. Shortest path from node upward to root parent
    2. Prerequisite chains (incoming 'requires' edges)
    
    Returns a structured bridge showing concept importance and prerequisites.
    """
    node_by_id = {n.id: n for n in graph.nodes}
    if node_id not in node_by_id:
        return {}
    
    # Build directed graph of dependencies (requires edges + part_of edges)
    dep_graph = nx.DiGraph()
    for n in graph.nodes:
        dep_graph.add_node(n.id)
    for e in graph.edges:
        if e.type in ("requires", "part_of"):
            weight = 1.0 / max(0.1, float(e.edge_confidence or 0.5))
            dep_graph.add_edge(e.from_id, e.to_id, weight=weight, edge=e)
    
    bridges = {"node_id": node_id, "upward_path": [], "prerequisite_chains": []}
    
    # Find root parents (part_of edges going upward)
    try:
        root_nodes = [n.id for n in graph.nodes if n.layer != "fine" and n.id != node_id]
        for root in root_nodes:
            if nx.has_path(dep_graph, node_id, root):
                try:
                    path = nx.shortest_path(dep_graph, node_id, root, weight="weight")
                    if len(path) <= max_path_length:
                        path_nodes = [node_by_id.get(nid) for nid in path]
                        bridges["upward_path"] = [
                            {"id": n.id, "title": n.title, "layer": n.layer} for n in path_nodes if n
                        ]
                        break
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
    except Exception:
        pass
    
    # Find prerequisite chains (requires edges, reversed direction)
    try:
        reverse_graph = dep_graph.reverse()
        for target in dep_graph.successors(node_id):
            edge_data = dep_graph.get_edge_data(node_id, target)
            if edge_data and edge_data.get("edge") and edge_data["edge"].type == "requires":
                target_node = node_by_id.get(target)
                if target_node:
                    bridges["prerequisite_chains"].append({
                        "prerequisite_id": target,
                        "title": target_node.title,
                        "reason": edge_data["edge"].reason,
                        "confidence": float(edge_data["edge"].edge_confidence or 0.5),
                    })
    except Exception:
        pass
    
    return bridges


def pool_and_score_spans(
    graph: Graph,
    node_id: str,
    query: str,
    llm: LLMClient,
    window_size: int = 3,
) -> Dict[str, Any]:
    """Pool and score spans for a node using rolling window density.
    
    Evaluates spans with a rolling window approach, calculating combined
    semantic density for match. Returns explicit start_ts and end_ts
    for media playback (audio/video seeking).
    """
    node = next((n for n in graph.nodes if n.id == node_id), None)
    if not node:
        return {}
    
    span_map = {s.span_id: s for s in graph.spans}
    node_spans = [span_map[sid] for sid in node.source_span_ids if sid in span_map]
    if not node_spans:
        return {"node_id": node_id, "start_ts": node.start_ts, "end_ts": node.end_ts, "pooled_spans": []}
    
    node_spans_sorted = sorted(node_spans, key=lambda s: s.timestamp)
    q_emb = llm.embed_texts([query])[0] if llm else None
    
    # Rolling window scoring
    scored_windows: List[Dict[str, Any]] = []
    for i in range(len(node_spans_sorted)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(node_spans_sorted), i + window_size // 2 + 1)
        window_spans = node_spans_sorted[window_start:window_end]
        
        # Compute combined density
        combined_text = " ".join(s.text for s in window_spans)
        lexical_score = _lexical_score(query, combined_text)
        semantic_score = 0.0
        if q_emb:
            window_embs = llm.embed_texts([s.text for s in window_spans])
            semantic_score = max([_cosine(q_emb, emb) for emb in window_embs], default=0.0)
        
        combined_score = 0.6 * semantic_score + 0.4 * lexical_score
        scored_windows.append({
            "window_start_idx": window_start,
            "window_end_idx": window_end,
            "center_span": asdict(node_spans_sorted[i]),
            "density_score": combined_score,
            "window_timestamp": node_spans_sorted[i].timestamp,
        })
    
    scored_windows.sort(key=lambda w: w["density_score"], reverse=True)
    best_windows = scored_windows[:3]
    
    if best_windows:
        start_ts = min(w["window_timestamp"] for w in best_windows)
        end_ts = max(w["window_timestamp"] for w in best_windows) + 5.0
    else:
        start_ts = node.start_ts
        end_ts = node.end_ts
    
    return {
        "node_id": node_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "pooled_spans": best_windows,
        "total_spans": len(node_spans),
    }
