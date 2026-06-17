from __future__ import annotations

import math
import re
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
            if not include_coarse and node.layer == "coarse":
                continue
            embedding_score = _cosine(q_emb, node.embedding)
            lexical = _lexical_score(query, node_retrieval_text(node))
            title_boost = _lexical_score(query, node.title) * 0.15
            layer_bias = 0.0
            if prefer_fine and node.layer == "fine":
                layer_bias += 0.06
            if broad_query and node.layer == "coarse":
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
