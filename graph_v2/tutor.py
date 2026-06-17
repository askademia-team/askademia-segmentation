from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal

from graph_v1.models import Graph, GraphEdge, GraphNode, TutorFunctionResult, TutorSessionState
from graph_v1.retrieval import GraphRetriever

from .llm import GraphV2LLMClient


QuestionIntent = Literal["definition", "prerequisite", "application", "example", "comparison", "broad_recap"]


class TutorRuntimeV2:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.llm = GraphV2LLMClient()
        self.retriever = GraphRetriever(graph, self.llm)
        self.node_by_id = {n.id: n for n in graph.nodes}
        self.edge_by_id = {e.id: e for e in graph.edges}
        self.coarse_nodes = [n for n in graph.nodes if n.layer == "coarse"]
        self.fine_nodes = [n for n in graph.nodes if n.layer == "fine"]

    def search_coarse_nodes(self, query: str, k: int) -> TutorFunctionResult:
        coarse_ids = [n.id for n in self.coarse_nodes]
        results = self.retriever.rank_nodes(query, candidate_ids=coarse_ids, k=k, include_coarse=True, prefer_fine=False)
        return TutorFunctionResult("search_coarse_nodes", {"query": query, "k": k, "results": results})

    def search_fine_nodes_within_coarse(self, query: str, coarse_ids: List[str], k: int) -> TutorFunctionResult:
        fine_ids = [n.id for n in self.fine_nodes if n.parent_coarse_id in set(coarse_ids)]
        results = self.retriever.rank_nodes(query, candidate_ids=fine_ids, k=k, include_coarse=False, prefer_fine=True)
        return TutorFunctionResult("search_fine_nodes_within_coarse", {"query": query, "coarse_ids": coarse_ids, "k": k, "results": results})

    def get_node(self, node_id: str) -> TutorFunctionResult:
        node = self.node_by_id.get(node_id)
        if not node:
            return TutorFunctionResult("get_node", {}, ok=False, error=f"node not found: {node_id}")
        return TutorFunctionResult("get_node", asdict(node))

    def get_neighbors(
        self,
        node_id: str,
        edge_types: List[str] | None,
        direction: Literal["out", "in", "both"] = "both",
        limit: int = 10,
    ) -> TutorFunctionResult:
        out = []
        seed = self.node_by_id.get(node_id)
        edge_types_set = set(edge_types or [])
        for e in self.graph.edges:
            if edge_types_set and e.type not in edge_types_set:
                continue
            if direction in ("out", "both") and e.from_id == node_id:
                other = self.node_by_id.get(e.to_id)
                if self._neighbor_allowed(seed, other, e):
                    out.append(asdict(e))
            if direction in ("in", "both") and e.to_id == node_id:
                other = self.node_by_id.get(e.from_id)
                if self._neighbor_allowed(seed, other, e):
                    out.append(asdict(e))
        return TutorFunctionResult("get_neighbors", {"node_id": node_id, "neighbors": out[:limit]})

    def get_evidence(self, node_id_or_edge_id: str, question: str = "") -> TutorFunctionResult:
        spans = {s.span_id: s for s in self.graph.spans}
        if node_id_or_edge_id.startswith("node_"):
            n = self.node_by_id.get(node_id_or_edge_id)
            if not n:
                return TutorFunctionResult("get_evidence", {}, ok=False, error="node not found")
            evidence = self.retriever.rank_spans_for_node(question, n.id, k=5) if question else [asdict(spans[sid]) for sid in n.source_span_ids if sid in spans][:5]
            return TutorFunctionResult("get_evidence", {"node_id": n.id, "evidence": evidence})
        e = self.edge_by_id.get(node_id_or_edge_id)
        if not e:
            return TutorFunctionResult("get_evidence", {}, ok=False, error="edge not found")
        evidence = [asdict(spans[sid]) for sid in e.evidence_span_ids if sid in spans][:5]
        return TutorFunctionResult("get_evidence", {"edge_id": e.id, "evidence": evidence})

    def stop_and_answer(self, answer: str, cited_node_ids: List[str], cited_span_ids: List[str]) -> TutorFunctionResult:
        return TutorFunctionResult("stop_and_answer", {"answer": answer, "cited_node_ids": cited_node_ids, "cited_span_ids": cited_span_ids})

    def run_guided_answer(
        self,
        question: str,
        session_state: TutorSessionState,
        top_k: int = 5,
        max_hops: int = 4,
    ) -> Dict[str, Any]:
        trace: List[Dict[str, Any]] = []
        intent = _classify_question_intent(question)
        trace.append({"function_name": "classify_intent", "payload": {"question": question, "intent": intent}, "ok": True, "error": None})

        coarse_search = self.search_coarse_nodes(question, max(3, top_k))
        trace.append(asdict(coarse_search))
        coarse_results = coarse_search.payload.get("results", [])
        if not coarse_results:
            return {"answer": "I could not find a relevant lecture topic yet.", "citations": {"nodes": [], "spans": []}, "trace": trace}

        selected_coarse_ids = self._select_coarse_topics(intent, coarse_results)
        trace.append({"function_name": "select_coarse_topics", "payload": {"selected_coarse_ids": selected_coarse_ids}, "ok": True, "error": None})

        fine_search = self.search_fine_nodes_within_coarse(question, selected_coarse_ids, max(6, top_k))
        trace.append(asdict(fine_search))
        fine_results = fine_search.payload.get("results", [])

        gathered_ids = selected_coarse_ids + [row["id"] for row in fine_results]
        visited: List[str] = []
        cited_nodes: List[str] = []
        cited_spans: List[str] = []

        for step in range(max_hops):
            ranked = self.retriever.rank_nodes(
                question,
                candidate_ids=gathered_ids,
                k=min(8, len(gathered_ids)),
                include_coarse=True,
                prefer_fine=(intent != "broad_recap"),
            )
            trace.append({"function_name": "rank_gathered", "payload": {"ranked": [(r['id'], r['score']) for r in ranked]}, "ok": True, "error": None})
            if not ranked:
                break
            top = ranked[0]
            top_id = top["id"]
            top_score = top["score"]
            if top_id not in visited:
                visited.append(top_id)
            if top_id not in cited_nodes:
                cited_nodes.append(top_id)
            ev = self.get_evidence(top_id, question)
            trace.append(asdict(ev))
            if ev.ok:
                for row in ev.payload.get("evidence", [])[:3]:
                    if row["span_id"] not in cited_spans:
                        cited_spans.append(row["span_id"])
            if step >= max_hops - 1 or (top_score >= 0.58 and len(cited_spans) >= 2):
                break
            seeds = [r["id"] for r in ranked[:2]]
            expansions: List[str] = []
            edge_types = self._edge_types_for_intent(intent)
            for seed in seeds:
                neigh = self.get_neighbors(seed, edge_types, direction="both", limit=12)
                trace.append(asdict(neigh))
                if not neigh.ok:
                    continue
                for edge in neigh.payload.get("neighbors", []):
                    nid = edge["to_id"] if edge["from_id"] == seed else edge["from_id"]
                    if nid not in gathered_ids and nid not in expansions:
                        expansions.append(nid)
            reranked = self.retriever.rank_nodes(
                question,
                candidate_ids=expansions,
                k=min(4, len(expansions)),
                include_coarse=True,
                prefer_fine=(intent != "broad_recap"),
            )
            for row in reranked:
                if row["id"] not in gathered_ids:
                    gathered_ids.append(row["id"])

        final_ranked = self.retriever.rank_nodes(
            question,
            candidate_ids=gathered_ids,
            k=3,
            include_coarse=True,
            prefer_fine=(intent != "broad_recap"),
        )
        answer_node_ids = [row["id"] for row in final_ranked[:2]]
        answer_nodes = [asdict(self.node_by_id[nid]) for nid in answer_node_ids if nid in self.node_by_id]
        evidence_rows: List[Dict[str, Any]] = []
        for nid in answer_node_ids:
            ev = self.get_evidence(nid, question)
            trace.append(asdict(ev))
            if ev.ok:
                for row in ev.payload.get("evidence", [])[:2]:
                    evidence_rows.append(row)
                    if row["span_id"] not in cited_spans:
                        cited_spans.append(row["span_id"])
        answer = self.llm.synthesize_tutor_answer(question, answer_nodes, evidence_rows)
        session_state.turn_count += 1
        session_state.recent_node_ids = (session_state.recent_node_ids + visited)[-20:]
        for node_id in visited:
            prev = session_state.concept_familiarity.get(node_id, "new")
            session_state.concept_familiarity[node_id] = "seen" if prev == "new" else prev
        stop = self.stop_and_answer(answer, list(dict.fromkeys(cited_nodes + answer_node_ids))[:3], cited_spans[:6])
        trace.append(asdict(stop))
        return {
            "answer": answer,
            "citations": {"nodes": stop.payload["cited_node_ids"], "spans": stop.payload["cited_span_ids"]},
            "trace": trace,
            "session_state": asdict(session_state),
        }

    def _neighbor_allowed(self, seed: GraphNode | None, other: GraphNode | None, edge: GraphEdge) -> bool:
        if not seed or not other:
            return False
        if edge.type == "part_of":
            return seed.layer != other.layer
        if seed.layer != other.layer:
            return False
        if seed.layer == "fine" and seed.parent_coarse_id != other.parent_coarse_id:
            return False
        return True

    def _select_coarse_topics(self, intent: QuestionIntent, coarse_results: List[Dict[str, Any]]) -> List[str]:
        if intent in {"comparison", "broad_recap"}:
            return [row["id"] for row in coarse_results[:2]]
        return [coarse_results[0]["id"]]

    def _edge_types_for_intent(self, intent: QuestionIntent) -> List[str]:
        if intent == "prerequisite":
            return ["requires"]
        if intent in {"application", "example"}:
            return ["example_of", "related_to", "references"]
        if intent == "comparison":
            return ["related_to", "references"]
        if intent == "broad_recap":
            return ["related_to", "part_of"]
        return ["related_to", "requires", "example_of", "references"]


def _classify_question_intent(question: str) -> QuestionIntent:
    q = question.lower()
    if any(k in q for k in ["before", "prereq", "prerequisite", "need to know", "foundation"]):
        return "prerequisite"
    if any(k in q for k in ["example", "for instance", "e.g."]):
        return "example"
    if any(k in q for k in ["how does", "how do", "apply", "used for", "help before"]):
        return "application"
    if any(k in q for k in ["compare", "contrast", "difference", "versus", "vs"]):
        return "comparison"
    if any(k in q for k in ["overview", "summary", "recap", "what did we cover", "big picture"]):
        return "broad_recap"
    return "definition"


def load_graph(path: Path) -> Graph:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    nodes = [GraphNode(**n) for n in raw.get("nodes", [])]
    edges = [GraphEdge(**e) for e in raw.get("edges", [])]
    from graph_v1.models import Span

    spans = [Span(**s) for s in raw.get("spans", [])]
    return Graph(lecture_id=raw["lecture_id"], nodes=nodes, edges=edges, spans=spans, metadata=raw.get("metadata", {}))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Graph V2 tutor QA")
    p.add_argument("--graph", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--session-id", default="session_v2")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-hops", type=int, default=4)
    p.add_argument("--feedback-log", default="")
    p.add_argument("--user-rating", type=int, default=-1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    graph = load_graph(Path(args.graph))
    runtime = TutorRuntimeV2(graph)
    state = TutorSessionState(session_id=args.session_id)
    result = runtime.run_guided_answer(args.question, state, top_k=args.top_k, max_hops=args.max_hops)
    print(json.dumps(result, indent=2))
    if args.feedback_log:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": args.session_id,
            "question": args.question,
            "top_k": args.top_k,
            "max_hops": args.max_hops,
            "answer": result.get("answer", ""),
            "cited_nodes": result.get("citations", {}).get("nodes", []),
            "cited_spans": result.get("citations", {}).get("spans", []),
            "trace": result.get("trace", []),
            "user_rating": args.user_rating if args.user_rating in (0, 1) else None,
        }
        fp = Path(args.feedback_log)
        fp.parent.mkdir(parents=True, exist_ok=True)
        with fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")


if __name__ == "__main__":
    main()
