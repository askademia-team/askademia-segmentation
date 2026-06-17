from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Literal

from .llm import LLMClient
from .models import Graph, GraphEdge, GraphNode, TutorFunctionResult, TutorSessionState
from .retrieval import GraphRetriever


QuestionIntent = Literal["definition", "prerequisite", "application", "example", "comparison", "broad_recap"]


class TutorRuntime:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.llm = LLMClient()
        self.retriever = GraphRetriever(graph, self.llm)
        self.node_by_id = {n.id: n for n in graph.nodes}
        self.edge_by_id = {e.id: e for e in graph.edges}

    def search_nodes(self, query: str, k: int) -> TutorFunctionResult:
        include_coarse = _classify_question_intent(query) == "broad_recap"
        results = self.retriever.rank_nodes(query, k=k, include_coarse=include_coarse, prefer_fine=True)
        return TutorFunctionResult("search_nodes", {"query": query, "k": k, "results": results})

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
        edge_types_set = set(edge_types or [])
        for e in self.graph.edges:
            if edge_types_set and e.type not in edge_types_set:
                continue
            if direction in ("out", "both") and e.from_id == node_id:
                out.append(asdict(e))
            if direction in ("in", "both") and e.to_id == node_id:
                out.append(asdict(e))
        return TutorFunctionResult("get_neighbors", {"node_id": node_id, "neighbors": out[:limit]})

    def get_evidence(self, node_id_or_edge_id: str, question: str = "") -> TutorFunctionResult:
        spans = {s.span_id: s for s in self.graph.spans}
        if node_id_or_edge_id.startswith("node_"):
            n = self.node_by_id.get(node_id_or_edge_id)
            if not n:
                return TutorFunctionResult("get_evidence", {}, ok=False, error="node not found")
            if question:
                evidence = self.retriever.rank_spans_for_node(question, n.id, k=5)
            else:
                evidence = [asdict(spans[sid]) for sid in n.source_span_ids if sid in spans][:5]
            return TutorFunctionResult("get_evidence", {"node_id": n.id, "evidence": evidence})
        e = self.edge_by_id.get(node_id_or_edge_id)
        if not e:
            return TutorFunctionResult("get_evidence", {}, ok=False, error="edge not found")
        evidence = [asdict(spans[sid]) for sid in e.evidence_span_ids if sid in spans]
        return TutorFunctionResult("get_evidence", {"edge_id": e.id, "evidence": evidence[:5]})

    def rank_nodes_for_question(self, question: str, candidate_ids: List[str]) -> TutorFunctionResult:
        ranked_rows = self.retriever.rank_nodes(
            question,
            candidate_ids=candidate_ids,
            k=max(1, len(candidate_ids)),
            include_coarse=True,
            prefer_fine=True,
        )
        ranked = [(row["id"], row["score"]) for row in ranked_rows]
        return TutorFunctionResult("rank_nodes_for_question", {"question": question, "ranked": ranked})

    def stop_and_answer(self, answer: str, cited_node_ids: List[str], cited_span_ids: List[str]) -> TutorFunctionResult:
        return TutorFunctionResult(
            "stop_and_answer",
            {"answer": answer, "cited_node_ids": cited_node_ids, "cited_span_ids": cited_span_ids},
        )

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

        search = self.search_nodes(question, top_k)
        trace.append(asdict(search))
        if not search.payload["results"]:
            return {
                "answer": "I could not find relevant lecture concepts yet. Can you clarify the concept name?",
                "citations": {"nodes": [], "spans": []},
                "trace": trace,
            }

        gathered_ids = [row["id"] for row in search.payload["results"]]
        visited: List[str] = []
        cited_spans: List[str] = []
        cited_nodes: List[str] = []
        edge_pref = self._edge_types_for_question(question, intent, session_state)

        for step in range(max_hops):
            ranked = self.rank_nodes_for_question(question, gathered_ids)
            trace.append(asdict(ranked))
            ranked_rows = ranked.payload.get("ranked", [])
            if not ranked_rows:
                break

            top_id, top_score = ranked_rows[0]
            if top_id not in visited:
                visited.append(top_id)
            if top_id not in cited_nodes:
                cited_nodes.append(top_id)

            node_res = self.get_node(top_id)
            trace.append(asdict(node_res))
            ev_res = self.get_evidence(top_id, question)
            trace.append(asdict(ev_res))
            if ev_res.ok:
                for ev in ev_res.payload.get("evidence", [])[:3]:
                    sid = ev["span_id"]
                    if sid not in cited_spans:
                        cited_spans.append(sid)

            decision = self._planner_decide(
                question=question,
                intent=intent,
                step=step,
                top_score=top_score,
                gathered_ids=gathered_ids,
                visited=visited,
                evidence_count=len(cited_spans),
                max_hops=max_hops,
            )
            trace.append({"function_name": "planner_decision", "payload": decision, "ok": True, "error": None})
            if decision["action"] == "answer":
                break

            seeds = decision["target_nodes"] or [nid for nid, _ in ranked_rows[:2]]
            expansions: List[str] = []
            for seed in seeds:
                neigh = self.get_neighbors(seed, decision["edge_types"], direction="both", limit=12)
                trace.append(asdict(neigh))
                if not neigh.ok:
                    continue
                for edge in neigh.payload.get("neighbors", []):
                    candidate = edge["to_id"] if edge["from_id"] == seed else edge["from_id"]
                    if candidate not in expansions and candidate not in gathered_ids:
                        expansions.append(candidate)

            if intent == "broad_recap":
                parent_expansions = self._coarse_parent_expansions(seeds)
                for cid in parent_expansions:
                    if cid not in expansions and cid not in gathered_ids:
                        expansions.append(cid)

            if not expansions:
                break

            reranked_expansions = self.retriever.rank_nodes(
                question,
                candidate_ids=expansions,
                k=min(6, len(expansions)),
                include_coarse=True,
                prefer_fine=True,
            )
            filtered = [row["id"] for row in reranked_expansions if row["score"] > 0.12]
            if not filtered:
                filtered = [row["id"] for row in reranked_expansions[:2]]
            for nid in filtered:
                if nid not in gathered_ids:
                    gathered_ids.append(nid)

        final_ranked = self.retriever.rank_nodes(
            question,
            candidate_ids=gathered_ids,
            k=min(4, len(gathered_ids)),
            include_coarse=True,
            prefer_fine=True,
        )
        answer_node_ids = [row["id"] for row in final_ranked[:2]]
        answer_nodes = [asdict(self.node_by_id[nid]) for nid in answer_node_ids if nid in self.node_by_id]
        evidence_rows = []
        for nid in answer_node_ids:
            ev = self.get_evidence(nid, question)
            trace.append(asdict(ev))
            if ev.ok:
                for row in ev.payload.get("evidence", [])[:2]:
                    evidence_rows.append(row)
                    if row["span_id"] not in cited_spans:
                        cited_spans.append(row["span_id"])

        answer = self.llm.synthesize_tutor_answer(question, answer_nodes, evidence_rows)
        cited_nodes = list(dict.fromkeys((cited_nodes + answer_node_ids)))[:3]
        cited_spans = cited_spans[:6]

        session_state.turn_count += 1
        session_state.recent_node_ids = (session_state.recent_node_ids + visited)[-20:]
        for v in visited:
            prev = session_state.concept_familiarity.get(v, "new")
            session_state.concept_familiarity[v] = "seen" if prev == "new" else prev

        stop = self.stop_and_answer(answer=answer, cited_node_ids=cited_nodes, cited_span_ids=cited_spans)
        trace.append(asdict(stop))

        return {
            "answer": answer,
            "citations": {"nodes": cited_nodes, "spans": cited_spans},
            "trace": trace,
            "session_state": asdict(session_state),
        }

    def _coarse_parent_expansions(self, seed_ids: List[str]) -> List[str]:
        out: List[str] = []
        for nid in seed_ids:
            node = self.node_by_id.get(nid)
            if node and node.parent_coarse_id and node.parent_coarse_id not in out:
                out.append(node.parent_coarse_id)
        return out

    def _edge_types_for_question(
        self,
        question: str,
        intent: QuestionIntent,
        session_state: TutorSessionState,
    ) -> List[str]:
        struggling_recent = any(session_state.concept_familiarity.get(nid) == "struggling" for nid in session_state.recent_node_ids[-3:])
        if intent == "prerequisite" or struggling_recent:
            return ["requires"]
        if intent in ("application", "example"):
            return ["example_of", "related_to", "references"]
        if intent == "comparison":
            return ["related_to", "references"]
        if intent == "broad_recap":
            return ["part_of", "related_to"]
        return ["related_to", "requires", "example_of", "references"]

    def _planner_decide(
        self,
        question: str,
        intent: QuestionIntent,
        step: int,
        top_score: float,
        gathered_ids: List[str],
        visited: List[str],
        evidence_count: int,
        max_hops: int,
    ) -> Dict[str, Any]:
        if step >= max_hops - 1:
            return {"action": "answer", "target_nodes": [], "edge_types": [], "stop_reason": "max_hops_reached"}
        if top_score >= 0.58 and evidence_count >= 2:
            return {"action": "answer", "target_nodes": [], "edge_types": [], "stop_reason": "high_confidence_and_evidence"}
        if evidence_count >= 4:
            return {"action": "answer", "target_nodes": [], "edge_types": [], "stop_reason": "enough_ranked_evidence"}
        target_nodes = list(dict.fromkeys(gathered_ids[:2] + visited[-1:]))[:2]
        return {
            "action": "traverse",
            "target_nodes": target_nodes,
            "edge_types": self._edge_types_for_question(question, intent, TutorSessionState(session_id="tmp", recent_node_ids=visited)),
            "stop_reason": "need_more_context",
        }


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
    from .models import Span

    spans = [Span(**s) for s in raw.get("spans", [])]
    return Graph(lecture_id=raw["lecture_id"], nodes=nodes, edges=edges, spans=spans, metadata=raw.get("metadata", {}))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run guided tutor QA over graph_v1 JSON")
    p.add_argument("--graph", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--session-id", default="session_1")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-hops", type=int, default=4)
    p.add_argument("--feedback-log", default="", help="Optional path to append weak-label QA feedback JSONL")
    p.add_argument("--user-rating", type=int, default=-1, help="Optional post-hoc rating (0/1) for answer quality")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    graph = load_graph(Path(args.graph))
    runtime = TutorRuntime(graph)
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
