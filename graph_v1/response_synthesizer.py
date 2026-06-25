"""ResponseSynthesizer: Convert structured graph data into conversational tutor answers.

Takes local anchors, bridge paths, and global summaries, and synthesizes
human-readable answers using conversational tutor strategies (Socratic method,
scaffolded feedback, etc.).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional
from dataclasses import asdict

from .llm import LLMClient
from .models import Graph, GraphNode, Span


SynthesisStrategy = Literal["socratic", "scaffolded", "direct"]


class ResponseSynthesizer:
    """Synthesize humanized tutor responses from structured graph data."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def synthesize(
        self,
        question: str,
        local_anchors: Dict[str, Any],
        bridge_paths: Dict[str, Any],
        global_summary: Optional[str] = None,
        strategy: SynthesisStrategy = "scaffolded",
    ) -> str:
        """Synthesize a conversational tutor answer.

        Args:
            question: The student's original question.
            local_anchors: Dict with {direct facts, timestamps, evidence}.
            bridge_paths: Dict with {upward_path, prerequisite_chains}.
            global_summary: High-level lecture theme or recap.
            strategy: Socratic (questions), scaffolded (step-by-step), direct (immediate).

        Returns:
            A humanized tutor answer string.
        """
        system_prompt = self._build_system_prompt(strategy)
        user_content = {
            "question": question,
            "local_anchors": local_anchors,
            "bridge_paths": bridge_paths,
            "global_summary": global_summary,
            "strategy": strategy,
        }

        if not self.llm.available():
            return self._fallback_synthesis(question, local_anchors, bridge_paths, global_summary)

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.chat_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_content)},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            answer = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
            return answer or self._fallback_synthesis(question, local_anchors, bridge_paths, global_summary)
        except Exception:
            return self._fallback_synthesis(question, local_anchors, bridge_paths, global_summary)

    def _build_system_prompt(self, strategy: SynthesisStrategy) -> str:
        """Build the LLM system prompt based on strategy."""
        base = (
            "You are an expert lecture tutor. Use the provided structured data to answer the student's question. "
            "Always be clear, concise, and directly address their question. Cite the evidence when available."
        )

        if strategy == "socratic":
            return (
                base
                + " Use the Socratic method: guide the student by asking clarifying questions, "
                "building on their existing knowledge, and encouraging them to discover answers themselves."
            )
        elif strategy == "scaffolded":
            return (
                base
                + " Use scaffolded feedback: break down the concept into steps, provide immediate context "
                "from prerequisites, and connect to the broader lecture theme."
            )
        else:  # direct
            return base + " Provide a direct, immediate answer with supporting evidence."

    def _fallback_synthesis(
        self,
        question: str,
        local_anchors: Dict[str, Any],
        bridge_paths: Dict[str, Any],
        global_summary: Optional[str] = None,
    ) -> str:
        """Fallback synthesis when LLM is unavailable."""
        parts = []

        # Start with global context if available
        if global_summary:
            parts.append(f"In the broader context of {global_summary}:")

        # Add prerequisite context from bridge paths
        prereqs = bridge_paths.get("prerequisite_chains", [])
        if prereqs:
            prereq_titles = [p.get("title", "concept") for p in prereqs]
            parts.append(f"To understand this, you should know: {', '.join(prereq_titles)}.")

        # Add local anchor facts
        direct_facts = local_anchors.get("direct_facts", [])
        if direct_facts:
            parts.append("Key facts: " + " ".join(str(f) for f in direct_facts))

        # Add evidence timing
        if local_anchors.get("start_ts") is not None and local_anchors.get("end_ts") is not None:
            start = local_anchors["start_ts"]
            end = local_anchors["end_ts"]
            parts.append(f"[See video/audio: {start:.1f}s - {end:.1f}s for details]")

        if not parts:
            return f"The lecture covers concepts related to your question. See the evidence spans for details."

        return " ".join(parts)

    def synthesize_with_evidence(
        self,
        question: str,
        ranked_nodes: List[Dict[str, Any]],
        evidence_spans: List[Dict[str, Any]],
        bridge_data: Optional[Dict[str, Any]] = None,
        global_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """High-level synthesis integrating nodes, spans, and bridges.

        Returns a structured response with answer, media coordinates, and path context.
        """
        # Extract local anchors from ranked nodes and spans
        local_anchors = {
            "top_node_id": ranked_nodes[0]["id"] if ranked_nodes else None,
            "top_node_title": ranked_nodes[0]["title"] if ranked_nodes else "",
            "direct_facts": [n["summary"] for n in ranked_nodes[:2]],
            "start_ts": min((e.get("timestamp") for e in evidence_spans), default=0.0),
            "end_ts": max((e.get("timestamp") for e in evidence_spans), default=0.0) + 10.0,
        }

        bridge_paths = bridge_data or {"upward_path": [], "prerequisite_chains": []}

        answer = self.synthesize(
            question=question,
            local_anchors=local_anchors,
            bridge_paths=bridge_paths,
            global_summary=global_summary,
            strategy="scaffolded",
        )

        return {
            "answer": answer,
            "media_coordinates": {
                "start_ts": local_anchors["start_ts"],
                "end_ts": local_anchors["end_ts"],
                "primary_modality": evidence_spans[0].get("modality", "audio") if evidence_spans else "audio",
            },
            "context_bridges": {
                "upward_path": bridge_paths.get("upward_path", []),
                "prerequisites": bridge_paths.get("prerequisite_chains", []),
            },
            "cited_nodes": [n["id"] for n in ranked_nodes[:3]],
            "cited_spans": [e.get("span_id") for e in evidence_spans[:5]],
        }
