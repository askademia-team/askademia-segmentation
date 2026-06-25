from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional

from .llm import LLMClient


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
        low_confidence: bool = False,
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
        cleaned_question = self._clean_text(question)
        cleaned_local_anchors = self._clean_anchor_payload(local_anchors)
        cleaned_bridge_paths = self._clean_bridge_payload(bridge_paths)
        cleaned_global_summary = self._clean_text(global_summary or "") or None
        user_content = self._build_plain_text_payload(
            question=cleaned_question,
            local_anchors=cleaned_local_anchors,
            bridge_paths=cleaned_bridge_paths,
            global_summary=cleaned_global_summary,
            strategy=strategy,
            low_confidence=low_confidence,
        )

        if not self.llm.available():
            return self._fallback_synthesis(question, local_anchors, bridge_paths, global_summary, low_confidence=low_confidence)

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.chat_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=600,
            )
            answer = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
            return answer or self._fallback_synthesis(question, local_anchors, bridge_paths, global_summary, low_confidence=low_confidence)
        except Exception:
            return self._fallback_synthesis(question, local_anchors, bridge_paths, global_summary, low_confidence=low_confidence)

    def _build_system_prompt(self, strategy: SynthesisStrategy) -> str:
        return (
            "You are an expert tutor. Your primary goal is to provide direct, natural, and human-like answers.\n\n"
            "CRITICAL FORMATTING RULES:\n"
            "1. NEVER include raw system logs, node names, timestamps, confidence scores, or metadata "
            "(e.g., do not say \"From lecture_24_n42\" or \"[4434s]\").\n"
            "2. DO NOT copy and paste large walls of transcript text or repeat conversational filler from the data source.\n"
            "3. Your output must consist ONLY of the clear, direct answer to the student's question, broken down logically with a few bullet points if necessary.\n"
            "4. If the retrieved context contains messy or repeated text (like OCR errors or repetitive video transcripts), "
            "silently synthesize the core meaning and rewrite it in clean, grammatical English.\n\n"
            "Tone: Professional, direct, and conversational-as if you are explaining it to a student in office hours."
        )

    def _build_plain_text_payload(
        self,
        question: str,
        local_anchors: Dict[str, Any],
        bridge_paths: Dict[str, Any],
        global_summary: Optional[str],
        strategy: SynthesisStrategy,
        low_confidence: bool,
    ) -> str:
        cleaned_facts = [
            self._clean_text(str(f))
            for f in (local_anchors.get("direct_facts", []) or [])
            if isinstance(f, str) and len(str(f).split()) > 4
        ]
        evidence_items = (local_anchors.get("evidence", []) or local_anchors.get("snippets", []))
        cleaned_snippets = [
            self._clean_text(item.get("text", "")) if isinstance(item, dict) else self._clean_text(str(item))
            for item in evidence_items
        ]
        cleaned_snippets = [s for s in cleaned_snippets if s and len(s.split()) > 4 and not self._looks_like_filler(s)]

        bridge_titles = [
            self._clean_text(str(p.get("title", "")))
            for p in bridge_paths.get("upward_path", [])
            if isinstance(p, dict) and p.get("title")
        ]
        prereq_titles = [
            self._clean_text(str(p.get("title", "")))
            for p in bridge_paths.get("prerequisite_chains", [])
            if isinstance(p, dict) and p.get("title")
        ]

        context_payload: List[str] = []
        if global_summary and not low_confidence:
            context_payload.append(f"OVERARCHING LECTURE THEME:\n{global_summary}")
        if cleaned_facts and not low_confidence:
            context_payload.append(
                "CORE TECHNICAL POINTS DISCUSSED:\n" + "\n".join(f"- {f}" for f in cleaned_facts[:6])
            )
        if (bridge_titles or prereq_titles) and not low_confidence:
            combined_links = sorted({t for t in bridge_titles + prereq_titles if t})
            if combined_links:
                context_payload.append(
                    "CONCEPTUAL PREREQUISITE AND LINKED TOPICS:\n"
                    f"This localized section builds directly into: {', '.join(combined_links)}"
                )
        if cleaned_snippets and not low_confidence:
            context_payload.append(
                "DIRECT LECTURE TRANSCRIPT EXCERPTS FOR CONTEXT:\n"
                + "\n".join(f'"{s}"' for s in cleaned_snippets[:2])
            )

        flat_context_string = "\n\n".join(context_payload).strip()
        if low_confidence:
            flat_context_string = "No explicit transcript snippets available."

        return (
            f"STUDENT QUESTION:\n{question}\n\n"
            "LECTURE REFERENCE MATERIAL:\n"
            f"{flat_context_string if flat_context_string else 'No explicit transcript snippets available.'}\n\n"
            "TASK:\n"
            "Using the lecture reference material provided above, write a fluid, highly educational textbook-style "
            "paragraph that explicitly answers the student's question. Do not state how you found the information "
            "or use meta-commentary."
        )

    def _clean_text(self, text: str) -> str:
        text = str(text or "")
        text = re.sub(r"(?i)\b(um+|uh+|er+|ah+|like|you know|sort of|kind of|basically|so yeah|okay so|i think|maybe)\b", " ", text)
        text = re.sub(r"\b(tricky early on|hard to interpret|not sure|i guess)\b", " ", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
        return text

    def _clean_anchor_payload(self, anchors: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in (anchors or {}).items():
            if isinstance(value, str):
                cleaned_value = self._clean_text(value)
                if cleaned_value and not self._looks_like_filler(cleaned_value) and len(cleaned_value.split()) >= 5:
                    cleaned[key] = cleaned_value
            elif isinstance(value, list):
                cleaned_list = []
                for item in value:
                    if isinstance(item, str):
                        cleaned_item = self._clean_text(item)
                        if cleaned_item and not self._looks_like_filler(cleaned_item) and len(cleaned_item.split()) >= 5:
                            cleaned_list.append(cleaned_item)
                    elif isinstance(item, dict):
                        cleaned_item = {
                            k: self._clean_text(v) if isinstance(v, str) else v
                            for k, v in item.items()
                            if not (isinstance(v, str) and (not self._clean_text(v) or self._looks_like_filler(self._clean_text(v))))
                        }
                        if isinstance(cleaned_item.get("title"), str) and len(cleaned_item["title"].split()) < 2:
                            continue
                        cleaned_list.append(cleaned_item)
                    else:
                        cleaned_list.append(item)
                if cleaned_list:
                    cleaned[key] = cleaned_list
            else:
                cleaned[key] = value
        return cleaned

    def _clean_bridge_payload(self, bridges: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {"upward_path": [], "prerequisite_chains": []}
        for node in (bridges or {}).get("upward_path", []):
            if not isinstance(node, dict):
                continue
            title = self._clean_text(str(node.get("title", "")))
            if title and not self._looks_like_filler(title):
                cleaned["upward_path"].append({**node, "title": title})
        for prereq in (bridges or {}).get("prerequisite_chains", []):
            if not isinstance(prereq, dict):
                continue
            title = self._clean_text(str(prereq.get("title", "")))
            reason = self._clean_text(str(prereq.get("reason", "")))
            if (title and not self._looks_like_filler(title)) or (reason and not self._looks_like_filler(reason)):
                cleaned["prerequisite_chains"].append({**prereq, "title": title or prereq.get("title", ""), "reason": reason or prereq.get("reason", "")})
        return cleaned

    def _fallback_synthesis(
        self,
        question: str,
        local_anchors: Dict[str, Any],
        bridge_paths: Dict[str, Any],
        global_summary: Optional[str] = None,
        low_confidence: bool = False,
    ) -> str:
        question = self._clean_text(question)
        local_anchors = self._clean_anchor_payload(local_anchors)
        bridge_paths = self._clean_bridge_payload(bridge_paths)
        global_summary = self._clean_text(global_summary or "") or None

        local_facts = [self._clean_text(str(f)) for f in local_anchors.get("direct_facts", []) if self._clean_text(str(f))]
        bridge_titles = [self._clean_text(str(p.get("title", ""))) for p in bridge_paths.get("upward_path", []) if isinstance(p, dict)]
        prereqs = [self._clean_text(str(p.get("title", ""))) for p in bridge_paths.get("prerequisite_chains", []) if isinstance(p, dict)]
        concept_bridge = self._conceptualize_bridge(bridge_titles or prereqs)
        moments_text = self._extract_moments_text(local_anchors)

        if low_confidence or not local_facts:
            if global_summary:
                return f"The standard answer is that this topic fits within {global_summary}, and the precise definition depends on the specific concept you mean."
            return "The standard answer is the direct textbook definition and its usual application."

        first_fact = local_facts[0] if local_facts else "the core concept is built from the lecture's central definitions and examples"
        second_fact = local_facts[1] if len(local_facts) > 1 else None

        opening = f"The direct answer is that {first_fact[0].lower() + first_fact[1:] if first_fact else first_fact}."
        details = []
        if second_fact:
            details.append(second_fact)
        if moments_text:
            details.append(moments_text)
        elif global_summary:
            details.append(f"In this lecture context, the topic sits within {global_summary}.")
        if concept_bridge:
            details.append(f"This connects to {concept_bridge}.")

        body = " ".join(d for d in details if d).strip()
        answer = f"{opening} {body}".strip()
        answer = re.sub(r"\s+", " ", answer).strip()
        if not answer:
            return "The direct answer is that the concept follows the lecture's core definition and then extends to its standard applications."
        return answer

    def _extract_moments_text(self, local_anchors: Dict[str, Any]) -> str:
        direct_facts = local_anchors.get("direct_facts", [])
        out: List[str] = []
        for fact in direct_facts:
            cleaned = self._clean_text(str(fact))
            if cleaned and not self._looks_like_filler(cleaned) and len(cleaned.split()) >= 5:
                out.append(cleaned)
        return " ".join(out[:2]).strip()

    def _looks_like_filler(self, text: str) -> bool:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return True
        filler_tokens = {"um", "uh", "er", "ah", "like", "yeah", "so", "okay", "tricky", "early", "on"}
        return len([t for t in tokens if t not in filler_tokens]) <= 2

    def _conceptualize_bridge(self, titles: List[str]) -> str:
        joined = ", ".join(titles)
        joined = self._clean_text(joined)
        lowered = joined.lower()
        if not lowered:
            return "the lecture's prerequisite ideas and related concepts"

        concept_map = [
            (
                ["image", "kernel", "kernels", "filter", "filters", "vision", "cnn", "convolution"],
                "computer vision and convolutional neural networks, especially how filters (kernels) act on images",
            ),
            (
                ["linear", "regression", "classification", "logistic", "feature", "loss"],
                "foundational linear models and classification, especially how the model turns features into decisions",
            ),
            (
                ["probability", "bayes", "posterior", "prior"],
                "probabilistic reasoning and Bayesian modeling",
            ),
        ]
        for keywords, concept in concept_map:
            if any(k in lowered for k in keywords):
                return concept

        stripped = re.sub(r"(?i)^cluster[:\s-]*", "", joined).strip()
        if stripped:
            return f"the lecture's prerequisite ideas and related concepts, especially {stripped}"
        return "the lecture's prerequisite ideas and related concepts"


    def synthesize_with_evidence(
        self,
        question: str,
        ranked_nodes: List[Dict[str, Any]],
        evidence_spans: List[Dict[str, Any]],
        bridge_data: Optional[Dict[str, Any]] = None,
        global_summary: Optional[str] = None,
        low_confidence: bool = False,
    ) -> Dict[str, Any]:
        """High-level synthesis integrating nodes, spans, and bridges.

        Returns a structured response with answer, media coordinates, and path context.
        """
        # Extract local anchors from ranked nodes and spans
        local_anchors = {
            "top_node_id": ranked_nodes[0]["id"] if ranked_nodes else None,
            "top_node_title": ranked_nodes[0]["title"] if ranked_nodes else "",
            "direct_facts": [] if low_confidence else [n.get("summary", "") for n in ranked_nodes[:2]],
            "evidence": [] if low_confidence else evidence_spans[:4],
            "snippets": [] if low_confidence else [e.get("text", "") for e in evidence_spans[:4]],
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
            low_confidence=low_confidence,
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
