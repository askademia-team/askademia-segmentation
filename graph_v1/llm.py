from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AzureOpenAI

from .models import ALLOWED_EDGE_TYPES, TransitionCandidate


def _clean_conversational_filler(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"(?i)\b(um+|uh+|er+|ah+|like|you know|sort of|kind of|basically|so yeah|okay so|i think|maybe)\b", " ", text)
    text = re.sub(r"(?i)\b(tricky early on|hard to interpret|not sure|i guess)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    return text


_TRANSCRIPT_TITLE_PATTERNS = (
    r"\b(we|you|i|let'?s|lets|let us|okay|ok|right|so|now|here|there)\b",
    r"\b(going to|gonna|want to|need to|take a look|look at|talk about|see how|show you|think about|can see|we can|you can|let me|this is|that is)\b",
    r"\b(discussion this week|i think i mentioned|right we're going|ok guys|um|uh)\b",
)


def _title_looks_transcript_like(title: str) -> bool:
    title = _clean_conversational_filler(title)
    if not title:
        return True
    words = re.findall(r"[a-zA-Z0-9]+", title.lower())
    if len(words) < 2 or len(words) > 5:
        return True
    lowered = title.lower()
    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in _TRANSCRIPT_TITLE_PATTERNS):
        return True
    if lowered.endswith(("?", ":", ";", ",")):
        return True
    if words[0] in {"we", "you", "i", "let", "lets", "okay", "ok", "right", "so", "now", "here", "there"}:
        return True
    return False


class LLMClient:
    def __init__(self) -> None:
        # Follow repo standard env loading and Azure client construction.
        load_dotenv("./keys.env")
        self.azure_endpoint = os.getenv("azure_endpoint")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_version = os.getenv("api_version")
        self.chat_deployment = (
            os.getenv("GRAPH_V1_AZURE_CHAT_DEPLOYMENT")
            or os.getenv("OPENAI_DEPLOYMENT")
            or "gpt-4o-mini-2"
        )
        self.embedding_deployment = (
            os.getenv("GRAPH_V1_AZURE_EMBEDDING_DEPLOYMENT")
            or os.getenv("OPENAI_EMBEDDING_DEPLOYMENT")
            or "text-embedding-3-small"
        )
        self.client = None
        if self.available():
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )

    def available(self) -> bool:
        return bool(self.azure_endpoint and self.api_key and self.api_version)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self.available() or self.client is None:
            return [self._fallback_embedding(text) for text in texts]
        try:
            response = self.client.embeddings.create(
                input=[t[:8000] for t in texts],
                model=self.embedding_deployment,
            )
            return [list(d.embedding) for d in response.data]
        except Exception:
            return [self._fallback_embedding(text) for text in texts]

    def synthesize_tutor_answer(
        self,
        question: str,
        nodes: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        low_confidence: bool = False,
    ) -> str:
        if not nodes or low_confidence:
            return "The standard answer is the direct textbook definition and its usual application."
        if not self.available() or self.client is None:
            lead = nodes[0]
            details = " ".join(_clean_conversational_filler(ev.get("text", "")) for ev in evidence[:2]).strip()
            title = _clean_conversational_filler(lead.get("title", "Relevant concept"))
            summary = _clean_conversational_filler(lead.get("summary", ""))
            return f"{title}: {summary} {details}".strip()
        system = (
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
        facts = [
            str(item.get("summary", ""))
            for item in nodes[:4]
            if len(str(item.get("summary", "")).split()) > 5
        ] + [str(ev.get("text", "")) for ev in evidence[:4] if len(str(ev.get("text", "")).split()) > 5]
        if facts:
            user = f"STUDENT QUESTION: {question}\n\nLECTURE NOTES:\n- " + "\n- ".join(facts)
        else:
            user = f"STUDENT QUESTION: {question}\n\nCONTEXT: None. Provide a general technical definition."
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=350,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
            return content or f"{nodes[0].get('title', 'Relevant concept')}: {nodes[0].get('summary', '')}"
        except Exception:
            lead = nodes[0]
            details = " ".join(ev.get("text", "") for ev in evidence[:2]).strip()
            return f"{lead.get('title', 'Relevant concept')}: {lead.get('summary', '')} {details}".strip()

    def extract_nodes_and_edges(
        self,
        window_text: str,
        window_id: str,
        span_ids: List[str],
        existing_nodes: List[Dict[str, Any]],
        max_nodes: int = 3,
    ) -> Dict[str, Any]:
        if not self.available():
            return self._fallback_extract(window_text, window_id, span_ids, existing_nodes, max_nodes=max_nodes)

        system = (
            "You build a concept graph for a single lecture. "
            "Return strict JSON with keys nodes and edges. "
            "Nodes: 0-3 concise concepts with title/summary/explanation/source_span_ids. "
            "Do NOT force new nodes if the window mostly repeats prior concepts. "
            "Prefer reusing existing nodes when concepts already exist. "
            "Only create a new node when genuinely new, distinct concept content appears. "
            "Edges: only types requires, part_of, related_to, example_of, references. "
            "Use only span IDs provided."
        )
        user = {
            "window_id": window_id,
            "span_ids": span_ids,
            "window_text": window_text[:10000],
            "existing_nodes": existing_nodes[-30:],
            "max_nodes": max_nodes,
            "schema": {
                "notes": [
                    "nodes can be an empty list when no new concept is introduced",
                    "if concept already exists, prefer edges that reference existing node IDs instead of creating duplicate nodes",
                ],
                "nodes": [
                    {
                        "temp_id": "string",
                        "title": "string",
                        "summary": "string",
                        "explanation": "string",
                        "source_span_ids": ["string"],
                        "start_ts": 0.0,
                        "end_ts": 0.0,
                    }
                ],
                "edges": [
                    {
                        "from_temp_or_existing": "string",
                        "to_temp_or_existing": "string",
                        "type": "requires|part_of|related_to|example_of|references",
                        "reason": "string",
                        "evidence_span_ids": ["string"],
                    }
                ],
            },
        }

        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
        except Exception:
            return self._fallback_extract(window_text, window_id, span_ids, existing_nodes, max_nodes=max_nodes)

        parsed = self._extract_json(content)
        if not isinstance(parsed, dict):
            return self._fallback_extract(window_text, window_id, span_ids, existing_nodes, max_nodes=max_nodes)
        return self._sanitize(parsed, span_ids, max_nodes)

    def choose_merge_candidate(self, node_a: Dict[str, Any], node_b: Dict[str, Any]) -> bool:
        if not self.available():
            return node_a["title"].lower() == node_b["title"].lower()
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only YES or NO. YES if the two concept nodes refer to the same concept.",
                    },
                    {"role": "user", "content": json.dumps({"a": node_a, "b": node_b})},
                ],
                temperature=0,
                max_tokens=5,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "").upper()
            return "YES" in content
        except Exception:
            return node_a["title"].lower() == node_b["title"].lower()

    def detect_transitions(
        self,
        window_text: str,
        window_id: str,
        valid_timestamps: List[float],
        previous_transitions: List[float] | None = None,
        prior_segments: List[Dict[str, Any]] | None = None,
    ) -> List[float]:
        candidates = self.detect_transition_candidates(
            window_text=window_text,
            window_id=window_id,
            valid_timestamps=valid_timestamps,
            previous_transitions=previous_transitions,
            prior_segments=prior_segments,
        )
        ranked = sorted(candidates, key=lambda c: c.shift_score, reverse=True)
        out: List[float] = []
        for c in ranked:
            if c.shift_type == "none":
                continue
            out.append(c.timestamp)
            if len(out) >= 2:
                break
        return sorted(set(out))

    def detect_transition_candidates(
        self,
        window_text: str,
        window_id: str,
        valid_timestamps: List[float],
        previous_transitions: List[float] | None = None,
        prior_segments: List[Dict[str, Any]] | None = None,
    ) -> List[TransitionCandidate]:
        if not valid_timestamps or not self.available() or self.client is None:
            return []
        system = (
            "Detect concept transition candidates in this lecture window. "
            "Return strict JSON only with key candidates. "
            "Each candidate needs timestamp, shift_score (0-1), shift_type (major|minor|none), confidence (0-1), rationale. "
            "Use only exact values from valid_timestamps."
        )
        user = {
            "window_id": window_id,
            "valid_timestamps": valid_timestamps,
            "previous_transitions": previous_transitions or [],
            "prior_segments": prior_segments or [],
            "window_text": window_text[:12000],
            "constraints": [
                "0-4 candidates",
                "avoid duplicates from previous_transitions",
                "if content continues prior segment(s), return no transition",
            ],
        }
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user)}],
                temperature=0.1,
                max_tokens=600,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
            parsed = self._extract_json(content)
            if not isinstance(parsed, dict):
                return []
            valid = set(float(v) for v in valid_timestamps)
            out: List[TransitionCandidate] = []
            for c in parsed.get("candidates", []):
                ts = c.get("timestamp")
                if not isinstance(ts, (int, float)):
                    continue
                tsf = float(ts)
                if tsf not in valid:
                    continue
                shift_type = str(c.get("shift_type") or "none").lower()
                if shift_type not in {"major", "minor", "none"}:
                    shift_type = "none"
                score = float(c.get("shift_score", 0.0))
                conf = float(c.get("confidence", 0.0))
                out.append(
                    TransitionCandidate(
                        timestamp=tsf,
                        shift_score=max(0.0, min(1.0, score)),
                        shift_type=shift_type,  # type: ignore[arg-type]
                        confidence=max(0.0, min(1.0, conf)),
                        rationale=str(c.get("rationale") or "")[:300],
                        source_window_id=window_id,
                    )
                )
            dedup = {}
            for c in out:
                prev = dedup.get(c.timestamp)
                if prev is None or c.shift_score > prev.shift_score:
                    dedup[c.timestamp] = c
            out2 = list(dedup.values())
            if len(out2) <= 2:
                return sorted(out2, key=lambda x: x.timestamp)
            reduced = self._reduce_candidates_to_two(
                valid_timestamps=valid_timestamps,
                proposed_candidates=out2,
                window_text=window_text,
                prior_segments=prior_segments or [],
            )
            return sorted(reduced, key=lambda x: x.timestamp)
        except Exception:
            return []

    def _reduce_candidates_to_two(
        self,
        valid_timestamps: List[float],
        proposed_candidates: List[TransitionCandidate],
        window_text: str,
        prior_segments: List[Dict[str, Any]],
    ) -> List[TransitionCandidate]:
        """
        Keep transition limit at 2 using a second model decision, rather than naive truncation.
        """
        if not self.available() or self.client is None:
            return sorted(proposed_candidates, key=lambda c: c.shift_score, reverse=True)[:2]
        system = (
            "Select at most 2 best transition candidates from candidates. "
            "Return strict JSON: {\"candidates\":[{\"timestamp\":...,\"shift_score\":...,\"shift_type\":...,\"confidence\":...,\"rationale\":...}]} "
            "using only valid_timestamps."
        )
        user = {
            "valid_timestamps": valid_timestamps,
            "candidate_transitions": [c.__dict__ for c in proposed_candidates],
            "prior_segments": prior_segments,
            "window_text": window_text[:12000],
            "selection_rule": "choose the most semantically meaningful major topic boundaries, not minor shifts",
        }
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user)}],
                temperature=0.0,
                max_tokens=300,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
            parsed = self._extract_json(content)
            if not isinstance(parsed, dict):
                return sorted(proposed_candidates, key=lambda c: c.shift_score, reverse=True)[:2]
            valid = set(float(v) for v in valid_timestamps)
            out: List[TransitionCandidate] = []
            for c in parsed.get("candidates", []):
                ts = c.get("timestamp")
                if not isinstance(ts, (int, float)):
                    continue
                tsf = float(ts)
                if tsf not in valid:
                    continue
                shift_type = str(c.get("shift_type") or "none").lower()
                if shift_type not in {"major", "minor", "none"}:
                    shift_type = "none"
                out.append(
                    TransitionCandidate(
                        timestamp=tsf,
                        shift_score=max(0.0, min(1.0, float(c.get("shift_score", 0.0)))),
                        shift_type=shift_type,  # type: ignore[arg-type]
                        confidence=max(0.0, min(1.0, float(c.get("confidence", 0.0)))),
                        rationale=str(c.get("rationale") or "")[:300],
                        source_window_id="reduce",
                    )
                )
            if not out:
                return sorted(proposed_candidates, key=lambda c: c.shift_score, reverse=True)[:2]
            return sorted(out, key=lambda c: c.shift_score, reverse=True)[:2]
        except Exception:
            return sorted(proposed_candidates, key=lambda c: c.shift_score, reverse=True)[:2]

    def _segment_concept_title(self, segment_text: str) -> str:
        cleaned = _clean_conversational_filler(segment_text)
        tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", cleaned.lower()) if len(t) >= 3]
        stopwords = {
            "this", "that", "with", "from", "have", "were", "been", "into", "when", "what", "where",
            "there", "their", "about", "because", "after", "before", "would", "could", "should", "also",
            "lecture", "section", "topic", "today", "then", "just", "like", "okay", "yeah", "uh", "um",
            "we", "you", "i", "let", "lets", "lets", "right", "so", "now", "here", "there",
            "going", "gonna", "want", "need", "talk", "look", "show", "see", "think", "mention",
            "discuss", "cover", "explain", "introduce", "walk", "through", "build", "use",
        }
        technical = [t for t in tokens if t not in stopwords]
        if not technical:
            return "Core Concept"
        # Prefer a compact 2-4 word technical title, with simple phrase heuristics.
        phrase = " ".join(technical[:4]).strip()
        if len(phrase.split()) < 2:
            phrase = " ".join((technical[:2] * 2)[:2])
        title = " ".join(word.capitalize() for word in phrase.split()[:4])[:80]
        if _title_looks_transcript_like(title):
            return "Core Concept"
        return title

    def _segment_concept_summary(self, segment_text: str, title: str) -> str:
        cleaned = _clean_conversational_filler(segment_text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Fallback text should still read like a textbook definition rather than a transcript echo.
        if not cleaned:
            return f"This section explains {title.lower()} and why it matters in the lecture."
        words = cleaned.split()
        focal = " ".join(words[:24]).strip()
        return f"This section develops {title.lower()} by explaining its role in the broader lecture and its main technical implications."

    def summarize_segment_to_node(
        self,
        segment_text: str,
        source_span_ids: List[str],
        start_ts: float,
        end_ts: float,
    ) -> Dict[str, Any]:
        abstract_title = self._segment_concept_title(segment_text)
        abstract_summary = self._segment_concept_summary(segment_text, abstract_title)
        if not self.available() or self.client is None:
            return {
                "title": abstract_title,
                "summary": abstract_summary,
                "explanation": abstract_summary,
                "source_span_ids": source_span_ids,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }
        system = (
            "You are a technical knowledge graph compiler specializing in Computer Science and Machine Learning. "
            "You will receive a dense, pre-cleaned chunk of a lecture transcript text. Your task is to transform this raw content into a formal, structured Knowledge Graph node object.\n\n"
            "RAW TRANSCRIPT DATA:\n"
            "\"\"\"\n"
            f"{segment_text[:14000]}\n"
            "\"\"\"\n\n"
            "STRICT SYSTEM SCHEMA CONTRACTS:\n"
            "1. TITLE GENERATION: Extract and generate a 2-to-4 word precise, formal academic title (e.g., \"Matrix Transposition\", \"Attention Matrix Dimensions\", \"Multivariate Gaussian\"). \n"
            "   - CRITICAL NEGATIVE CONSTRAINT: You are strictly forbidden from writing titles using casual speech patterns, conversational grammar fragments, or phrasing found in the raw text (e.g., do NOT name a node \"But Information Get Table\", \"Alright Has One Child\", or \"So What Am I Saying\"). If no clear technical concept is found, name the node \"Thematic Lecture Overview\".\n\n"
            "2. SUMMARY SYNTHESIS: Synthesize an authoritative, grammatically immaculate textbook summary definition of the technical mechanics discussed. Completely filter out any narrative speaking patterns or first-person references (\"I think\", \"the professor talks about\"). Write with objective educational distance.\n\n"
            "OUTPUT SPECIFICATION:\n"
            "Return exclusively a valid JSON string matching this layout:\n"
            "{\n"
            "    \"title\": \"CONCISE TECHNICAL CONCEPT NAME\",\n"
            "    \"summary\": \"Formal academic textbook definition summary.\",\n"
            "    \"explanation\": \"Deep conceptual validation details.\"\n"
            "}"
        )
        user = {
            "start_ts": start_ts,
            "end_ts": end_ts,
        }
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user)}],
                temperature=0.0,
                max_tokens=1200,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
            parsed = self._extract_json(content)
            if not isinstance(parsed, dict):
                raise ValueError("invalid JSON")
            title = _clean_conversational_filler(str(parsed.get("title") or abstract_title)).strip()[:80]
            summary = _clean_conversational_filler(str(parsed.get("summary") or abstract_summary)).strip()
            explanation = _clean_conversational_filler(str(parsed.get("explanation") or abstract_summary)).strip()
            if len(title.split()) > 4 or len(title.split()) < 2 or _title_looks_transcript_like(title):
                title = abstract_title
            if not summary or len(summary.split()) < 6:
                summary = abstract_summary
            if not explanation or len(explanation.split()) < 8:
                explanation = abstract_summary
            return {
                "title": title,
                "summary": summary,
                "explanation": explanation,
                "source_span_ids": source_span_ids,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }
        except Exception:
            return {
                "title": abstract_title,
                "summary": abstract_summary,
                "explanation": abstract_summary,
                "source_span_ids": source_span_ids,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }

    def summarize_hyperedge_to_node(
        self,
        cluster_id: str,
        layer_name: str,
        member_nodes: List[Dict[str, Any]],
        internal_edges: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        if not self.available() or self.client is None:
            title = _clean_conversational_filler(f"Cluster {layer_name} summary")
            summary = " ; ".join(_clean_conversational_filler(n["title"]) for n in member_nodes[:3])
            explanation = "This cluster groups related concepts: " + summary
            return {"title": title, "summary": summary, "explanation": explanation}
        system = (
            "You are summarizing a semantic cluster of lecture concepts. "
            "Produce a concise cluster node title, a 1-sentence summary, and a short explanation. "
            "Use only the member nodes and internal edge connections provided."
        )
        payload = {
            "cluster_id": cluster_id,
            "layer": layer_name,
            "member_nodes": member_nodes,
            "internal_edges": internal_edges,
        }
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
            parsed = self._extract_json(content)
            if not isinstance(parsed, dict):
                raise ValueError("invalid JSON")
            return {
                "title": _clean_conversational_filler(str(parsed.get("title") or f"Cluster {layer_name}")).strip()[:120],
                "summary": _clean_conversational_filler(str(parsed.get("summary") or "")).strip(),
                "explanation": _clean_conversational_filler(str(parsed.get("explanation") or "")).strip(),
            }
        except Exception:
            title = _clean_conversational_filler(f"Cluster {layer_name} summary")
            summary = " ; ".join(_clean_conversational_filler(n["title"]) for n in member_nodes[:3])
            explanation = "This cluster groups related concepts: " + summary
            return {"title": title, "summary": summary, "explanation": explanation}

    def summarize_mst_component(
        self,
        query: str,
        component_id: str,
        node_rows: List[Dict[str, Any]],
        edge_rows: List[Dict[str, Any]],
    ) -> str:
        if not self.available() or self.client is None:
            titles = ", ".join(n.get("title", "") for n in node_rows[:3])
            return f"Core concepts: {titles}."
        system = (
            "You are creating a high-level summary of the most important concepts in a lecture graph component. "
            "Use only the provided MST nodes and edges to produce a concise broad explanation."
        )
        payload = {
            "query": query,
            "component_id": component_id,
            "nodes": node_rows,
            "edges": edge_rows,
        }
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                temperature=0.2,
                max_tokens=350,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
            return content
        except Exception:
            titles = ", ".join(n.get("title", "") for n in node_rows[:3])
            return f"Core concepts: {titles}."
        system = (
            "Summarize this lecture segment into one concept node. "
            "Return strict JSON with keys: title, summary, explanation. "
            "Title should be 3-8 words and must not include timestamps/brackets."
        )
        user = {"start_ts": start_ts, "end_ts": end_ts, "segment_text": segment_text[:14000]}
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user)}],
                temperature=0.2,
                max_tokens=1200,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
            parsed = self._extract_json(content)
            if not isinstance(parsed, dict):
                raise ValueError("invalid JSON")
            return {
                "title": str(parsed.get("title") or "Context Segment").strip()[:120],
                "summary": str(parsed.get("summary") or "").strip(),
                "explanation": str(parsed.get("explanation") or "").strip(),
                "source_span_ids": source_span_ids,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }
        except Exception:
            title = " ".join(segment_text.split()[:6])[:80] or "Context Segment"
            return {
                "title": title,
                "summary": segment_text[:280],
                "explanation": segment_text[:900],
                "source_span_ids": source_span_ids,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }

    def propose_edges_for_new_node(
        self,
        new_node: Dict[str, Any],
        existing_nodes: List[Dict[str, Any]],
        valid_span_ids: List[str],
    ) -> List[Dict[str, Any]]:
        if not existing_nodes:
            return []
        if not self.available() or self.client is None:
            return [
                {
                    "to_existing": existing_nodes[-1]["id"],
                    "type": "related_to",
                    "reason": "Heuristic adjacent lecture context.",
                    "evidence_span_ids": valid_span_ids[:1],
                }
            ]
        system = (
            "Given one new concept and existing nodes, propose 0-3 edges. "
            "Use only semantic types: requires, related_to, example_of, references. "
            "Do NOT emit part_of. part_of is assigned structurally elsewhere. "
            "Semantic edges must connect nodes at the same abstraction level. "
            "Use requires only when the new concept depends on understanding the target concept; "
            "in that case the edge direction is new_concept -> prerequisite. "
            "Use example_of only when the new concept is a concrete example/application of the target concept; "
            "in that case the edge direction is example -> broader concept. "
            "Use references only when the segment explicitly mentions or invokes the target concept. "
            "Use related_to only as a fallback when the connection is real but weaker than the other types. "
            "Return strict JSON: {\"edges\":[{\"to_existing\":...,\"type\":...,\"reason\":...,\"evidence_span_ids\":[...]}]}."
        )
        user = {"new_node": new_node, "existing_nodes": existing_nodes[-40:], "valid_span_ids": valid_span_ids}
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user)}],
                temperature=0.2,
                max_tokens=900,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
            parsed = self._extract_json(content)
            if not isinstance(parsed, dict):
                return []
            out = []
            valid = set(valid_span_ids)
            for e in parsed.get("edges", []):
                et = str(e.get("type") or "")
                tgt = str(e.get("to_existing") or "")
                if et == "part_of":
                    continue
                if et not in ALLOWED_EDGE_TYPES or not tgt:
                    continue
                ev = [s for s in e.get("evidence_span_ids", []) if s in valid]
                if not ev:
                    continue
                out.append(
                    {
                        "to_existing": tgt,
                        "type": et,
                        "reason": str(e.get("reason") or "").strip()[:300],
                        "evidence_span_ids": ev,
                    }
                )
            return out[:3]
        except Exception:
            return []

    def propose_adjacent_merge_groups(self, candidate_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return merge groups over adjacent candidate nodes.
        Output format: [{"node_ids": [...], "reason": "..."}]
        """
        if len(candidate_nodes) < 2:
            return []
        if not self.available() or self.client is None:
            return []
        system = (
            "You are performing merge planning for lecture concept nodes. "
            "Only propose groups of nodes that represent the same concept. "
            "Only merge temporally adjacent/near-adjacent nodes from the provided list. "
            "Handle topic-sandwich patterns: if a brief minor digression appears between two nodes of the same concept, "
            "you may merge all nodes together if concept continuity is strong; otherwise keep them separate. "
            "The shorter and less conceptually central the middle node is, the more likely a merge is appropriate. "
            "Return strict JSON: {\"merge_groups\":[{\"node_ids\":[...],\"reason\":\"...\"}]}. "
            "If no safe merges, return an empty list."
        )
        user = {
            "candidate_nodes": candidate_nodes,
            "rules": [
                "node_ids in each group must be from candidate_nodes only",
                "group size must be >=2",
                "prefer group size 2; use larger groups only when duplicate continuity is very clear",
                "groups may contain 2 or more nodes when justified",
                "do not merge different concepts even if related",
            ],
        }
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user)}],
                temperature=0.0,
                max_tokens=1000,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
            parsed = self._extract_json(content)
            if not isinstance(parsed, dict):
                return []
            out = []
            valid_ids = {str(n.get("id")) for n in candidate_nodes}
            for g in parsed.get("merge_groups", []):
                ids = [str(x) for x in g.get("node_ids", []) if str(x) in valid_ids]
                if len(ids) < 2:
                    continue
                out.append({"node_ids": ids, "reason": str(g.get("reason") or "").strip()[:300]})
            return out
        except Exception:
            return []

    def _extract_json(self, content: str) -> Dict[str, Any] | None:
        try:
            return json.loads(content)
        except Exception:
            pass
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except Exception:
                return None
        return None

    def _fallback_embedding(self, text: str) -> List[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        dims = 64
        vec = [0.0] * dims
        for tok in tokens:
            idx = hash(tok) % dims
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm <= 1e-9:
            return vec
        return [v / norm for v in vec]

    def _sanitize(self, payload: Dict[str, Any], span_ids: List[str], max_nodes: int) -> Dict[str, Any]:
        valid_span = set(span_ids)
        out_nodes = []
        for n in payload.get("nodes", [])[:max_nodes]:
            source_ids = [s for s in n.get("source_span_ids", []) if s in valid_span]
            if not source_ids:
                continue
            raw_title = str(n.get("title") or "Untitled").strip()[:120]
            title = raw_title
            if _title_looks_transcript_like(title):
                title = self._segment_concept_title(f"{raw_title} {n.get('summary') or ''}")
            if _title_looks_transcript_like(title):
                title = "Core Concept"
            out_nodes.append(
                {
                    "temp_id": str(n.get("temp_id") or ""),
                    "title": title,
                    "summary": str(n.get("summary") or "").strip(),
                    "explanation": str(n.get("explanation") or "").strip(),
                    "source_span_ids": source_ids,
                    "start_ts": float(n.get("start_ts", 0.0)),
                    "end_ts": float(n.get("end_ts", 0.0)),
                }
            )

        out_edges = []
        for e in payload.get("edges", []):
            edge_type = str(e.get("type") or "")
            if edge_type not in ALLOWED_EDGE_TYPES:
                continue
            ev = [s for s in e.get("evidence_span_ids", []) if s in valid_span]
            if not ev:
                continue
            out_edges.append(
                {
                    "from_temp_or_existing": str(e.get("from_temp_or_existing") or ""),
                    "to_temp_or_existing": str(e.get("to_temp_or_existing") or ""),
                    "type": edge_type,
                    "reason": str(e.get("reason") or "").strip()[:300],
                    "evidence_span_ids": ev,
                }
            )
        return {"nodes": out_nodes, "edges": out_edges}

    def _fallback_extract(
        self,
        window_text: str,
        window_id: str,
        span_ids: List[str],
        existing_nodes: List[Dict[str, Any]],
        max_nodes: int = 3,
    ) -> Dict[str, Any]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", window_text) if s.strip()]
        snippets = sentences[:max_nodes] or [window_text[:200]]
        nodes = []
        for i, snippet in enumerate(snippets):
            title = self._segment_concept_title(snippet)
            nodes.append(
                {
                    "temp_id": f"{window_id}_n{i+1}",
                    "title": title[:80],
                    "summary": snippet[:240],
                    "explanation": snippet[:700],
                    "source_span_ids": span_ids[: min(len(span_ids), 3)],
                    "start_ts": 0.0,
                    "end_ts": 0.0,
                }
            )

        edges = []
        if existing_nodes and nodes:
            edges.append(
                {
                    "from_temp_or_existing": nodes[0]["temp_id"],
                    "to_temp_or_existing": existing_nodes[-1]["id"],
                    "type": "related_to",
                    "reason": "Heuristic relation from adjacent lecture context.",
                    "evidence_span_ids": span_ids[:1],
                }
            )
        return {"nodes": nodes, "edges": edges}
