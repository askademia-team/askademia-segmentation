from __future__ import annotations

import json
from typing import Any, Dict, List

from graph_v1.llm import LLMClient
from graph_v1.models import TransitionCandidate


class GraphV2LLMClient(LLMClient):
    def detect_coarse_transition_candidates(
        self,
        *,
        window_text: str,
        window_id: str,
        valid_timestamps: List[float],
        previous_transitions: List[float] | None = None,
        recent_nodes: List[Dict[str, Any]] | None = None,
        coarse_target_nodes: int = 12,
    ) -> List[TransitionCandidate]:
        if not valid_timestamps:
            return []
        if not self.available() or self.client is None:
            return self._fallback_transition_candidates(valid_timestamps, window_id, coarse=True)
        system = (
            "You are doing the coarse first pass of lecture segmentation. "
            "Find only MAJOR lecture-level topic shifts. "
            "Ignore examples, code snippets, short digressions, and local implementation details. "
            "Prefer very few boundaries and broad topic containers. "
            "Return strict JSON with key candidates. "
            "Each candidate must include timestamp, shift_score (0-1), shift_type (major|minor|none), confidence (0-1), rationale. "
            "Use only exact values from valid_timestamps."
        )
        user = {
            "window_id": window_id,
            "valid_timestamps": valid_timestamps,
            "previous_transitions": previous_transitions or [],
            "recent_coarse_nodes": recent_nodes or [],
            "window_text": window_text[:14000],
            "target_total_coarse_nodes": coarse_target_nodes,
            "constraints": [
                "return 0-2 candidates",
                "prefer 0 or 1 unless two major topic changes clearly occur",
                "do not force a boundary if the same broad topic continues",
                "use shift_type=major for true broad topic shifts only",
            ],
        }
        return self._parse_transition_candidates(system, user, valid_timestamps, window_id, coarse=True)

    def detect_fine_transition_candidates(
        self,
        *,
        window_text: str,
        window_id: str,
        valid_timestamps: List[float],
        coarse_node: Dict[str, Any],
        recent_nodes: List[Dict[str, Any]] | None = None,
        fine_target_nodes: int = 5,
    ) -> List[TransitionCandidate]:
        if not valid_timestamps:
            return []
        if not self.available() or self.client is None:
            return self._fallback_transition_candidates(valid_timestamps, window_id, coarse=False)
        system = (
            "You are doing the fine second pass of lecture segmentation inside one already-known coarse topic. "
            "Find meaningful subtopic shifts within that coarse topic. "
            "Do not create sentence-level fragments. "
            "Return strict JSON with key candidates. "
            "Each candidate must include timestamp, shift_score (0-1), shift_type (major|minor|none), confidence (0-1), rationale. "
            "Use only exact values from valid_timestamps."
        )
        user = {
            "window_id": window_id,
            "coarse_topic": coarse_node,
            "valid_timestamps": valid_timestamps,
            "recent_fine_nodes": recent_nodes or [],
            "window_text": window_text[:14000],
            "target_fine_nodes_within_coarse": fine_target_nodes,
            "constraints": [
                "return 0-2 candidates",
                "prefer meaningful subtopic shifts, not tiny transitions",
                "if the current subtopic continues, return no transition",
                "major is allowed only for unusually large subtopic changes inside the same coarse topic",
            ],
        }
        return self._parse_transition_candidates(system, user, valid_timestamps, window_id, coarse=False)

    def propose_same_layer_edges(
        self,
        *,
        new_node: Dict[str, Any],
        existing_nodes: List[Dict[str, Any]],
        valid_span_ids: List[str],
        layer: str,
        coarse_context: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        if not existing_nodes:
            return []
        if not self.available() or self.client is None:
            return [
                {
                    "to_existing": existing_nodes[-1]["id"],
                    "type": "related_to",
                    "reason": f"Heuristic adjacent {layer} context.",
                    "evidence_span_ids": valid_span_ids[:1],
                }
            ]
        system = (
            f"Given one new {layer} concept and existing {layer} nodes, propose 0-3 semantic edges. "
            "Use only: requires, related_to, example_of, references. "
            "Do NOT emit part_of. "
            "Only connect to nodes in the provided existing_nodes list. "
            "Use requires for concept -> prerequisite, example_of for example -> broader concept, related_to as fallback. "
            "Return strict JSON: {\"edges\":[{\"to_existing\":...,\"type\":...,\"reason\":...,\"evidence_span_ids\":[...]}]}."
        )
        user = {
            "new_node": new_node,
            "existing_nodes": existing_nodes[-30:],
            "valid_span_ids": valid_span_ids,
            "coarse_context": coarse_context,
        }
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
                if et not in {"requires", "related_to", "example_of", "references"} or not tgt:
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

    def _parse_transition_candidates(
        self,
        system: str,
        user: Dict[str, Any],
        valid_timestamps: List[float],
        window_id: str,
        *,
        coarse: bool,
    ) -> List[TransitionCandidate]:
        try:
            response = self.client.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user)}],
                temperature=0.1,
                max_tokens=700,
            )
            content = str((response.choices[0].message.content if response and response.choices else "") or "").strip()
            parsed = self._extract_json(content)
            if not isinstance(parsed, dict):
                return self._fallback_transition_candidates(valid_timestamps, window_id, coarse=coarse)
            valid = set(float(v) for v in valid_timestamps)
            out: List[TransitionCandidate] = []
            for c in parsed.get("candidates", []):
                ts = c.get("timestamp")
                if not isinstance(ts, (int, float)):
                    continue
                tsf = float(ts)
                if tsf not in valid:
                    continue
                shift_type = str(c.get("shift_type") or ("major" if coarse else "minor")).lower()
                if shift_type not in {"major", "minor", "none"}:
                    shift_type = "major" if coarse else "minor"
                out.append(
                    TransitionCandidate(
                        timestamp=tsf,
                        shift_score=max(0.0, min(1.0, float(c.get("shift_score", 0.0)))),
                        shift_type=shift_type,  # type: ignore[arg-type]
                        confidence=max(0.0, min(1.0, float(c.get("confidence", 0.0)))),
                        rationale=str(c.get("rationale") or "")[:300],
                        source_window_id=window_id,
                    )
                )
            dedup: Dict[float, TransitionCandidate] = {}
            for cand in out:
                prev = dedup.get(cand.timestamp)
                if prev is None or cand.shift_score > prev.shift_score:
                    dedup[cand.timestamp] = cand
            return sorted(dedup.values(), key=lambda c: (c.shift_score, c.timestamp), reverse=True)[:2]
        except Exception:
            return self._fallback_transition_candidates(valid_timestamps, window_id, coarse=coarse)

    def _fallback_transition_candidates(
        self,
        valid_timestamps: List[float],
        window_id: str,
        *,
        coarse: bool,
    ) -> List[TransitionCandidate]:
        if len(valid_timestamps) < 2:
            return []
        mid_idx = len(valid_timestamps) // 2
        ts = float(valid_timestamps[mid_idx])
        return [
            TransitionCandidate(
                timestamp=ts,
                shift_score=0.72 if coarse else 0.56,
                shift_type="major" if coarse else "minor",
                confidence=0.35,
                rationale="fallback_midpoint_candidate",
                source_window_id=window_id,
            )
        ]
