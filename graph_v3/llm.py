from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:  # type: ignore[misc]
        return False

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None  # type: ignore[assignment]


def _extract_json(content: str) -> Dict[str, Any] | None:
    if not content:
        return None
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(content[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _lexical_overlap(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def resolve_env() -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("azure_endpoint")
    endpoint_source = "AZURE_OPENAI_ENDPOINT" if os.getenv("AZURE_OPENAI_ENDPOINT") else ("azure_endpoint" if os.getenv("azure_endpoint") else "missing")

    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("api_version")
    api_version_source = "AZURE_OPENAI_API_VERSION" if os.getenv("AZURE_OPENAI_API_VERSION") else ("api_version" if os.getenv("api_version") else "missing")

    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_key_source = "AZURE_OPENAI_API_KEY" if os.getenv("AZURE_OPENAI_API_KEY") else ("OPENAI_API_KEY" if os.getenv("OPENAI_API_KEY") else "missing")

    chat_deployment = (
        os.getenv("GRAPH_V3_AZURE_CHAT_DEPLOYMENT")
        or os.getenv("OPENAI_DEPLOYMENT")
        or "gpt-4o-mini"
    )
    if os.getenv("GRAPH_V3_AZURE_CHAT_DEPLOYMENT"):
        chat_deployment_source = "GRAPH_V3_AZURE_CHAT_DEPLOYMENT"
    elif os.getenv("OPENAI_DEPLOYMENT"):
        chat_deployment_source = "OPENAI_DEPLOYMENT"
    else:
        chat_deployment_source = "default:gpt-4o-mini"

    embedding_deployment = (
        os.getenv("GRAPH_V3_AZURE_EMBEDDING_DEPLOYMENT")
        or os.getenv("OPENAI_EMBEDDING_DEPLOYMENT")
        or "text-embedding-3-small"
    )
    if os.getenv("GRAPH_V3_AZURE_EMBEDDING_DEPLOYMENT"):
        embedding_deployment_source = "GRAPH_V3_AZURE_EMBEDDING_DEPLOYMENT"
    elif os.getenv("OPENAI_EMBEDDING_DEPLOYMENT"):
        embedding_deployment_source = "OPENAI_EMBEDDING_DEPLOYMENT"
    else:
        embedding_deployment_source = "default:text-embedding-3-small"

    return {
        "endpoint": endpoint,
        "api_version": api_version,
        "api_key": api_key,
        "endpoint_source": endpoint_source,
        "api_version_source": api_version_source,
        "api_key_source": api_key_source,
        "chat_deployment": chat_deployment,
        "chat_deployment_source": chat_deployment_source,
        "embedding_deployment": embedding_deployment,
        "embedding_deployment_source": embedding_deployment_source,
        "used_endpoint_fallback": endpoint_source != "AZURE_OPENAI_ENDPOINT",
        "used_api_version_fallback": api_version_source != "AZURE_OPENAI_API_VERSION",
        "used_api_key_fallback": api_key_source != "AZURE_OPENAI_API_KEY",
        "used_chat_deployment_fallback": chat_deployment_source != "GRAPH_V3_AZURE_CHAT_DEPLOYMENT",
        "used_embedding_deployment_fallback": embedding_deployment_source != "GRAPH_V3_AZURE_EMBEDDING_DEPLOYMENT",
    }


class GraphV3LLMClient:
    def __init__(self, *, prefer_remote: bool = True) -> None:
        self.prefer_remote = prefer_remote
        self.env = resolve_env()
        self.client = None
        if self.available():
            assert AzureOpenAI is not None
            self.client = AzureOpenAI(
                azure_endpoint=self.env["endpoint"],
                api_key=self.env["api_key"],
                api_version=self.env["api_version"],
            )

    def available(self) -> bool:
        return bool(
            self.prefer_remote
            and AzureOpenAI is not None
            and self.env["endpoint"]
            and self.env["api_key"]
            and self.env["api_version"]
        )

    def env_metadata(self) -> Dict[str, Any]:
        return {
            "endpoint_source": self.env["endpoint_source"],
            "api_version_source": self.env["api_version_source"],
            "api_key_source": self.env["api_key_source"],
            "chat_deployment_source": self.env["chat_deployment_source"],
            "embedding_deployment_source": self.env["embedding_deployment_source"],
            "used_endpoint_fallback": self.env["used_endpoint_fallback"],
            "used_api_version_fallback": self.env["used_api_version_fallback"],
            "used_api_key_fallback": self.env["used_api_key_fallback"],
            "used_chat_deployment_fallback": self.env["used_chat_deployment_fallback"],
            "used_embedding_deployment_fallback": self.env["used_embedding_deployment_fallback"],
            "remote_available": self.available(),
            "prefer_remote": self.prefer_remote,
        }

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self.available() or self.client is None:
            return [self._fallback_embedding(t) for t in texts]
        try:
            resp = self.client.embeddings.create(
                model=self.env["embedding_deployment"],
                input=[t[:8000] for t in texts],
            )
            return [list(row.embedding) for row in resp.data]
        except Exception:
            return [self._fallback_embedding(t) for t in texts]

    def gate_transition(
        self,
        *,
        phase: str,
        candidate_ts: float,
        candidate_score: float,
        left_text: str,
        right_text: str,
    ) -> Dict[str, Any]:
        heuristic = self._heuristic_gate(
            phase=phase,
            candidate_ts=candidate_ts,
            candidate_score=candidate_score,
            left_text=left_text,
            right_text=right_text,
        )
        if not self.available() or self.client is None:
            return {**heuristic, "source": "heuristic"}
        system = (
            "You validate candidate topic boundaries in lecture transcripts. "
            "Return strict JSON only: {\"keep\":bool,\"shift_type\":\"major|minor|none\",\"confidence\":0-1,\"rationale_tag\":\"...\"}. "
            "Be conservative and reject weak/noisy candidates."
        )
        user = {
            "phase": phase,
            "candidate_ts": candidate_ts,
            "candidate_score": candidate_score,
            "left_context": left_text[:2200],
            "right_context": right_text[:2200],
            "heuristic_suggestion": heuristic,
        }
        try:
            resp = self.client.chat.completions.create(
                model=self.env["chat_deployment"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)},
                ],
                temperature=0.0,
                max_tokens=180,
            )
            content = str((resp.choices[0].message.content if resp and resp.choices else "") or "").strip()
            parsed = _extract_json(content)
            if not isinstance(parsed, dict):
                return {**heuristic, "source": "heuristic_fallback_bad_json"}
            keep = bool(parsed.get("keep", heuristic["keep"]))
            shift_type = str(parsed.get("shift_type", heuristic["shift_type"])).lower()
            if shift_type not in {"major", "minor", "none"}:
                shift_type = heuristic["shift_type"]
            confidence = float(parsed.get("confidence", heuristic["confidence"]))
            confidence = max(0.0, min(1.0, confidence))
            rationale_tag = str(parsed.get("rationale_tag", "") or "").strip()[:120]
            return {
                "keep": keep and shift_type != "none",
                "shift_type": shift_type,
                "confidence": confidence,
                "rationale_tag": rationale_tag or "llm_gate",
                "source": "llm",
            }
        except Exception:
            return {**heuristic, "source": "heuristic_fallback_exception"}

    def summarize_segment_to_node(
        self,
        *,
        segment_text: str,
        source_span_ids: List[str],
        start_ts: float,
        end_ts: float,
    ) -> Dict[str, Any]:
        fallback = self._fallback_summary(
            segment_text=segment_text,
            source_span_ids=source_span_ids,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if not self.available() or self.client is None:
            return fallback
        system = (
            "Summarize this lecture segment as one concept node. "
            "Return strict JSON only: {\"title\":\"...\",\"summary\":\"...\",\"explanation\":\"...\"}. "
            "Title must be specific, 3-10 words, no timestamps."
        )
        user = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "segment_text": segment_text[:14000],
        }
        try:
            resp = self.client.chat.completions.create(
                model=self.env["chat_deployment"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            content = str((resp.choices[0].message.content if resp and resp.choices else "") or "").strip()
            parsed = _extract_json(content)
            if not isinstance(parsed, dict):
                return fallback
            title = str(parsed.get("title") or fallback["title"]).strip()[:120]
            summary = str(parsed.get("summary") or "").strip() or fallback["summary"]
            explanation = str(parsed.get("explanation") or "").strip() or fallback["explanation"]
            return {
                "title": title,
                "summary": summary,
                "explanation": explanation,
                "source_span_ids": source_span_ids,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }
        except Exception:
            return fallback

    def propose_same_layer_edges(
        self,
        *,
        phase: str,
        new_node: Dict[str, Any],
        existing_nodes: List[Dict[str, Any]],
        valid_span_ids: List[str],
    ) -> List[Dict[str, Any]]:
        if not existing_nodes:
            return []
        heuristic = self._heuristic_edges(
            new_node=new_node,
            existing_nodes=existing_nodes,
            valid_span_ids=valid_span_ids,
        )
        if not self.available() or self.client is None:
            return heuristic
        system = (
            f"You are proposing conservative semantic edges between {phase} nodes in a lecture graph. "
            "Return strict JSON only: {\"edges\":[{\"to_existing\":\"...\",\"type\":\"requires|related_to|example_of|references\",\"reason\":\"...\",\"confidence\":0-1,\"evidence_span_ids\":[...] }]} "
            "Use at most 2 edges, high precision over recall."
        )
        user = {
            "new_node": new_node,
            "existing_nodes": existing_nodes[-20:],
            "valid_span_ids": valid_span_ids,
        }
        try:
            resp = self.client.chat.completions.create(
                model=self.env["chat_deployment"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            content = str((resp.choices[0].message.content if resp and resp.choices else "") or "").strip()
            parsed = _extract_json(content)
            if not isinstance(parsed, dict):
                return heuristic
            out: List[Dict[str, Any]] = []
            valid = set(valid_span_ids)
            allowed = {"requires", "related_to", "example_of", "references"}
            for row in parsed.get("edges", []):
                if not isinstance(row, dict):
                    continue
                et = str(row.get("type") or "").strip()
                tgt = str(row.get("to_existing") or "").strip()
                if et not in allowed or not tgt:
                    continue
                conf = max(0.0, min(1.0, float(row.get("confidence", 0.0))))
                if conf < 0.65:
                    continue
                evidence = [sid for sid in row.get("evidence_span_ids", []) if sid in valid]
                if not evidence:
                    continue
                out.append(
                    {
                        "to_existing": tgt,
                        "type": et,
                        "reason": str(row.get("reason") or "llm_edge").strip()[:220],
                        "confidence": conf,
                        "evidence_span_ids": evidence[:3],
                    }
                )
            return out[:2] if out else heuristic
        except Exception:
            return heuristic

    def _heuristic_gate(
        self,
        *,
        phase: str,
        candidate_ts: float,
        candidate_score: float,
        left_text: str,
        right_text: str,
    ) -> Dict[str, Any]:
        overlap = _lexical_overlap(left_text, right_text)
        lexical_change = 1.0 - overlap
        score = 0.7 * max(0.0, min(1.0, candidate_score)) + 0.3 * lexical_change
        if phase == "coarse":
            keep = score >= 0.58
            shift_type = "major" if score >= 0.72 else ("minor" if keep else "none")
        else:
            keep = score >= 0.52
            shift_type = "major" if score >= 0.75 else ("minor" if keep else "none")
        return {
            "keep": keep and shift_type != "none",
            "shift_type": shift_type,
            "confidence": score,
            "rationale_tag": f"heuristic_overlap_{overlap:.2f}@{candidate_ts:.1f}s",
        }

    def _heuristic_edges(
        self,
        *,
        new_node: Dict[str, Any],
        existing_nodes: List[Dict[str, Any]],
        valid_span_ids: List[str],
    ) -> List[Dict[str, Any]]:
        text_new = f"{new_node.get('title', '')} {new_node.get('summary', '')} {new_node.get('explanation', '')}"
        best: tuple[float, Dict[str, Any] | None] = (0.0, None)
        for cand in existing_nodes[-12:]:
            text_old = f"{cand.get('title', '')} {cand.get('summary', '')} {cand.get('explanation', '')}"
            sim = _lexical_overlap(text_new, text_old)
            if sim > best[0]:
                best = (sim, cand)
        if best[1] is None or best[0] < 0.28 or not valid_span_ids:
            return []
        return [
            {
                "to_existing": str(best[1].get("id")),
                "type": "related_to",
                "reason": f"heuristic_lexical_similarity={best[0]:.2f}",
                "confidence": max(0.65, min(0.9, best[0] + 0.35)),
                "evidence_span_ids": valid_span_ids[:2],
            }
        ]

    def _fallback_summary(
        self,
        *,
        segment_text: str,
        source_span_ids: List[str],
        start_ts: float,
        end_ts: float,
    ) -> Dict[str, Any]:
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", segment_text)
        title = " ".join(words[:7]).strip() or "Lecture Segment"
        summary = " ".join(segment_text.split())[:280]
        explanation = " ".join(segment_text.split())[:900]
        return {
            "title": title[:120],
            "summary": summary,
            "explanation": explanation,
            "source_span_ids": source_span_ids,
            "start_ts": start_ts,
            "end_ts": end_ts,
        }

    def _fallback_embedding(self, text: str) -> List[float]:
        dims = 96
        vec = [0.0] * dims
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % dims
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm <= 1e-12:
            return vec
        return [v / norm for v in vec]
