from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List


_FILLER_PATTERN = re.compile(
    r"(?ix)\b(?:"
    r"um+|uh+|er+|ah+|like|you\s+know|sort\s+of|kind\s+of|basically|so\s+yeah|okay\s+so|"
    r"i\s+think|maybe|alright|hey\s+everyone|discussion\s+this\s+week|hi\s+everyone|am\s+i\s+saying"
    r")\b"
)

_LOGISTICS_PATTERN = re.compile(
    r"(?ix)\b(?:homework\s+\d+|assignment\s+\d+|grades?|rubric|submission|turn\s+in|canvas|gradescope|syllabus|office\s+hours)\b"
)

_BROKEN_PUNCT_PATTERN = re.compile(r"(?:\s*,\s*,\s*|\.\.+|\.{3,})")
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass
class CleanedChunk:
    span_id: str
    start_ts: float
    end_ts: float
    text: str
    source_span_ids: List[str]
    modality: str


class LectureDataCleaner:
    def __init__(self) -> None:
        self.filler_pattern = _FILLER_PATTERN
        self.logistics_pattern = _LOGISTICS_PATTERN

    def clean_text(self, text: str) -> str:
        text = str(text or "")
        text = self.filler_pattern.sub(" ", text)
        text = self.logistics_pattern.sub(" ", text)
        text = re.sub(r"(?i)\b(?:um+|uh+|er+|ah+|like|you know|sort of|kind of|basically|so yeah|okay so|i think|maybe|alright|hey everyone|discussion this week|hi everyone|am i saying)\b", " ", text)
        text = re.sub(r"(?i)\b(?:homework\s+\d+|assignment\s+\d+|grades?|rubric|submission|turn in|canvas|gradescope|syllabus|office hours)\b", " ", text)
        text = _BROKEN_PUNCT_PATTERN.sub(" ", text)
        text = _WHITESPACE_PATTERN.sub(" ", text).strip(" ,.;:-")
        return text

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _looks_like_filler(self, text: str) -> bool:
        lowered = text.lower()
        if any(phrase in lowered for phrase in ("where's the library", "got got counting", "let's just", "okay so")):
            return True
        tokens = self._tokenize(text)
        if not tokens:
            return True
        filler_tokens = [t for t in tokens if t in {"um", "uh", "er", "ah", "like", "you", "know", "sort", "kind", "basically", "yeah", "okay", "alright", "maybe", "think"}]
        return len(filler_tokens) / max(1, len(tokens)) > 0.4

    def batch_process_spans(self, raw_spans: List[Dict[str, Any]], min_words: int = 60) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        buffer_texts: List[str] = []
        buffer_source_ids: List[str] = []
        buffer_start_ts: float | None = None
        buffer_end_ts: float | None = None
        buffer_modality = "audio"

        def reset() -> None:
            nonlocal buffer_texts, buffer_source_ids, buffer_start_ts, buffer_end_ts
            buffer_texts = []
            buffer_source_ids = []
            buffer_start_ts = None
            buffer_end_ts = None

        def flush(force: bool = False) -> None:
            nonlocal buffer_texts, buffer_source_ids, buffer_start_ts, buffer_end_ts, buffer_modality
            if not buffer_texts:
                return
            candidate_text = self.clean_text(" ".join(buffer_texts))
            token_count = len(self._tokenize(candidate_text))
            if not candidate_text:
                reset()
                return
            if token_count < min_words and not force:
                return
            if self._looks_like_filler(candidate_text):
                if processed:
                    processed[-1]["text"] = self.clean_text(f"{processed[-1]['text']} {candidate_text}")
                    processed[-1]["source_span_ids"] = sorted(set(processed[-1]["source_span_ids"] + buffer_source_ids))
                    processed[-1]["end_ts"] = max(float(processed[-1]["end_ts"]), float(buffer_end_ts or processed[-1]["end_ts"]))
                reset()
                return
            processed.append(
                {
                    "span_id": f"chunk_{len(processed) + 1}",
                    "start_ts": float(buffer_start_ts or 0.0),
                    "end_ts": float(buffer_end_ts or buffer_start_ts or 0.0),
                    "text": candidate_text,
                    "source_span_ids": list(dict.fromkeys(buffer_source_ids)),
                    "modality": buffer_modality,
                }
            )
            reset()

        for idx, raw in enumerate(raw_spans):
            cleaned = self.clean_text(str(raw.get("text", "")))
            if not cleaned:
                continue
            if buffer_start_ts is None:
                buffer_start_ts = float(raw.get("timestamp", 0.0))
                buffer_modality = str(raw.get("modality", "audio"))
            buffer_end_ts = float(raw.get("timestamp", buffer_start_ts))
            buffer_texts.append(cleaned)
            buffer_source_ids.append(str(raw.get("span_id", f"raw_{idx}")))
            if len(self._tokenize(" ".join(buffer_texts))) >= min_words:
                flush()

        flush(force=True)
        return processed
