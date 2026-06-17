from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional


ALLOWED_EDGE_TYPES = {"requires", "part_of", "related_to", "example_of", "references"}


@dataclass
class Span:
    span_id: str
    lecture_id: str
    timestamp: float
    text: str
    modality: Literal["audio", "video"]


@dataclass
class Window:
    window_id: str
    lecture_id: str
    start_ts: float
    end_ts: float
    span_ids: List[str]
    text: str


@dataclass
class GraphNode:
    id: str
    lecture_id: str
    title: str
    summary: str
    explanation: str
    source_span_ids: List[str]
    start_ts: float
    end_ts: float
    layer: Literal["coarse", "fine"] = "fine"
    parent_coarse_id: Optional[str] = None
    node_kind: Literal["lecture", "notes"] = "lecture"
    external_ref: Optional[str] = None
    embedding: Optional[List[float]] = None


@dataclass
class GraphEdge:
    id: str
    from_id: str
    to_id: str
    type: str
    reason: str
    evidence_span_ids: List[str]
    edge_confidence: float = 0.5
    evidence_count: int = 0


@dataclass
class Graph:
    lecture_id: str
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lecture_id": self.lecture_id,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "spans": [asdict(s) for s in self.spans],
            "metadata": self.metadata,
        }


@dataclass
class TransitionCandidate:
    timestamp: float
    shift_score: float
    shift_type: Literal["major", "minor", "none"]
    confidence: float
    rationale: str
    source_window_id: str


@dataclass
class SegmentationLayers:
    coarse_boundaries: List[float] = field(default_factory=list)
    fine_boundaries: List[float] = field(default_factory=list)
    calibration_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TutorSessionState:
    session_id: str
    recent_node_ids: List[str] = field(default_factory=list)
    concept_familiarity: Dict[str, str] = field(default_factory=dict)
    turn_count: int = 0


@dataclass
class TutorFunctionResult:
    function_name: str
    payload: Dict[str, Any]
    ok: bool = True
    error: Optional[str] = None


def validate_edge_type(edge_type: str) -> bool:
    return edge_type in ALLOWED_EDGE_TYPES
