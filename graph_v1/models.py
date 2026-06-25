from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional


ALLOWED_EDGE_TYPES = {"requires", "part_of", "related_to", "example_of", "references"}


def coerce_layer_value(layer: Any) -> int:
    if isinstance(layer, int):
        return layer
    if isinstance(layer, float) and layer.is_integer():
        return int(layer)
    if isinstance(layer, str):
        stripped = layer.strip().lower()
        if stripped.isdigit():
            return int(stripped)
        if stripped == "fine":
            return 0
        if stripped == "coarse":
            return 1
        if stripped.startswith("layer_"):
            suffix = stripped.split("_", 1)[1]
            if suffix.isdigit():
                return int(suffix)
    return 0


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
    layer: int = 0
    parent_node_id: Optional[str] = None
    parent_coarse_id: Optional[str] = None
    cluster_member_ids: List[str] = field(default_factory=list)
    node_kind: Literal["lecture", "notes"] = "lecture"
    external_ref: Optional[str] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self) -> None:
        self.layer = coerce_layer_value(self.layer)
        if self.parent_node_id is None and self.parent_coarse_id is not None:
            self.parent_node_id = self.parent_coarse_id
        if self.parent_coarse_id is None and self.parent_node_id is not None:
            self.parent_coarse_id = self.parent_node_id


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
class GraphHyperedge:
    id: str
    layer: int
    cluster_node_id: str
    member_node_ids: List[str]
    internal_edge_ids: List[str]
    title: str
    summary: str
    explanation: str
    source_span_ids: List[str]
    start_ts: float
    end_ts: float


@dataclass
class Graph:
    lecture_id: str
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    hyperedges: List[GraphHyperedge] = field(default_factory=list)
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lecture_id": self.lecture_id,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
            "hyperedges": [asdict(h) for h in self.hyperedges],
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
