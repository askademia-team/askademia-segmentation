# Advanced Tutor Features: Contextual Bridges, Timestamp Pooling, and Humanized Responses

## Overview

Three major enhancements have been added to the knowledge graph tutor system to improve answer quality and student experience:

1. **Contextual Bridge Extraction** - Dijkstra-based path finding showing why a concept matters
2. **Timestamp Pooling** - Continuous rolling window density scoring for precise media coordinates  
3. **ResponseSynthesizer** - Humanized answer generation using LLM with conversational strategies

## 1. Contextual Bridge Extraction

**Location:** `graph_v1/retrieval.py::extract_contextual_bridge()`

**Purpose:** Extract "contextual bridges" showing why a concept matters and what prerequisites are needed.

**Algorithm:**
- Uses Dijkstra's algorithm on a weighted dependency graph (part_of and requires edges)
- Finds shortest path from target node upward to root parents
- Identifies prerequisite chains (incoming 'requires' edges)
- Returns structured bridge data with hierarchical context

**Usage:**
```python
from graph_v1.retrieval import extract_contextual_bridge
from graph_v1.tutor import load_graph

graph = load_graph(Path("graph.json"))
bridges = extract_contextual_bridge(graph, node_id="node_42")

# Result structure:
# {
#   "node_id": "node_42",
#   "upward_path": [
#     {"id": "node_42", "title": "Regex patterns", "layer": "fine"},
#     {"id": "node_99", "title": "Text processing", "layer": "coarse"}
#   ],
#   "prerequisite_chains": [
#     {
#       "prerequisite_id": "node_10",
#       "title": "String basics",
#       "reason": "regex operates on strings",
#       "confidence": 0.95
#     }
#   ]
# }
```

**Benefit:** Tutors and frontend apps can show "why this matters" by displaying upward context and prerequisites before diving into the concept.

---

## 2. Timestamp Pooling with Density Scoring

**Location:** `graph_v1/retrieval.py::pool_and_score_spans()`

**Purpose:** Find the most semantically relevant spans for a node and return precise media coordinates.

**Algorithm:**
- Groups spans associated with a node
- Applies rolling window of configurable size (default: 3 spans)
- Computes combined density score for each window: `0.6 * semantic + 0.4 * lexical`
- Semantic score: embedding-based similarity to query
- Lexical score: term overlap with query
- Returns top spans sorted by density with explicit start_ts/end_ts

**Usage:**
```python
from graph_v1.retrieval import pool_and_score_spans
from graph_v1.llm import LLMClient

graph = load_graph(Path("graph.json"))
llm = LLMClient()
result = pool_and_score_spans(graph, "node_42", "How does regex work?", llm, window_size=3)

# Result structure:
# {
#   "node_id": "node_42",
#   "start_ts": 245.0,           # Exact start for media playback
#   "end_ts": 290.0,             # Exact end for media playback
#   "pooled_spans": [
#     {
#       "window_start_idx": 0,
#       "window_end_idx": 3,
#       "center_span": {...},    # Most relevant span in window
#       "density_score": 0.87,
#       "window_timestamp": 250.0
#     },
#     ...
#   ],
#   "total_spans": 12
# }
```

**Benefit:** Frontend can directly seek audio/video to `start_ts`-`end_ts` without guessing. The density scoring ensures we highlight the most relevant moment in the node's evidence.

---

## 3. ResponseSynthesizer: Humanized Answer Generation

**Location:** `graph_v1/response_synthesizer.py`

**Purpose:** Synthesize conversational tutor answers that combine local facts, contextual bridges, and global summaries.

**Strategies:**
- **Socratic**: Guide students with questions, encouraging discovery
- **Scaffolded** (default): Break down concept into steps, provide prerequisite context, connect to lecture theme  
- **Direct**: Provide immediate, well-supported answer

**Usage:**
```python
from graph_v1.response_synthesizer import ResponseSynthesizer
from graph_v1.llm import LLMClient

llm = LLMClient()
synth = ResponseSynthesizer(llm)

local_anchors = {
    "top_node_id": "node_42",
    "top_node_title": "Regex patterns",
    "direct_facts": ["Regex uses metacharacters like . and *"],
    "start_ts": 245.0,
    "end_ts": 290.0,
}

bridge_paths = {
    "upward_path": [
        {"id": "node_42", "title": "Regex patterns", "layer": "fine"},
        {"id": "node_99", "title": "Text processing", "layer": "coarse"}
    ],
    "prerequisite_chains": [
        {"prerequisite_id": "node_10", "title": "String basics", "reason": "...", "confidence": 0.95}
    ]
}

answer = synth.synthesize(
    question="How do regex patterns work?",
    local_anchors=local_anchors,
    bridge_paths=bridge_paths,
    global_summary="Advanced text processing techniques",
    strategy="scaffolded"
)
# Returns: "To understand this, you should know: String basics. Key facts: Regex uses metacharacters like . and *. [See video/audio: 245.0s - 290.0s for details]"
```

**High-Level Integration:**
```python
from graph_v1.tutor import TutorRuntime

rt = TutorRuntime(graph)

# Get bridge context
bridge_result = rt.get_contextual_bridge("node_42")

# Get media coordinates
media_result = rt.get_media_coordinates("node_42", "How does regex work?")

# Synthesize humanized answer
ranked_nodes = [...]  # from retrieval
evidence_spans = [...]  # from retrieval
result = rt.synthesize_humanized_answer(
    question="How do regex patterns work?",
    ranked_nodes=ranked_nodes,
    evidence_spans=evidence_spans,
    include_bridges=True
)

# Result contains:
# {
#   "answer": "Humanized conversational answer...",
#   "media_coordinates": {"start_ts": 245.0, "end_ts": 290.0, "primary_modality": "audio"},
#   "context_bridges": {
#     "upward_path": [...],
#     "prerequisites": [...]
#   },
#   "cited_nodes": ["node_42", "node_99", ...],
#   "cited_spans": ["a_123", "a_124", ...]
# }
```

**Fallback Mode:** When LLM is unavailable, ResponseSynthesizer builds a structured answer from bridge paths and facts deterministically.

---

## Integration Points

### TutorRuntime New Methods
- `get_contextual_bridge(node_id)` → Extracts bridge structure
- `get_media_coordinates(node_id, query)` → Computes timestamp pooling
- `synthesize_humanized_answer(question, ranked_nodes, evidence_spans, include_bridges)` → Generates conversational answer

### Complete Flow Example
```python
from graph_v1.tutor import load_graph, TutorRuntime, TutorSessionState

graph = load_graph(Path("graph.json"))
rt = TutorRuntime(graph)

question = "How do regex patterns work?"
search_result = rt.search_nodes(question, top_k=5)
ranked = search_result.payload["results"][:3]

# Get evidence spans
evidence = rt.get_evidence(ranked[0]["id"], question).payload.get("evidence", [])

# Synthesize with bridges and media coordinates
answer_result = rt.synthesize_humanized_answer(
    question,
    ranked,
    evidence,
    include_bridges=True
)

print(answer_result.payload["answer"])
print(f"Watch at: {answer_result.payload['media_coordinates']}")
```

---

## Testing

New regression tests validate:
- `test_contextual_bridge_extraction()` - Dijkstra upward path finding
- `test_timestamp_pooling_with_density()` - Rolling window scoring
- `test_response_synthesizer_fallback()` - Fallback synthesis without LLM
- `test_tutor_synthesize_humanized_answer()` - Integration with tutor runtime

Run tests:
```bash
python3 -m pytest graph_v1/tests.py::test_contextual_bridge_extraction -v
python3 -m pytest graph_v1/tests.py::test_timestamp_pooling_with_density -v
python3 -m pytest graph_v1/tests.py::test_response_synthesizer_fallback -v
python3 -m pytest graph_v1/tests.py::test_tutor_synthesize_humanized_answer -v
```

All 13 tests pass (9 existing + 4 new).

---

## Performance and Limitations

- **Bridge Extraction:** O(V + E log V) Dijkstra's on dependency graph. Fast for typical graphs (100-300 nodes)
- **Timestamp Pooling:** O(S * D) where S = spans, D = embedding dimension. Efficient with caching
- **Synthesis:** Calls LLM (if available) with ~500 token budget. Falls back to deterministic generation when LLM unavailable
- **Media Accuracy:** Timestamps are based on pooled span evidence; frontend should handle edge cases (e.g., if spans are sparse)

---

## Frontend Integration Hints

1. **Contextual Bridges** → Display as breadcrumb (upward_path) and "Prerequisite" sidebar (prerequisite_chains)
2. **Media Coordinates** → Pass start_ts/end_ts to audio/video player with `seek()` method
3. **Humanized Answers** → Display as main answer text; optionally show Socratic follow-ups as chat bubbles
4. **Modality Selection** → Use `primary_modality` to prioritize audio or video player display
