# Graph V3 Advantages and Design Rationale

## 1) Problem Statement and Why V3 Exists

Existing segmentation approaches in this repo produce promising graphs, but they still show instability in boundary placement and structural consistency (for example, coverage gaps/overlaps in validation outputs). Graph V3 is designed to improve reliability while staying compatible with the current graph ecosystem.

The objective for V3 phase 1 is:
- produce cleaner hierarchical segmentation (coarse + fine)
- reduce structural validation failures caused by segmentation instability
- make segmentation decisions inspectable so contributors can learn from failures quickly

This phase intentionally prioritizes segmentation reliability and traceability over tutor/runtime changes.

---

## 2) Side-by-Side Comparison: V1 vs V2 vs V3

| Dimension | graph_v1 | graph_v2 | graph_v3 |
|---|---|---|---|
| Segmentation strategy | Windowed transition candidates; fine/coarse built in same flow (hier rebuilt later) | Coarse-first, then fine-within-coarse | Coarse-first, then fine-within-coarse with constrained selector + deterministic repair |
| Boundary signal source | Primarily LLM transition proposals (with score normalization) | Primarily LLM transition proposals per pass | Deterministic embedding-shift + lexical shift candidates, then lightweight LLM gating |
| Hierarchy construction | `part_of` rebuilt after merges | Structural `part_of` during fine pass + cleanup | Structural `part_of` only + explicit fine-within-coarse containment repair |
| Edge policy | Semantic edges from LLM; cleanup rules applied | Same-layer semantic edges, extra cross-coarse restrictions for fine nodes | Conservative same-layer semantic edges, stricter confidence/evidence filtering, `part_of` strictly structural |
| Repair/validation guarantees | Validation can still show fine gaps/overlaps | Improved hierarchy, but sample runs still show coverage errors | Deterministic post-selection repair targets no-gap/no-overlap inside containers |
| Explainability/debuggability | Merge logs + weak labels | Merge logs + weak labels (coarse/fine) | Rich trace artifact: window scores, gated decisions, selector metadata, repair actions, edge trace |

---

## 3) Weaknesses in V1/V2 That V3 Explicitly Targets

### A) Coverage gaps and overlaps
Observed weakness:
- sample validation reports in prior versions can include fine coverage gap/overlap errors.

V3 response:
- constrained boundary selector with duration constraints
- deterministic repair pass to enforce contiguous coverage in each container
- explicit validation checks for container-level coverage

### B) Boundary instability from LLM-only transitions
Observed weakness:
- transition quality can vary with prompt/output noise.

V3 response:
- first-stage deterministic candidate generation from embedding + lexical shifts
- LLM used as a lightweight verifier, not sole boundary source

### C) Hierarchy drift after segmentation/edge operations
Observed weakness:
- parent-child relations can become fragile when boundaries shift.

V3 response:
- strict `part_of` semantics
- deterministic parent containment and child realignment pass

### D) Semantic edge noise
Observed weakness:
- aggressive semantic edge proposals can add low-value traversal paths.

V3 response:
- conservative edge generation
- stricter confidence/evidence filtering
- no cross-layer semantic edges, no cross-coarse fine semantic edges

---

## 4) How to Read V3 Outputs (Learning Guide)

## 4.1 `graph_v3_trace.json`
Use this file first when diagnosing segmentation quality.

Key sections:
- `coarse.window_scores`: boundary score rows per window
- `coarse.raw_candidates`: local peak candidates before dedupe/gate
- `coarse.gated_candidates`: candidate decisions and rationale tags
- `coarse.selector`: constrained-selection strategy and metadata
- `coarse.repair_actions`: post-selection structural edits
- `fine.<coarse_id>.*`: same structure per coarse topic
- `edge_trace`: semantic edges emitted with confidence

## 4.2 `graph_v3_weak_labels.json`
Tracks accepted/rejected candidates and explicit reason codes. Useful for spotting repeated failure patterns (for example, candidates repeatedly rejected for being too close to segment ends).

## 4.3 `graph_v3_validation.json`
Use to verify structural goals:
- no fine gaps/overlaps inside coarse parents
- valid `part_of` relations
- no invalid semantic edge patterns

---

## 5) Diagnosing Common Failure Modes

### Too many tiny segments
Check:
- `candidate_peak_percentile`
- `signal_smooth_width`
- `duration_penalty`

Action:
- raise percentile, increase smoothing, increase duration penalty.

### Missing meaningful boundaries
Check:
- `window_scores` distribution and gated rejections
- whether gate confidence is too strict

Action:
- lower candidate peak percentile slightly, reduce gate strictness, review contexts around rejected candidates.

### Coarse nodes look too broad
Check:
- coarse target count and coarse min/max durations

Action:
- increase `coarse_target_nodes` or lower `coarse_max_segment_sec`.

### Fine segmentation too fragmented
Check:
- fine duration bounds and selected boundary count

Action:
- increase `fine_min_segment_sec`, increase smoothing, raise peak percentile.

---

## 6) Safe Tuning Order

When tuning, change one axis at a time:

1. `candidate_peak_percentile`
2. `signal_smooth_width`
3. `duration_penalty`
4. duration constraints (`fine_min_segment_sec`, `fine_max_segment_sec`)
5. target counts (`coarse_target_nodes`, `fine_target_nodes`)

Re-run and compare:
- validation error counts
- number of selected boundaries
- repair action count (high counts can indicate unstable raw segmentation)

---

## 7) Known Limitations (Phase 1)

- V3 still relies on summary generation quality for node text fields.
- This phase does not yet optimize tutor answer quality directly.
- Benchmark-driven evaluation is deferred; phase 1 focuses on reliable graph construction and interpretability.

---

## 8) Future Extensions

Likely next steps after phase 1:
- benchmark-based retrieval/answer evaluation versus v1/v2
- candidate calibration using question-conditioned signals
- stronger edge typing calibration and pruning
- optional cross-lecture concept linking once single-lecture reliability is stable
