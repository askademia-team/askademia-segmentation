# graph_v1

Earlier single-lecture graph prototype. The active coarse-first pipeline is in `graph_v2/`.

## What This Code Does

Main components:

- `transition_builder.py`: current hierarchical transition-scored graph builder
- `builder.py`: earlier flat graph builder and shared graph utilities
- `render_temporal.py`: stacked temporal Graphviz renderer for hierarchical graph JSONs
- `tutor.py`: graph Q&A runtime with retrieval and traversal
- `retrieval.py`: node and span retrieval over the graph
- `validate.py`: graph validation checks
- `llm.py`: Azure/OpenAI calls for transitions, node summaries, edges, merges, embeddings, and answer synthesis
- `models.py`: graph dataclasses and shared types
- `run_e2e.py`: build graph plus save one QA trace
- `tests.py`: lightweight regression tests

## Data Location

Inputs:
- `graph_data/source_lectures/<course_id>/audio/...`
- `graph_data/source_lectures/<course_id>/video/...`

Outputs:
- `graph_v1/builds/<course_id>/<lecture_id>/...`

## Primary Builder

Use `transition_builder.py` for the current pipeline.

Example: build Lecture 6

```bash
python3 -m graph_v1.transition_builder \
  --audio "graph_data/source_lectures/data100_sp26/audio/Lecture 6 (Feb 5).json" \
  --video "graph_data/source_lectures/data100_sp26/video/Lecture 6 (Feb 5).json" \
  --output "graph_v1/builds/data100_sp26/lecture6/graph_hier.json" \
  --window-sec 75 \
  --overlap-sec 15 \
  --merge-every-n 1 \
  --min-segment-sec 25 \
  --max-segment-sec 180 \
  --major-threshold 0.70 \
  --minor-threshold 0.35 \
  --verbose
```

Expected outputs next to `graph_hier.json`:
- `graph_hier_merge_log.json`
- `graph_hier_weak_labels.json`
- `graph_hier_validation.json`

## Render Temporal Graph PNG

This recreates the stacked-column temporal view:
- coarse nodes across the top in time order
- fine nodes stacked directly under their coarse parent
- semantic edges overlaid without `part_of`

```bash
python3 -m graph_v1.render_temporal \
  --graph "graph_v1/builds/data100_sp26/lecture6/graph_hier.json"
```

Default outputs:
- `graph_v1/builds/data100_sp26/lecture6/visuals/graph_hier_temporal_final.dot`
- `graph_v1/builds/data100_sp26/lecture6/visuals/graph_hier_temporal_final.png`

Optional:
- `--output-prefix <path-without-extension>`
- `--hide-edge-labels`
- `--title-limit 28`

## Earlier Builder

`builder.py` builds a simpler non-hierarchical graph.

Example:

```bash
python3 -m graph_v1.builder \
  --audio "graph_data/source_lectures/data100_sp26/audio/Lecture 6 (Feb 5).json" \
  --video "graph_data/source_lectures/data100_sp26/video/Lecture 6 (Feb 5).json" \
  --output "graph_v1/builds/data100_sp26/lecture6/graph.json" \
  --window-sec 75 \
  --overlap-sec 15 \
  --merge-every-n 10
```

## Run Tutor Q&A

```bash
python3 -m graph_v1.tutor \
  --graph "graph_v1/builds/data100_sp26/lecture6/graph_hier.json" \
  --question "How does canonicalization help before merging datasets?" \
  --top-k 5 \
  --max-hops 4
```

Optional:
- `--feedback-log <path>`: append tutor run metadata as JSONL
- `--user-rating 0|1`: store a simple post-hoc quality label

## End-to-End Demo

```bash
python3 -m graph_v1.run_e2e \
  --audio "graph_data/source_lectures/data100_sp26/audio/Lecture 6 (Feb 5).json" \
  --video "graph_data/source_lectures/data100_sp26/video/Lecture 6 (Feb 5).json" \
  --output-dir "graph_v1/builds/data100_sp26/demo" \
  --question "What are the first topics covered in this lecture?"
```

Outputs:
- `graph.json`
- `graph_merge_log.json`
- `graph_validation.json`
- `qa_trace.json`

## Environment

`llm.py` loads environment variables from:
- [keys.env](/Users/meenakshimittal/askademia-segmentation/keys.env)

Expected variables are the repo-standard Azure/OpenAI ones, including:
- `azure_endpoint`
- `OPENAI_API_KEY`
- `api_version`

Optional deployment overrides:
- `GRAPH_V1_AZURE_CHAT_DEPLOYMENT`
- `GRAPH_V1_AZURE_EMBEDDING_DEPLOYMENT`

If model calls are unavailable, parts of the pipeline fall back to deterministic heuristics, but the main graph quality is intended for LLM-backed runs.
