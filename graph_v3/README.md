# graph_v3

Reliability-first hierarchical lecture graph pipeline.

`graph_v3` is an additive experiment focused on segmentation quality and explainability:
- deterministic embedding-shift boundary candidates
- lightweight LLM gate for keep/drop + shift type
- constrained boundary selection under duration + target-node constraints
- deterministic repair to remove gaps/overlaps and enforce hierarchy validity
- rich trace artifacts for debugging and learning

## What It Produces

Given one lecture audio transcript and optional video OCR transcript, it produces:
- `graph_v3.json`: graph nodes/edges/spans/metadata
- `graph_v3_validation.json`: structural validation report
- `graph_v3_weak_labels.json`: accepted/rejected boundary candidates with reasons
- `graph_v3_trace.json`: detailed segmentation traces (scores, gate decisions, selector metadata, repair edits)
- `visuals/graph_v3.dot` and `visuals/graph_v3.png`

Note: the builder enforces that output paths include a `graph_v3` directory segment.

## Build Example

```bash
python3 -m graph_v3.builder \
  --audio "graph_data/source_lectures/data100_sp26/audio/Lecture 6 (Feb 5).json" \
  --video "graph_data/source_lectures/data100_sp26/video/Lecture 6 (Feb 5).json" \
  --output "graph_v3/builds/data100_sp26/lecture6/graph_v3.json" \
  --verbose
```

For deterministic local-only operation (no remote LLM/API calls):

```bash
python3 -m graph_v3.builder \
  --audio "graph_data/source_lectures/data100_sp26/audio/Lecture 6 (Feb 5).json" \
  --video "graph_data/source_lectures/data100_sp26/video/Lecture 6 (Feb 5).json" \
  --output "graph_v3/builds/data100_sp26/lecture6/graph_v3.json" \
  --no-remote-llm
```

## Key CLI Flags

- `--coarse-window-sec`, `--coarse-overlap-sec`
- `--fine-window-sec`, `--fine-overlap-sec`
- `--coarse-min-segment-sec`, `--coarse-max-segment-sec`
- `--fine-min-segment-sec`, `--fine-max-segment-sec`
- `--coarse-target-nodes`, `--fine-target-nodes`
- `--signal-smooth-width`
- `--candidate-peak-percentile`
- `--llm-gate-threshold`
- `--duration-penalty`
- `--no-remote-llm`

## Environment Resolution (v3)

`graph_v3` reads `.env` at repo root and resolves service config with precedence:

- Endpoint: `AZURE_OPENAI_ENDPOINT` -> `azure_endpoint`
- API version: `AZURE_OPENAI_API_VERSION` -> `api_version`
- API key: `AZURE_OPENAI_API_KEY` -> `OPENAI_API_KEY`
- Chat deployment: `GRAPH_V3_AZURE_CHAT_DEPLOYMENT` -> `OPENAI_DEPLOYMENT` -> default
- Embedding deployment: `GRAPH_V3_AZURE_EMBEDDING_DEPLOYMENT` -> `OPENAI_EMBEDDING_DEPLOYMENT` -> default

Resolution sources and fallback flags are persisted in graph metadata under `metadata.env_resolution`.

## Tests

```bash
python3 graph_v3/tests.py
```

These tests cover signal extraction, constrained selection, repair behavior, env precedence resolution, and a Lecture 6 smoke test.

## Temporal PNG (stacked view)

Generate a stacked temporal visualization (same style as v1 temporal output):

```bash
python3 -m graph_v3.render_temporal \
  --graph "graph_v3/builds/data100_sp26/lecture6/graph_v3.json"
```
