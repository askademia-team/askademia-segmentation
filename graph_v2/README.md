# graph_v2

Coarse-first hierarchical lecture graph pipeline.

## What This Code Does

Main components:
- `builder.py`: two-pass graph builder
- `llm.py`: coarse-pass and fine-pass LLM prompts plus same-layer edge proposal
- `tutor.py`: coarse-first tutor runtime
- `validate.py`: v2-specific structural validation
- `tests.py`: lightweight v2 regression tests

## Data Location

Inputs:
- `graph_data/source_lectures/<course_id>/audio/...`
- `graph_data/source_lectures/<course_id>/video/...`

Outputs:
- `graph_v2/builds/<course_id>/<lecture_id>/...`

## Build Lecture 6

```bash
python3 -m graph_v2.builder \
  --audio "graph_data/source_lectures/data100_sp26/audio/Lecture 6 (Feb 5).json" \
  --video "graph_data/source_lectures/data100_sp26/video/Lecture 6 (Feb 5).json" \
  --output "graph_v2/builds/data100_sp26/lecture6/graph_v2.json" \
  --coarse-window-sec 240 \
  --coarse-overlap-sec 60 \
  --fine-window-sec 90 \
  --fine-overlap-sec 20 \
  --merge-every-n 1 \
  --coarse-min-segment-sec 180 \
  --coarse-max-segment-sec 900 \
  --fine-min-segment-sec 25 \
  --fine-max-segment-sec 240 \
  --coarse-target-nodes 12 \
  --fine-target-nodes 5 \
  --verbose
```

Expected outputs:
- `graph_v2.json`
- `graph_v2_merge_log.json`
- `graph_v2_validation.json`
- `graph_v2_weak_labels.json`
- `visuals/graph_v2.dot`
- `visuals/graph_v2.png`
- `visuals/graph_v2_coarse.dot`
- `visuals/graph_v2_coarse.png`

## Render Temporal Graph PNG

For the stacked temporal view over any hierarchical graph JSON:

```bash
python3 -m graph_v1.render_temporal \
  --graph "graph_v2/builds/data100_sp26/lecture6/graph_v2.json"
```

This writes a Graphviz DOT and PNG under the graph's sibling `visuals/` directory.

## Run Tutor Q&A

```bash
python3 -m graph_v2.tutor \
  --graph "graph_v2/builds/data100_sp26/lecture6/graph_v2.json" \
  --question "How does canonicalization help before merging datasets?" \
  --top-k 5 \
  --max-hops 4
```
