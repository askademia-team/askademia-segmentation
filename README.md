# askademia-segmentation

This repo currently contains two separate systems:

- `graph_v2/`: the active coarse-first hierarchical lecture graph pipeline
- `graph_v1/`: the earlier single-lecture concept graph prototype
- `legacy_segmentation/`: older chunking, tree-building, retrieval, and experiment assets

## Active Pipeline

The active workflow is:

1. load one lecture's audio transcript and optional video OCR
2. detect topic transitions over overlapping windows
3. build fine and coarse graph nodes
4. add semantic edges and structural `part_of` edges
5. merge near-duplicate adjacent nodes
6. validate the graph
7. answer questions with graph retrieval and traversal

Code:
- [graph_v2/README.md](/Users/meenakshimittal/askademia-segmentation/graph_v2/README.md)
- [graph_v1/README.md](/Users/meenakshimittal/askademia-segmentation/graph_v1/README.md)

Data:
- [graph_data/source_lectures](/Users/meenakshimittal/askademia-segmentation/graph_data/source_lectures)
- [graph_v1/builds](/Users/meenakshimittal/askademia-segmentation/graph_v1/builds)
- [graph_v2/builds](/Users/meenakshimittal/askademia-segmentation/graph_v2/builds)

## Repo Layout

- [graph_v2](/Users/meenakshimittal/askademia-segmentation/graph_v2): active coarse-first graph builder and tutor
- [graph_v1](/Users/meenakshimittal/askademia-segmentation/graph_v1): earlier graph prototype and shared utilities
- [graph_data](/Users/meenakshimittal/askademia-segmentation/graph_data): current lecture source inputs
- [legacy_segmentation](/Users/meenakshimittal/askademia-segmentation/legacy_segmentation): old chunk/tree pipeline and legacy data
- [ta_responses](/Users/meenakshimittal/askademia-segmentation/ta_responses): labeled TA-response spreadsheet used by older evaluation utilities
- [keys.env](/Users/meenakshimittal/askademia-segmentation/keys.env): local Azure/OpenAI environment variables

## Current Source Data

Current lecture inputs are stored under:

- `graph_data/source_lectures/<course_id>/audio/`
- `graph_data/source_lectures/<course_id>/video/`

Example:
- [Lecture 6 audio](/Users/meenakshimittal/askademia-segmentation/graph_data/source_lectures/data100_sp26/audio/Lecture 6 (Feb 5).json)
- [Lecture 6 video](/Users/meenakshimittal/askademia-segmentation/graph_data/source_lectures/data100_sp26/video/Lecture 6 (Feb 5).json)

## Current Build Outputs

Generated graph artifacts are stored under:

- `graph_v1/builds/<course_id>/<lecture_id>/`
- `graph_v2/builds/<course_id>/<lecture_id>/`

Typical files:
- `graph.json` or `graph_hier.json`: graph nodes, edges, spans, metadata
- `graph_merge_log.json`: node merge audit
- `graph_validation.json`: graph validation report
- `graph_weak_labels.json`: accepted/rejected transition labels
- `qa_trace.json`: saved tutor run trace
- `visuals/`: rendered graph images and Graphviz source

## Quick Start

Build Lecture 6 with the current Graph V2 pipeline:

```bash
python3 -m graph_v2.builder \
  --audio "graph_data/source_lectures/data100_sp26/audio/Lecture 6 (Feb 5).json" \
  --video "graph_data/source_lectures/data100_sp26/video/Lecture 6 (Feb 5).json" \
  --output "graph_v2/builds/data100_sp26/lecture6/graph_v2.json" \
  --verbose
```

Run tutor QA on that graph:

```bash
python3 -m graph_v2.tutor \
  --graph "graph_v2/builds/data100_sp26/lecture6/graph_v2.json" \
  --question "How does canonicalization help before merging datasets?" \
  --top-k 5 \
  --max-hops 4
```

Render the stacked temporal graph image for any hierarchical graph JSON:

```bash
python3 -m graph_v1.render_temporal \
  --graph "graph_v2/builds/data100_sp26/lecture6/graph_v2.json"
```

## Legacy Code

Older chunking and tree-building code is documented here:

- [legacy_segmentation/README.md](/Users/meenakshimittal/askademia-segmentation/legacy_segmentation/README.md)
