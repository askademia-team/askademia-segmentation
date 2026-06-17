# graph_data

This folder stores source lecture inputs for the graph pipelines.

## Layout

- `source_lectures/`: lecture inputs consumed by `graph_v1`
- build artifacts now live under:
  - `graph_v1/builds/`
  - `graph_v2/builds/`

## source_lectures

Convention:

- `source_lectures/<course_id>/audio/...`
- `source_lectures/<course_id>/video/...`

These files are timestamped JSON inputs used during graph construction.

Current course:
- [data100_sp26](/Users/meenakshimittal/askademia-segmentation/graph_data/source_lectures/data100_sp26)

## Build Outputs

Versioned build locations:

- [graph_v1/builds](/Users/meenakshimittal/askademia-segmentation/graph_v1/builds)
- [graph_v2/builds](/Users/meenakshimittal/askademia-segmentation/graph_v2/builds)
