from __future__ import annotations

import argparse
import json
from pathlib import Path

from graph_v1.render_temporal import default_output_prefix, render_temporal_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render stacked temporal PNG for a graph_v3 hierarchical graph JSON."
    )
    parser.add_argument("--graph", required=True, help="Path to graph_v3 JSON.")
    parser.add_argument(
        "--output-prefix",
        help="Output path prefix without extension. Defaults to <graph_dir>/visuals/<graph_stem>_temporal_final.",
    )
    parser.add_argument("--title-limit", type=int, default=28, help="Max title length per node label.")
    parser.add_argument(
        "--hide-edge-labels",
        action="store_true",
        help="Hide semantic edge labels while preserving legend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = Path(args.graph)
    output_prefix = Path(args.output_prefix) if args.output_prefix else default_output_prefix(graph_path)
    dot_path, png_path = render_temporal_graph(
        graph_path=graph_path,
        output_prefix=output_prefix,
        title_limit=args.title_limit,
        show_edge_labels=not args.hide_edge_labels,
    )
    print(json.dumps({"graph": str(graph_path), "dot": str(dot_path), "png": str(png_path)}, indent=2))


if __name__ == "__main__":
    main()

