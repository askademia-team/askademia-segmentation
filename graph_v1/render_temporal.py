from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .models import Graph, GraphEdge, GraphNode, Span


EDGE_COLORS = {
    "related_to": "#94a3b8",
    "requires": "#d97706",
    "example_of": "#059669",
    "references": "#7c3aed",
}


def load_graph(graph_path: Path) -> Graph:
    raw = json.loads(graph_path.read_text())
    nodes = [GraphNode(**node) for node in raw.get("nodes", [])]
    edges = [GraphEdge(**edge) for edge in raw.get("edges", [])]
    spans = [Span(**span) for span in raw.get("spans", [])]
    return Graph(
        lecture_id=raw["lecture_id"],
        nodes=nodes,
        edges=edges,
        spans=spans,
        metadata=raw.get("metadata", {}),
    )


def short_title(text: str, limit: int = 28) -> str:
    safe = (text or "Untitled").replace('"', "'")
    return safe if len(safe) <= limit else safe[: limit - 1] + "…"


def node_label(node: GraphNode, title_limit: int = 28) -> str:
    return f"{short_title(node.title, limit=title_limit)}\\n[{node.start_ts:.0f}-{node.end_ts:.0f}s]"


def order_hierarchical_nodes(graph: Graph) -> Tuple[Dict[int, List[GraphNode]], Dict[str, List[GraphNode]]]:
    layer_nodes: Dict[int, List[GraphNode]] = defaultdict(list)
    for node in graph.nodes:
        layer_nodes[int(node.layer)].append(node)
    if not layer_nodes:
        raise ValueError("Temporal renderer requires a hierarchical graph.")

    for nodes in layer_nodes.values():
        nodes.sort(key=lambda node: (node.start_ts, node.end_ts, node.id))

    children_by_parent: Dict[str, List[GraphNode]] = defaultdict(list)
    for node in graph.nodes:
        parent_id = node.parent_node_id or node.parent_coarse_id
        if parent_id:
            children_by_parent[parent_id].append(node)

    for nodes in children_by_parent.values():
        nodes.sort(key=lambda node: (node.start_ts, node.end_ts, node.id))

    return layer_nodes, children_by_parent


def build_stacked_layout_dot(graph: Graph, title_limit: int = 28) -> str:
    layer_nodes, children = order_hierarchical_nodes(graph)
    layer_ids = sorted(layer_nodes.keys(), reverse=True)
    lines: List[str] = []
    lines.append("digraph G {")
    lines.append("  rankdir=TB;")
    lines.append('  graph [bgcolor="white", splines=ortho, overlap=false, pad=0.35, nodesep=0.55, ranksep=0.85];')
    lines.append('  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.10,0.08"];')
    lines.append('  edge [arrowhead=none, color="#cbd5e1", penwidth=1.1];')
    palette = ["#dbeafe", "#ecfccb", "#fef3c7", "#fae8ff", "#ffe4e6", "#e0f2fe", "#dcfce7", "#ede9fe"]
    border_palette = ["#2563eb", "#65a30d", "#d97706", "#a855f7", "#ef4444", "#0891b2", "#16a34a", "#7c3aed"]

    for layer in layer_ids:
        nodes = layer_nodes[layer]
        lines.append("  { rank=same;")
        for idx, node in enumerate(nodes):
            fill = palette[layer % len(palette)]
            color = border_palette[layer % len(border_palette)]
            penwidth = "1.8" if layer > 0 else "1.0"
            lines.append(
                f'    "{node.id}" [group="g{idx}", label="{node_label(node, title_limit=title_limit)}", '
                f'fillcolor="{fill}", color="{color}", penwidth={penwidth}];'
            )
        lines.append("  }")

    for parent_id, child_list in children.items():
        if not child_list:
            continue
        lines.append(f'  "{parent_id}" -> "{child_list[0].id}" [weight=100];')
        for left, right in zip(child_list, child_list[1:]):
            lines.append(f'  "{left.id}" -> "{right.id}" [weight=100];')

    lines.append("}")
    return "\n".join(lines) + "\n"


def parse_plain_layout(plain_text: str) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    graph_width = 0.0
    graph_height = 0.0
    positions: Dict[str, Dict[str, float]] = {}
    for line in plain_text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "graph":
            graph_width = float(parts[2])
            graph_height = float(parts[3])
        elif parts[0] == "node":
            _, node_id, x, y, width, height = parts[:6]
            positions[node_id] = {
                "x": float(x) * 72.0,
                "y": float(y) * 72.0,
                "w": float(width),
                "h": float(height),
            }
    if not positions:
        raise ValueError("Could not parse node positions from Graphviz plain output.")
    return graph_width * 72.0, graph_height * 72.0, positions


def build_fixed_overlay_dot(graph: Graph, title_limit: int = 28, show_edge_labels: bool = True) -> str:
    layer_nodes, children = order_hierarchical_nodes(graph)
    stacked_dot = build_stacked_layout_dot(graph, title_limit=title_limit)
    with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as handle:
        handle.write(stacked_dot)
        temp_dot_path = Path(handle.name)

    try:
        plain_text = subprocess.check_output(["dot", "-Tplain", str(temp_dot_path)], text=True)
    finally:
        temp_dot_path.unlink(missing_ok=True)

    graph_width, graph_height, positions = parse_plain_layout(plain_text)
    canvas_width = graph_width + 240.0

    lines: List[str] = []
    lines.append("digraph G {")
    lines.append("  layout=neato;")
    lines.append(
        f'  graph [bb="0,0,{canvas_width:.2f},{graph_height:.2f}", bgcolor="white", overlap=false, splines=true, '
        'notranslate=true, outputorder=edgesfirst, pad=0.2];'
    )
    lines.append('  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.10,0.08"];')
    lines.append('  edge [fontname="Helvetica", fontsize=8];')

    for node in graph.nodes:
        pos = positions.get(node.id)
        if not pos:
            continue
        fill = "#dbeafe" if node.layer > 0 else "#ecfccb"
        color = "#2563eb" if node.layer > 0 else "#65a30d"
        penwidth = "1.8" if node.layer > 0 else "1.0"
        lines.append(
            f'  "{node.id}" [label="{node_label(node, title_limit=title_limit)}", pos="{pos["x"]:.2f},{pos["y"]:.2f}!", '
            f'pin=true, width={pos["w"]:.4f}, height={pos["h"]:.4f}, fillcolor="{fill}", color="{color}", penwidth={penwidth}];'
        )

    for parent_id, child_list in children.items():
        if not child_list:
            continue
        lines.append(f'  "{parent_id}" -> "{child_list[0].id}" [arrowhead=none, color="#cbd5e1", penwidth=1.2];')
        for left, right in zip(child_list, child_list[1:]):
            lines.append(f'  "{left.id}" -> "{right.id}" [arrowhead=none, color="#e2e8f0", penwidth=1.0];')

    seen_edges = set()
    for edge in graph.edges:
        if edge.type == "part_of":
            continue
        if edge.from_id == edge.to_id:
            continue
        if edge.from_id not in positions or edge.to_id not in positions:
            continue
        key = (edge.from_id, edge.to_id, edge.type)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        color = EDGE_COLORS.get(edge.type, "#6b7280")
        label_part = f', label="{edge.type}"' if show_edge_labels else ""
        lines.append(
            f'  "{edge.from_id}" -> "{edge.to_id}" [color="{color}", fontcolor="{color}", penwidth=0.9{label_part}];'
        )

    legend_x = graph_width + 140.0
    legend_y = graph_height - 60.0
    lines.append(
        f'  legend_coarse [label="Hierarchical node", pos="{legend_x:.2f},{legend_y:.2f}!", pin=true, width=1.3, height=0.45, '
        'fillcolor="#dbeafe", color="#2563eb", penwidth=1.4];'
    )
    lines.append(
        f'  legend_fine [label="Base node", pos="{legend_x:.2f},{legend_y - 65.0:.2f}!", pin=true, width=1.3, height=0.45, '
        'fillcolor="#ecfccb", color="#65a30d", penwidth=1.0];'
    )
    for idx, (edge_type, color) in enumerate(EDGE_COLORS.items()):
        y = legend_y - 150.0 - idx * 50.0
        lines.append(
            f'  legend_{edge_type}_a [label="", shape=point, width=0.01, pos="{legend_x - 35.0:.2f},{y:.2f}!", '
            'pin=true, color="#ffffff", fillcolor="#ffffff"];'
        )
        lines.append(
            f'  legend_{edge_type}_b [label="{edge_type}", shape=plaintext, pos="{legend_x + 40.0:.2f},{y:.2f}!", '
            f'pin=true, fontcolor="{color}"];'
        )
        lines.append(f'  legend_{edge_type}_a -> legend_{edge_type}_b [color="{color}", penwidth=1.0];')

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_temporal_graph(
    graph_path: Path,
    output_prefix: Path,
    title_limit: int = 28,
    show_edge_labels: bool = True,
) -> Tuple[Path, Path]:
    if shutil.which("dot") is None or shutil.which("neato") is None:
        raise RuntimeError("Graphviz binaries 'dot' and 'neato' must be installed and on PATH.")

    graph = load_graph(graph_path)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    final_dot_text = build_fixed_overlay_dot(
        graph,
        title_limit=title_limit,
        show_edge_labels=show_edge_labels,
    )
    dot_path = output_prefix.with_suffix(".dot")
    png_path = output_prefix.with_suffix(".png")
    dot_path.write_text(final_dot_text)
    subprocess.run(
        ["neato", "-n2", "-Tpng", str(dot_path), "-o", str(png_path)],
        check=True,
    )
    return dot_path, png_path


def default_output_prefix(graph_path: Path) -> Path:
    visuals_dir = graph_path.parent / "visuals"
    graph_stem = graph_path.stem
    return visuals_dir / f"{graph_stem}_temporal_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a stacked temporal PNG for a hierarchical lecture graph.")
    parser.add_argument("--graph", required=True, help="Path to graph JSON.")
    parser.add_argument(
        "--output-prefix",
        help="Output path prefix without extension. Defaults to <graph_dir>/visuals/<graph_stem>_temporal_final.",
    )
    parser.add_argument("--title-limit", type=int, default=28, help="Maximum title length shown in each node label.")
    parser.add_argument(
        "--hide-edge-labels",
        action="store_true",
        help="Hide semantic edge labels on the graph while keeping the legend.",
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
