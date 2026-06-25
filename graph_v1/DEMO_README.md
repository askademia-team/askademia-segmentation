# Graph V1 Demo — Interactive Hierarchical Visualizer

This demo shows the GMM+BIC hierarchical clusters, contextual bridges, and timestamp pooling using a small mocked lecture graph.

Prerequisites
- Python 3.8+ installed
- Recommended: create and activate a venv

Quick setup (macOS / Linux):
```bash
cd /Users/akkaedoodle/askademia-segmentation
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install streamlit pyvis networkx
```

Run the demo (Streamlit + PyVis):
```bash
cd /Users/akkaedoodle/askademia-segmentation
streamlit run graph_v1/demo_app.py
```

The demo automatically adds the repo root to Python's path, so imports resolve correctly.

What the demo does
- Opens an interactive pyvis network in your browser
- Provides three scenarios:
  - Scenario A: Broad recap (MST backbone vs random baseline)
  - Scenario B: Prerequisite query (Dijkstra path along `requires` edges)
  - Scenario C: Hyper-specific query (simulated density pooling → precise playback window)
- Highlights nodes and edges and prints old vs new system outputs side-by-side

Notes
- The demo uses an embedded mock dataset (10 nodes) and runs offline.
- To adapt to a real graph JSON, replace `build_sample_graph()` in `graph_v1/demo_app.py` with a loader that reads your graph and maps node/span fields.

Troubleshooting
- If the browser tab does not open automatically, the Streamlit UI shows a local URL (usually `http://localhost:8501`) — open it manually.
- If `pyvis` renders slowly, close other browser tabs or reduce the demo graph size.