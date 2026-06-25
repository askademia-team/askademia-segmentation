"""Sample graph generator for the interactive demo.
Produces a small multi-layer lecture graph (10 nodes) with hyperedges, spans,
and edges containing confidence scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class DemoSpan:
    span_id: str
    timestamp: float
    text: str
    modality: str = "audio"


def build_sample_graph() -> Dict[str, Any]:
    """Return a dict describing nodes, edges, hyperedges, spans for the demo.

    Layers:
      - layer 0: fine nodes (detailed segments)
      - layer 1: cluster parents (hyperedge nodes)
      - layer 2: global thematic nodes
    """
    # Create 10 fine nodes across three micro-topics
    nodes = [
        # Topic: Linear models
        {"id": "n1", "title": "Linear Regression Intro", "layer": 0, "source_spans": ["s1", "s2"], "start_ts": 10, "end_ts": 70},
        {"id": "n2", "title": "Ordinary Least Squares", "layer": 0, "source_spans": ["s3"], "start_ts": 70, "end_ts": 120},
        {"id": "n3", "title": "Regularization (Ridge, Lasso)", "layer": 0, "source_spans": ["s4"], "start_ts": 120, "end_ts": 170},
        # Topic: Optimization
        {"id": "n4", "title": "Gradient Descent", "layer": 0, "source_spans": ["s5"], "start_ts": 180, "end_ts": 230},
        {"id": "n5", "title": "Learning Rate and Convergence", "layer": 0, "source_spans": ["s6"], "start_ts": 230, "end_ts": 270},
        # Topic: Classification
        {"id": "n6", "title": "Logistic Regression", "layer": 0, "source_spans": ["s7"], "start_ts": 280, "end_ts": 330},
        {"id": "n7", "title": "Sigmoid and Loss", "layer": 0, "source_spans": ["s8"], "start_ts": 330, "end_ts": 370},
        {"id": "n8", "title": "Evaluation (AUC/ROC)", "layer": 0, "source_spans": ["s9"], "start_ts": 370, "end_ts": 410},
        # Cross-cutting topics
        {"id": "n9", "title": "Feature Scaling", "layer": 0, "source_spans": ["s10"], "start_ts": 420, "end_ts": 460},
        {"id": "n10", "title": "Bias-Variance Tradeoff", "layer": 0, "source_spans": ["s11"], "start_ts": 460, "end_ts": 510},
    ]

    # Hyperedges / clusters: two clusters L1 grouping linear-model nodes and classification nodes
    hyperedges = [
        {"id": "c1", "layer": 1, "cluster_node_id": "cluster_L1_1", "member_node_ids": ["n1", "n2", "n3"], "title": "Linear Models"},
        {"id": "c2", "layer": 1, "cluster_node_id": "cluster_L1_2", "member_node_ids": ["n6", "n7", "n8"], "title": "Classification"},
    ]

    # Global thematic node (layer 2)
    global_nodes = [{"id": "g1", "title": "Supervised Learning", "layer": 2, "member_clusters": ["c1", "c2"]}]

    # Edges (from -> to), types and confidences
    edges = [
        {"from": "n2", "to": "n1", "type": "part_of", "confidence": 0.9},
        {"from": "n3", "to": "n1", "type": "part_of", "confidence": 0.85},
        {"from": "n4", "to": "n5", "type": "related_to", "confidence": 0.7},
        {"from": "n5", "to": "n4", "type": "related_to", "confidence": 0.65},
        {"from": "n6", "to": "n7", "type": "part_of", "confidence": 0.9},
        {"from": "n7", "to": "n6", "type": "related_to", "confidence": 0.6},
        {"from": "n9", "to": "n2", "type": "requires", "confidence": 0.8},
        {"from": "n10", "to": "n3", "type": "requires", "confidence": 0.75},
        # cluster membership edges
        {"from": "n1", "to": "cluster_L1_1", "type": "part_of", "confidence": 0.95},
        {"from": "n2", "to": "cluster_L1_1", "type": "part_of", "confidence": 0.95},
        {"from": "n3", "to": "cluster_L1_1", "type": "part_of", "confidence": 0.95},
        {"from": "n6", "to": "cluster_L1_2", "type": "part_of", "confidence": 0.95},
        {"from": "n7", "to": "cluster_L1_2", "type": "part_of", "confidence": 0.95},
        {"from": "n8", "to": "cluster_L1_2", "type": "part_of", "confidence": 0.95},
        # clusters -> global
        {"from": "cluster_L1_1", "to": "g1", "type": "part_of", "confidence": 0.9},
        {"from": "cluster_L1_2", "to": "g1", "type": "part_of", "confidence": 0.9},
    ]

    # Mock spans (text snippets) mapped to span ids referenced above
    spans = {
        "s1": DemoSpan("s1", 12.0, "Introduction to linear regression and modeling."),
        "s2": DemoSpan("s2", 50.0, "Intuition about fitting a line by minimizing squared error."),
        "s3": DemoSpan("s3", 80.0, "Closed form Ordinary Least Squares derivation."),
        "s4": DemoSpan("s4", 130.0, "Regularization techniques: Ridge and Lasso.").__dict__,
        "s5": DemoSpan("s5", 190.0, "Gradient descent algorithm overview.").__dict__,
        "s6": DemoSpan("s6", 240.0, "Effect of learning rate on convergence.").__dict__,
        "s7": DemoSpan("s7", 285.0, "Logistic regression introduction.").__dict__,
        "s8": DemoSpan("s8", 340.0, "Sigmoid activation and loss functions.").__dict__,
        "s9": DemoSpan("s9", 380.0, "ROC and AUC metrics for model evaluation.").__dict__,
        "s10": DemoSpan("s10", 430.0, "Feature scaling is important for gradient-based methods.").__dict__,
        "s11": DemoSpan("s11", 470.0, "Bias vs variance tradeoff illustrated." ).__dict__,
    }

    return {
        "nodes": nodes,
        "hyperedges": hyperedges,
        "global_nodes": global_nodes,
        "edges": edges,
        "spans": spans,
    }
