"""Interactive demo using Streamlit and PyVis to visualize the hierarchical graph.

Run with:
    streamlit run graph_v1/demo_app.py

This script renders a lecture graph demo and shows several retrieval views
that compare the traversal outputs against the final answer.
"""

from __future__ import annotations

import sys
import os
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Tuple

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from graph_v1.data_cleaner import LectureDataCleaner

# Add repo root to sys.path so graph_v1 can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_v1.demo_data import build_sample_graph

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.mixture import GaussianMixture
except Exception:
    MiniBatchKMeans = None
    TfidfVectorizer = None
    TruncatedSVD = None
    GaussianMixture = None


# -- Utility: convert demo data into networkx graph

def build_nx_graph(demo: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()
    # Add fine nodes
    for n in demo["nodes"]:
        G.add_node(
            n["id"],
            title=n["title"],
            layer=n["layer"],
            start_ts=n.get("start_ts"),
            end_ts=n.get("end_ts"),
            text=n.get("text"),
            source_spans=n.get("source_spans", []),
        )
    # Add cluster nodes
    for c in demo["hyperedges"]:
        cid = c["cluster_node_id"]
        G.add_node(
            cid,
            title=c.get("title", cid),
            layer=1,
            is_cluster=True,
            members=c.get("member_node_ids", []),
            text=c.get("summary"),
            source_spans=c.get("source_span_ids", []),
        )
    # Add global nodes
    for g in demo.get("global_nodes", []):
        gid = g["id"]
        G.add_node(gid, title=g.get("title", gid), layer=2, is_global=True, members=g.get("member_clusters", []), text=g.get("summary"), source_spans=g.get("source_span_ids", []))
    # Add edges
    for e in demo["edges"]:
        G.add_edge(e["from"], e["to"], type=e.get("type", "related_to"), confidence=e.get("confidence", 0.5))
    return G


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def _clean_conversational_filler(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"(?i)\b(um+|uh+|er+|ah+|like|you know|sort of|kind of|basically|so yeah|okay so|i think|maybe)\b", " ", text)
    text = re.sub(r"(?i)\b(tricky early on|hard to interpret|not sure|i guess)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _title_looks_transcript_like(title: str) -> bool:
    title = _clean_conversational_filler(title)
    if not title:
        return True
    words = _tokenize(title)
    if len(words) < 2 or len(words) > 5:
        return True
    lowered = title.lower()
    if re.search(r"\b(we|you|i|let'?s|lets|let us|okay|ok|right|so|now|here|there)\b", lowered):
        return True
    if re.search(r"\b(going to|gonna|want to|need to|take a look|look at|talk about|see how|show you|think about|can see|we can|you can|let me|this is|that is)\b", lowered):
        return True
    if re.search(r"\b(discussion this week|i think i mentioned|right we're going|ok guys|um|uh)\b", lowered):
        return True
    if words[0] in {"we", "you", "i", "let", "lets", "okay", "ok", "right", "so", "now", "here", "there"}:
        return True
    return False


def _abstract_concept_title(text: str) -> str:
    cleaned = _clean_conversational_filler(text)
    tokens = [t for t in _tokenize(cleaned) if len(t) >= 3]
    stopwords = {
        "this", "that", "with", "from", "have", "were", "been", "into", "when", "what", "where",
        "there", "their", "about", "because", "after", "before", "would", "could", "should", "also",
        "lecture", "section", "topic", "today", "then", "just", "like", "okay", "yeah", "uh", "um",
        "going", "right", "guys", "mentioned", "discussion", "week", "we", "you", "i", "let", "lets",
        "let's", "now", "here", "there", "so", "talk", "look", "show", "see", "think", "mention",
        "discuss", "cover", "explain", "introduce", "walk", "through", "build", "use", "want", "need",
    }
    technical = [t for t in tokens if t not in stopwords]
    if not technical:
        return "Core Concept"
    title_words = technical[:4]
    if len(title_words) < 2:
        title_words = (title_words * 2)[:2]
    title = " ".join(word.capitalize() for word in title_words[:4])
    return "Core Concept" if _title_looks_transcript_like(title) else title


def _abstract_concept_summary(text: str, title: str) -> str:
    title_lower = title.lower()
    return f"This section explains {title_lower} as a core concept in the lecture and describes how it is used in practice."


def _load_json_entries(folder_path: str, filename: str) -> List[Dict[str, Any]]:
    fpath = os.path.join(folder_path, filename)
    if not os.path.isfile(fpath):
        return []
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _chunk_source_entries(
    lecture_id: str,
    source_name: str,
    entries: List[Dict[str, Any]],
    chunk_window_seconds: int,
    source_weight: float,
) -> List[Dict[str, Any]]:
    cleaner = LectureDataCleaner()
    cleaned_chunks = cleaner.batch_process_spans(
        [
            {
                "span_id": f"{lecture_id}_{source_name}_{i + 1}",
                "timestamp": float(entry.get("timestamp", 0.0)),
                "text": str(entry.get("text", "")),
                "modality": source_name,
            }
            for i, entry in enumerate(entries)
        ],
        min_words=60,
    )
    weighted_chunks: List[Dict[str, Any]] = []
    for i, ch in enumerate(cleaned_chunks):
        cleaned_text = str(ch.get("text", "")).strip()
        if not cleaned_text:
            continue
        weighted_chunks.append(
            {
                "source": source_name,
                "source_weight": source_weight,
                "span_id": ch.get("span_id") or f"{lecture_id}_{source_name}_s{i + 1}",
                "start_ts": float(ch["start_ts"]),
                "end_ts": float(ch["end_ts"]),
                "text": cleaned_text,
                "source_span_ids": ch.get("source_span_ids", []),
            }
        )
    return weighted_chunks


def _combine_weighted_chunks(audio_chunks: List[Dict[str, Any]], ocr_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    combined: List[Dict[str, Any]] = []
    max_len = max(len(audio_chunks), len(ocr_chunks))
    for idx in range(max_len):
        parts: List[str] = []
        source_span_ids: List[str] = []
        source_weights: Dict[str, float] = {}
        source_modalities: List[str] = []
        source_chunks: List[Dict[str, Any]] = []
        start_ts = None
        end_ts = None
        for chunk_list in (ocr_chunks, audio_chunks):
            if idx >= len(chunk_list):
                continue
            chunk = chunk_list[idx]
            source_span_ids.append(chunk["span_id"])
            source_weights[chunk["source"]] = float(chunk["source_weight"])
            source_modalities.append(chunk["source"])
            source_chunks.append(chunk)
            start_ts = chunk["start_ts"] if start_ts is None else min(start_ts, chunk["start_ts"])
            end_ts = chunk["end_ts"] if end_ts is None else max(end_ts, chunk["end_ts"])
            repeat = 2 if chunk["source"] == "ocr" else 1
            parts.extend([chunk["text"]] * repeat)
        candidate_text = " ".join(parts).strip()
        if not candidate_text:
            continue
        combined.append(
            {
                "start_ts": float(start_ts or 0.0),
                "end_ts": float(end_ts or 0.0),
                "text": candidate_text,
                "span_ids": sorted(set(source_span_ids)),
                "source_weights": source_weights,
                "source_modalities": sorted(set(source_modalities)),
                "source_chunks": source_chunks,
            }
        )
    return combined


def _chunk_lecture_segments(segments: List[Dict[str, Any]], window_seconds: int = 90) -> List[Dict[str, Any]]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: float(x.get("timestamp", 0.0)))
    chunks: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []
    chunk_start = float(segments[0].get("timestamp", 0.0))
    for seg in segments:
        ts = float(seg.get("timestamp", 0.0))
        if current and ts - chunk_start > window_seconds:
            text = " ".join([c.get("text", "").strip() for c in current if c.get("text")]).strip()
            if text:
                chunks.append(
                    {
                        "start_ts": float(current[0].get("timestamp", 0.0)),
                        "end_ts": float(current[-1].get("timestamp", 0.0)),
                        "text": text,
                    }
                )
            current = []
            chunk_start = ts
        current.append(seg)

    if current:
        text = " ".join([c.get("text", "").strip() for c in current if c.get("text")]).strip()
        if text:
            chunks.append(
                {
                    "start_ts": float(current[0].get("timestamp", 0.0)),
                    "end_ts": float(current[-1].get("timestamp", 0.0)),
                    "text": text,
                }
            )
    return chunks


def _fit_tfidf_embeddings(items: List[Dict[str, Any]]) -> Tuple[Any, Any, Any]:
    texts = [it["text"] for it in items]
    vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    if TruncatedSVD is not None and X.shape[1] > 32:
        n_components = min(64, max(8, min(X.shape[0] - 1, X.shape[1] - 1)))
        if n_components >= 2:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            reduced = svd.fit_transform(X)
            return vec, X, reduced
    return vec, X, X.toarray()


def _select_gmm_labels(embeddings: Any, max_components: int) -> Tuple[int, Any]:
    if GaussianMixture is None:
        raise RuntimeError("GaussianMixture unavailable")
    n = len(embeddings)
    max_components = max(2, min(max_components, n))
    best_k = 1
    best_labels = None
    best_bic = float("inf")
    for k in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, reg_covar=1e-4)
            gmm.fit(embeddings)
            bic = gmm.bic(embeddings)
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_labels = gmm.predict(embeddings)
        except Exception:
            continue
    if best_labels is None:
        raise RuntimeError("Unable to fit GMM")
    return best_k, best_labels


def _dynamic_layer_plan(n_items: int, avg_words: float) -> Tuple[int, int]:
    if n_items < 8:
        return 1, 2
    size_depth = int(round(math.log2(max(2, n_items))))
    density_depth = 1 if avg_words < 55 else 2 if avg_words < 90 else 3
    max_layers = max(2, min(8, size_depth - 1 + density_depth))
    base_max_clusters = max(2, min(30, int(round((n_items ** 0.5) * (1.4 if avg_words >= 60 else 1.1)))))
    return max_layers, base_max_clusters


def _layer_cluster_cap(n_items: int, layer: int, base_cap: int) -> int:
    intensity = max(0.35, 1.0 - 0.12 * (layer - 1))
    estimate = int(round(base_cap * intensity))
    return max(2, min(max(2, n_items - 1), estimate))


def _cluster_layer_gmm(
    items: List[Dict[str, Any]],
    layer: int,
    max_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if TfidfVectorizer is None or GaussianMixture is None or TruncatedSVD is None:
        return [], [], []
    if len(items) < 4:
        return [], [], []

    vec, X, embeddings = _fit_tfidf_embeddings(items)
    if len(embeddings) < 4:
        return [], [], []

    try:
        best_k, labels = _select_gmm_labels(embeddings, max_k)
    except Exception:
        return [], [], []

    if best_k <= 1 or best_k >= len(items):
        return [], [], []

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        groups[int(lab)].append(idx)
    if len(groups) < 2:
        return [], [], []

    vocab = vec.get_feature_names_out()
    parent_nodes: List[Dict[str, Any]] = []
    parent_edges: List[Dict[str, Any]] = []
    hyperedges: List[Dict[str, Any]] = []

    if hasattr(X, "toarray"):
        dense = X.toarray()
    else:
        dense = embeddings

    for gi, (_, member_idx) in enumerate(sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)):
        if len(member_idx) < 2:
            continue
        member_items = [items[i] for i in member_idx]
        if hasattr(X, "shape") and X.shape[1] > 0:
            cluster_vector = dense[member_idx].mean(axis=0)
            top_term_ids = cluster_vector.argsort()[-4:][::-1]
            top_terms = [vocab[t] for t in top_term_ids if cluster_vector[t] > 0]
        else:
            top_terms = []
        title = "Semantic cluster: " + (", ".join(top_terms) if top_terms else f"layer {layer} topic {gi + 1}")
        pid = f"cluster_l{layer}_{gi + 1}"
        parent_text = " ".join([m["text"] for m in member_items[:10]])
        start_ts = min([m.get("start_ts", 0.0) for m in member_items])
        end_ts = max([m.get("end_ts", 0.0) for m in member_items])
        span_ids = sorted({sid for m in member_items for sid in m.get("source_spans", [])})
        parent_nodes.append(
            {
                "id": pid,
                "title": title,
                "layer": layer,
                "text": parent_text,
                "source_spans": span_ids,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }
        )
        hyperedges.append(
            {
                "id": f"hyper_{layer}_{gi + 1}",
                "layer": layer,
                "cluster_node_id": pid,
                "member_node_ids": [m["id"] for m in member_items],
                "internal_edge_ids": [],
                "title": title,
                "summary": parent_text[:280],
                "explanation": "Semantic membership hyperedge linking fine lecture chunks into one higher-level topic.",
                "source_span_ids": span_ids,
                "start_ts": start_ts,
                "end_ts": end_ts,
            }
        )
        for m in member_items:
            parent_edges.append({"from": m["id"], "to": pid, "type": "part_of", "confidence": 0.92})

    return parent_nodes, parent_edges, hyperedges


def build_graph_from_media_folders(
    audio_folder_path: str,
    ocr_folder_path: str | None = None,
    chunk_window_seconds: int = 90,
    audio_weight: float = 1.0,
    ocr_weight: float = 2.25,
) -> Dict[str, Any]:
    audio_files = sorted([f for f in os.listdir(audio_folder_path) if f.lower().endswith(".json")])
    ocr_files = sorted([f for f in os.listdir(ocr_folder_path)] if ocr_folder_path and os.path.isdir(ocr_folder_path) else [])
    file_names = sorted(set(audio_files) | set(ocr_files))

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    hyperedges: List[Dict[str, Any]] = []
    spans: Dict[str, Dict[str, Any]] = {}

    # Layer 0: low-level lecture chunks.
    for fname in file_names:
        lecture_id = _sanitize_name(os.path.splitext(fname)[0])
        audio_entries = _load_json_entries(audio_folder_path, fname)
        ocr_entries = _load_json_entries(ocr_folder_path, fname) if ocr_folder_path else []
        if not audio_entries and not ocr_entries:
            continue

        audio_chunks = _chunk_source_entries(lecture_id, "audio", audio_entries, chunk_window_seconds, audio_weight)
        ocr_chunks = _chunk_source_entries(lecture_id, "ocr", ocr_entries, chunk_window_seconds, ocr_weight)
        if not audio_chunks and not ocr_chunks:
            continue

        weighted_chunks = _combine_weighted_chunks(audio_chunks, ocr_chunks)
        prev_node_id = None
        buffered_chunks: List[Dict[str, Any]] = []
        for i, ch in enumerate(weighted_chunks):
            buffered_chunks.append(ch)
            candidate_text = " ".join(c["text"] for c in buffered_chunks if c.get("text")).strip()
            if not candidate_text:
                continue
            token_count = len(_tokenize(candidate_text))
            if token_count < 45:
                continue
            density = len([t for t in _tokenize(candidate_text) if len(t) > 2]) / max(1, token_count)
            if density < 0.52:
                continue
            nid = f"{lecture_id}_n{i+1}"
            title = _abstract_concept_title(candidate_text)
            abstract_summary = _abstract_concept_summary(candidate_text, title)
            source_span_ids = list(ch["span_ids"])
            nodes.append(
                {
                    "id": nid,
                    "title": title,
                    "layer": 0,
                    "source_spans": source_span_ids,
                    "start_ts": ch["start_ts"],
                    "end_ts": ch["end_ts"],
                    "text": abstract_summary,
                    "raw_text": candidate_text,
                    "source_weights": {k: float(v) for k, v in ch.get("source_weights", {}).items()},
                    "source_modalities": ch.get("source_modalities", []),
                }
            )
            for source_chunk in ch.get("source_chunks", []):
                spans[source_chunk["span_id"]] = {
                    "span_id": source_chunk["span_id"],
                    "timestamp": source_chunk["start_ts"],
                    "text": source_chunk["text"],
                    "modality": source_chunk["source"],
                    "lecture_id": lecture_id,
                    "weight": float(source_chunk["source_weight"]),
                }
            if prev_node_id is not None:
                edges.append(
                    {
                        "from": prev_node_id,
                        "to": nid,
                        "type": "temporal_next",
                        "confidence": 0.88,
                    }
                )
            prev_node_id = nid
            buffered_chunks = []

    avg_words = 0.0
    if nodes:
        avg_words = sum(len(_tokenize(n.get("text", ""))) for n in nodes) / len(nodes)
    max_layers, base_max_clusters = _dynamic_layer_plan(len(nodes), avg_words)
    created_layers: List[Dict[str, Any]] = []

    # Dynamic semantic hierarchy: arbitrary number of layers.
    current = [n for n in nodes]
    layer = 1
    while True:
        layer_cap = _layer_cluster_cap(len(current), layer, base_max_clusters)
        parent_nodes, parent_edges, parent_hyperedges = _cluster_layer_gmm(current, layer=layer, max_k=layer_cap)
        if not parent_nodes:
            break
        created_layers.append(
            {
                "layer_idx": layer,
                "input_nodes": len(current),
                "created_clusters": len(parent_nodes),
                "max_clusters": layer_cap,
            }
        )
        nodes.extend(parent_nodes)
        edges.extend(parent_edges)
        hyperedges.extend(parent_hyperedges)
        if len(parent_nodes) == 1:
            break
        compression_ratio = len(parent_nodes) / max(1, len(current))
        if len(parent_nodes) >= len(current) or compression_ratio > 0.92:
            break
        current = parent_nodes
        layer += 1
        if layer > max_layers:
            break

    return {
        "nodes": nodes,
        "edges": edges,
        "hyperedges": hyperedges,
        "global_nodes": [],
        "spans": spans,
        "metadata": {
            "source_folders": {
                "audio": audio_folder_path,
                "ocr": ocr_folder_path,
            },
            "source_weights": {
                "audio": audio_weight,
                "ocr": ocr_weight,
            },
            "semantic_cluster_layers": {
                "created_layers": created_layers,
                "max_layers": max_layers,
                "base_max_clusters": base_max_clusters,
            },
        },
    }


def build_graph_from_audio_folder(folder_path: str, chunk_window_seconds: int = 90) -> Dict[str, Any]:
    return build_graph_from_media_folders(folder_path, None, chunk_window_seconds=chunk_window_seconds)


@st.cache_data(show_spinner=True)
def load_graph_data(folder_path: str, ocr_folder_path: str | None = None) -> Dict[str, Any]:
    if folder_path and os.path.isdir(folder_path):
        try:
            graph = build_graph_from_media_folders(folder_path, ocr_folder_path)
            if graph.get("nodes"):
                return graph
        except Exception:
            pass
    return build_sample_graph()


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if t]


def _lexical_overlap_score(query: str, text: str) -> float:
    q = set(_tokenize(query))
    t = set(_tokenize(text))
    if not q or not t:
        return 0.0
    return len(q.intersection(t)) / max(1, len(q))


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def _phrase_match_score(query: str, text: str) -> float:
    nq = _normalized_text(query)
    nt = _normalized_text(text)
    if not nq or not nt:
        return 0.0

    query_phrases = [
        nq,
        nq.replace("  ", " "),
        " ".join(nq.split()[:3]),
        " ".join(nq.split()[:4]),
    ]
    text_phrases = [
        nt,
        nt.replace("  ", " "),
        " ".join(nt.split()[:3]),
        " ".join(nt.split()[:4]),
    ]
    if any(qp and qp in nt for qp in query_phrases):
        return 1.0
    if any(tp and tp in nq for tp in text_phrases):
        return 0.9

    q_tokens = nq.split()
    t_tokens = nt.split()
    if len(q_tokens) >= 2:
        q_2grams = {" ".join(q_tokens[i : i + 2]) for i in range(len(q_tokens) - 1)}
        t_join = " ".join(t_tokens)
        if any(g and g in t_join for g in q_2grams):
            return 0.8
    return 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_iso_timestamp_to_epoch(raw: Any) -> float | None:
    if not isinstance(raw, str):
        return None
    ts = raw.strip()
    if not ts:
        return None
    try:
        # Accept both timezone-aware and naive ISO strings.
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return float(datetime.fromisoformat(ts).timestamp())
    except Exception:
        return None


def _fine_time_bounds(G: nx.DiGraph) -> Tuple[float, float]:
    fine_nodes = [d for _, d in G.nodes(data=True) if d.get("layer") == 0]
    if not fine_nodes:
        return (0.0, 0.0)
    starts = [float(d.get("start_ts", 0.0)) for d in fine_nodes]
    ends = [float(d.get("end_ts", 0.0)) for d in fine_nodes]
    return (min(starts), max(ends))


def _expected_lecture_ts_from_query_timestamp(
    G: nx.DiGraph,
    query_timestamp_epoch: float | None,
    question_time_bounds: Tuple[float, float] | None,
) -> float | None:
    if query_timestamp_epoch is None or question_time_bounds is None:
        return None
    q_min, q_max = question_time_bounds
    if q_max <= q_min:
        return None
    g_min, g_max = _fine_time_bounds(G)
    if g_max <= g_min:
        return None
    query_pos = _clamp01((query_timestamp_epoch - q_min) / max(1e-9, (q_max - q_min)))
    return g_min + query_pos * (g_max - g_min)


def _node_timestamp_alignment_score(G: nx.DiGraph, node_id: str, expected_lecture_ts: float | None) -> float:
    if expected_lecture_ts is None:
        return 0.0
    node = G.nodes[node_id]
    g_min, g_max = _fine_time_bounds(G)
    graph_span = max(1.0, g_max - g_min)
    spread = max(25.0, graph_span * 0.16)
    node_mid = (float(node.get("start_ts", 0.0)) + float(node.get("end_ts", 0.0))) / 2.0
    dist = abs(node_mid - expected_lecture_ts)
    return math.exp(-dist / spread)


def infer_query_target_node(
    G: nx.DiGraph,
    query: str,
    min_score: float = 0.12,
    query_timestamp_epoch: float | None = None,
    question_time_bounds: Tuple[float, float] | None = None,
) -> Tuple[str, float]:
    ranked = rank_low_level_nodes(
        G,
        query,
        top_k=5,
        query_timestamp_epoch=query_timestamp_epoch,
        question_time_bounds=question_time_bounds,
    )
    if not ranked:
        return (None, 0.0)
    best = ranked[0]
    if best["priority"] == 0 and best["semantic"] < min_score:
        return (None, best["semantic"])
    return (best["node_id"], float(best["combined_score"]))


def rank_low_level_nodes(
    G: nx.DiGraph,
    query: str,
    top_k: int = 5,
    query_timestamp_epoch: float | None = None,
    question_time_bounds: Tuple[float, float] | None = None,
) -> List[Dict[str, Any]]:
    fine_nodes = [n for n, d in G.nodes(data=True) if int(d.get("layer", 0)) == 0]
    query_norm = _normalized_text(query)
    scored = []
    expected_lecture_ts = _expected_lecture_ts_from_query_timestamp(G, query_timestamp_epoch, question_time_bounds)
    for nid in fine_nodes:
        node = G.nodes[nid]
        title = node.get("title", nid)
        text = str(node.get("text") or "")
        title_norm = _normalized_text(title)
        text_norm = _normalized_text(text)
        exact_title = 1.0 if query_norm and query_norm in title_norm else 0.0
        exact_text = 0.9 if query_norm and query_norm in text_norm else 0.0
        phrase_title = _phrase_match_score(query, title)
        phrase_text = _phrase_match_score(query, text)
        semantic = _lexical_overlap_score(query, f"{title} {text}")
        priority = 3 if exact_title > 0 else 2 if exact_text > 0 else 1 if (phrase_title > 0 or phrase_text > 0) else 0
        text_match = max(exact_title, exact_text, phrase_title, phrase_text)
        timestamp_score = _node_timestamp_alignment_score(G, nid, expected_lecture_ts)
        text_strength = float(priority) + 0.25 * text_match + 0.1 * semantic
        if expected_lecture_ts is not None:
            # Heavy weightage for timestamp-context alignment while preserving
            # priority ordering as the first tie-break dimension.
            combined = 0.65 * timestamp_score + 0.35 * (text_strength / 3.35)
            sort_key = (priority, combined, text_match, semantic)
        else:
            combined = text_strength / 3.35
            sort_key = (priority, text_match, semantic)
        scored.append(
            {
                "node_id": nid,
                "priority": priority,
                "text_match": text_match,
                "semantic": semantic,
                "timestamp_score": timestamp_score,
                "combined_score": combined,
                "sort_key": sort_key,
            }
        )
    scored.sort(key=lambda row: row["sort_key"], reverse=True)
    return scored[:top_k]


def compute_query_mst(G: nx.DiGraph, target_id: str, max_hops: int = 2) -> Tuple[List[str], List[Tuple[str, str]]]:
    und = nx.Graph()
    for n in G.nodes():
        und.add_node(n)
    for u, v, d in G.edges(data=True):
        und.add_edge(u, v, weight=float(d.get("confidence", 0.5)))

    if target_id not in und:
        return [], []

    # Build a local neighborhood around the embedding-target node.
    lengths = nx.single_source_shortest_path_length(und, target_id, cutoff=max_hops)
    neighborhood_nodes = list(lengths.keys())
    sub = und.subgraph(neighborhood_nodes)
    if sub.number_of_nodes() <= 1:
        return neighborhood_nodes, []

    mst = nx.maximum_spanning_tree(sub, weight="weight")
    return list(mst.nodes()), list(mst.edges())


def compute_dijkstra_bridge(G: nx.DiGraph, target_id: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    DG = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if d.get("type") in ("part_of", "requires", "related_to"):
            conf = max(0.1, float(d.get("confidence", 0.5)))
            DG.add_edge(u, v, weight=1.0 / conf)

    abstract_candidates = [n for n, d in G.nodes(data=True) if int(d.get("layer", 0)) > 0]
    best_path = []
    best_cost = None
    for dst in abstract_candidates:
        try:
            path = nx.shortest_path(DG, source=target_id, target=dst, weight="weight")
            cost = nx.path_weight(DG, path, weight="weight")
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_path = path
        except Exception:
            continue

    return best_path, list(zip(best_path, best_path[1:]))


def extract_semantic_moments(
    G: nx.DiGraph,
    demo: Dict[str, Any],
    query: str,
    focus_nodes: List[str],
    top_k: int = 4,
    expected_lecture_ts: float | None = None,
) -> List[Dict[str, Any]]:
    spans = demo.get("spans", {})
    candidates = []
    for nid in focus_nodes:
        node = G.nodes[nid]
        if int(node.get("layer", 0)) != 0:
            continue
        title = node.get("title", nid)
        for sid in node.get("source_spans", []):
            sp = spans.get(sid)
            if not sp:
                continue
            text = sp.get("text", "")
            semantic = _lexical_overlap_score(query, f"{title} {text}")
            density = min(1.0, len(_tokenize(text)) / 20.0)
            moment_ts = float(sp.get("timestamp", 0.0))
            if expected_lecture_ts is not None:
                g_min, g_max = _fine_time_bounds(G)
                spread = max(25.0, (g_max - g_min) * 0.16)
                time_score = math.exp(-abs(moment_ts - expected_lecture_ts) / spread)
                # Heavy timestamp bias for context alignment.
                score = 0.6 * time_score + 0.25 * semantic + 0.15 * density
            else:
                score = 0.7 * semantic + 0.3 * density
            candidates.append(
                {
                    "node_id": nid,
                    "span_id": sid,
                    "timestamp": moment_ts,
                    "text": text,
                    "score": score,
                    "title": title,
                }
            )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _collect_string_values_for_keys(obj: Any, keys: set[str]) -> List[str]:
    results: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in keys and isinstance(v, str):
                txt = v.strip()
                if txt:
                    results.append(txt)
            results.extend(_collect_string_values_for_keys(v, keys))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_collect_string_values_for_keys(item, keys))
    return results


def _collect_question_records_for_keys(obj: Any, keys: set[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        ts_epoch = _parse_iso_timestamp_to_epoch(obj.get("timestamp"))
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in keys and isinstance(v, str):
                query = v.strip()
                if query:
                    out.append(
                        {
                            "query": query,
                            "timestamp": obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None,
                            "timestamp_epoch": ts_epoch,
                        }
                    )
            out.extend(_collect_question_records_for_keys(v, keys))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_collect_question_records_for_keys(item, keys))
    return out


def extract_question_records_from_json(obj: Any) -> List[Dict[str, Any]]:
    preferred = _collect_question_records_for_keys(obj, {"user_input"})
    if preferred:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for row in preferred:
            key = (row["query"], row.get("timestamp"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    fallback_keys = {"question", "query", "student_question", "prompt", "user_question"}
    fallback = _collect_question_records_for_keys(obj, fallback_keys)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for row in fallback:
        key = (row["query"], row.get("timestamp"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def extract_questions_from_json(obj: Any) -> List[str]:
    return _dedupe_keep_order([row["query"] for row in extract_question_records_from_json(obj)])


def compute_retrieval_trace(
    G: nx.DiGraph,
    demo: Dict[str, Any],
    query: str,
    query_timestamp_epoch: float | None = None,
    question_time_bounds: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    expected_lecture_ts = _expected_lecture_ts_from_query_timestamp(G, query_timestamp_epoch, question_time_bounds)
    target, target_score = infer_query_target_node(
        G,
        query,
        query_timestamp_epoch=query_timestamp_epoch,
        question_time_bounds=question_time_bounds,
    )
    if not target:
        return {
            "target": None,
            "target_score": target_score,
            "ood": True,
            "mst_nodes": [],
            "mst_edges": [],
            "dijkstra_nodes": [],
            "dijkstra_edges": [],
            "moments": [],
            "moment_nodes": [],
            "expected_lecture_ts": expected_lecture_ts,
        }

    ranked_rows = rank_low_level_nodes(
        G,
        query,
        top_k=3,
        query_timestamp_epoch=query_timestamp_epoch,
        question_time_bounds=question_time_bounds,
    )
    low_level_focus = [row["node_id"] for row in ranked_rows]
    mst_nodes, mst_edges = compute_query_mst(G, target_id=target, max_hops=2)
    dijkstra_nodes, dijkstra_edges = compute_dijkstra_bridge(G, target_id=target)
    focus = sorted(set(low_level_focus + mst_nodes + dijkstra_nodes + [target]))
    moments = extract_semantic_moments(
        G,
        demo,
        query,
        focus_nodes=focus,
        top_k=3,
        expected_lecture_ts=expected_lecture_ts,
    )
    moment_nodes = sorted(set([m["node_id"] for m in moments]))
    return {
        "target": target,
        "target_score": target_score,
        "ood": False,
        "mst_nodes": mst_nodes,
        "mst_edges": mst_edges,
        "dijkstra_nodes": dijkstra_nodes,
        "dijkstra_edges": dijkstra_edges,
        "moments": moments,
        "moment_nodes": moment_nodes,
        "expected_lecture_ts": expected_lecture_ts,
        "ranked_low_level_nodes": ranked_rows,
    }


def synthesize_final_response(query: str, G: nx.DiGraph, trace: Dict[str, Any]) -> Dict[str, str]:
    if trace.get("ood"):
        return {
            "local_evidence": "No confident local evidence match in this lecture graph.",
            "abstract_bridge": "No bridge executed because no reliable target node was found.",
            "moments": "No moments extracted.",
            "final_answer": (
                "I could not confidently ground this question in the current lecture graph. "
                "Try a query tied to lecture topics (e.g., regularization, logistic regression, AUC/ROC), "
                "or use a graph built from a lecture/person domain that includes this entity."
            ),
        }

    target = trace["target"]
    mst_titles = [G.nodes[n].get("title", n) for n in trace["mst_nodes"][:4]]
    bridge_titles = [G.nodes[n].get("title", n) for n in trace["dijkstra_nodes"]]
    moments = trace["moments"][:2]
    expected_lecture_ts = trace.get("expected_lecture_ts")

    target_title = G.nodes[target].get("title", target)
    mst_summary = ", ".join(mst_titles) if mst_titles else "no local concepts"
    bridge_summary = " -> ".join(bridge_titles) if bridge_titles else "no abstract bridge path"
    if moments:
        moment_summary = " ".join([f"[{m['timestamp']:.0f}s] {m['text']}" for m in moments])
    else:
        moment_summary = "I did not find a clean example snippet, so I am keeping the explanation conceptual."

    if expected_lecture_ts is not None:
        temporal_hint = "This matches the part of the lecture where the idea is introduced and then developed into its practical form."
    else:
        temporal_hint = ""

    final_answer = (
        f"For '{query}', the lecture starts from '{target_title}'. The nearby concepts ({mst_summary}) provide the foundation, "
        f"the broader bridge ({bridge_summary}) connects to the next idea, and the supporting snippets are: {moment_summary} "
        f"{temporal_hint}".strip()
    )
    return {
        "local_evidence": f"Closest lecture concept: {target_title}. Nearby concepts: {mst_summary}.",
        "abstract_bridge": f"Concept bridge: {bridge_summary}.",
        "moments": moment_summary,
        "final_answer": final_answer,
    }


# -- PyVis render helper

def render_pyvis(
    G: nx.DiGraph,
    target_node: str = None,
    mst_nodes: List[str] = None,
    mst_edges: List[Tuple[str, str]] = None,
    dijkstra_nodes: List[str] = None,
    dijkstra_edges: List[Tuple[str, str]] = None,
    moment_nodes: List[str] = None,
    title_text: str = "",
) -> str:
    # Inline assets are more reliable on Safari where cross-origin/CDN loading in
    # embedded contexts can be blocked by privacy settings.
    net = Network(height="750px", width="100%", notebook=False, cdn_resources="in_line")
    net.barnes_hut()

    mst_nodes = mst_nodes or []
    mst_edges = mst_edges or []
    dijkstra_nodes = dijkstra_nodes or []
    dijkstra_edges = dijkstra_edges or []
    moment_nodes = moment_nodes or []
    traversal_nodes = set(mst_nodes) | set(dijkstra_nodes) | set(moment_nodes)
    grey_color = "#d1d5db"
    grey_edge = "#cbd5e1"

    def _edge_in_mst(u: str, v: str) -> bool:
        return (u, v) in mst_edges or (v, u) in mst_edges

    # Color palette: base layer colors + traversal overlays.
    palette = {0: "#a7c5eb", 1: "#5b8def", 2: "#7c6dff"}

    for n, d in G.nodes(data=True):
        layer = d.get("layer", 0)
        title = d.get("title", n)
        size = 20 if layer == 0 else 34 if layer == 1 else 46
        color = grey_color
        # Moment nodes should always stay pink, even when they overlap with
        # Dijkstra or target overlays.
        if n in moment_nodes:
            color = "#ff4fc3"
        elif n in dijkstra_nodes:
            color = "#228b22"
        elif n in mst_nodes:
            color = "#ff4fc3"
        elif target_node and n == target_node:
            color = "#ff00b8"
            size = max(size, 30)
        opacity = 1.0 if n in traversal_nodes or (target_node and n == target_node) else 0.28
        net.add_node(n, label=title, title=title, color=color, size=size, font={"color": "#111111" if opacity >= 0.8 else "#9ca3af"}, opacity=opacity)

    for u, v, data in G.edges(data=True):
        etype = data.get("type", "related_to")
        conf = float(data.get("confidence", 0.5))
        width = max(1.0, conf * 5)
        color = grey_edge
        dashes = False
        label = ""
        if _edge_in_mst(u, v):
            color = "#ff4fc3"
            label = "Traversal"
        if (u, v) in dijkstra_edges:
            color = "#228b22"
            dashes = True
            label = "Dijkstra"
        edge_opacity = 1.0 if (_edge_in_mst(u, v) or (u, v) in dijkstra_edges) else 0.18
        net.add_edge(
            u,
            v,
            value=conf,
            title=f"{etype} ({conf:.2f})",
            width=width,
            color=color,
            dashes=dashes,
            label=label,
            opacity=edge_opacity,
        )

    # Group cluster members visually by creating invisible nodes or circles (pyvis limitations)
    # We'll rely on node color and labels for clusters in this simple demo.

    # Generate HTML directly and keep notebook=False to avoid notebook template issues.
    return net.generate_html(notebook=False)


# -- Legacy baseline handlers

def legacy_baseline_a(G: nx.DiGraph, demo: Dict[str, Any]) -> Dict[str, Any]:
    """Broad/Thematic Query

    Old: pick random nodes
    New: find connected components and MST backbone
    """
    import random

    # Old baseline
    old_nodes = random.sample([n for n in G.nodes() if G.nodes[n].get("layer", 0) == 0], 3)

    # New system: connected components and MST of undirected projection
    und = nx.Graph()
    und.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        und.add_edge(u, v, weight=d.get("confidence", 0.5))
    comps = list(nx.connected_components(und))
    # pick the largest component
    largest = max(comps, key=lambda c: len(c))
    sub = und.subgraph(largest)
    if sub.number_of_nodes() == 1:
        mst_edges = []
        mst_nodes = list(sub.nodes())
    else:
        mst = nx.maximum_spanning_tree(sub, weight="weight")
        mst_edges = list(mst.edges())
        mst_nodes = list(mst.nodes())

    old_payload = "\n".join([f"{n}: {G.nodes[n]['title']}" for n in old_nodes])
    new_payload = "\n".join([f"{n}: {G.nodes[n]['title']}" for n in mst_nodes])

    return {"old_nodes": old_nodes, "old_payload": old_payload, "new_nodes": mst_nodes, "new_edges": mst_edges, "new_payload": new_payload}


def legacy_baseline_b(G: nx.DiGraph, demo: Dict[str, Any], target_id: str) -> Dict[str, Any]:
    """Prerequisite Query: find shortest path along 'requires' edges backward to parent hyperedge.

    Old: highlight only the target node
    New: Dijkstra shortest-path along requires edges
    """
    # Old baseline
    old_nodes = [target_id]

    # Build directed graph with weights inversely proportional to confidence
    DG = nx.DiGraph()
    for u, v, d in G.edges(data=True):
        if d.get("type") in ("requires", "part_of"):
            weight = 1.0 / max(0.1, float(d.get("confidence", 0.5)))
            DG.add_edge(u, v, weight=weight)

    # Find parent hyperedge candidates (cluster nodes)
    parents = [n for n, dd in G.nodes(data=True) if dd.get("layer") == 1]
    shortest_path = []
    for p in parents:
        try:
            path = nx.shortest_path(DG, source=target_id, target=p, weight="weight")
            if not shortest_path or len(path) < len(shortest_path):
                shortest_path = path
        except Exception:
            continue

    new_nodes = shortest_path if shortest_path else []
    new_edges = list(zip(new_nodes, new_nodes[1:]))

    old_payload = f"{target_id}: {G.nodes[target_id]['title']}"
    new_payload = " -> ".join([G.nodes[n]["title"] for n in new_nodes]) if new_nodes else "No prerequisite path found"

    return {"old_nodes": old_nodes, "old_payload": old_payload, "new_nodes": new_nodes, "new_edges": new_edges, "new_payload": new_payload}


def legacy_baseline_c(G: nx.DiGraph, demo: Dict[str, Any], target_id: str) -> Dict[str, Any]:
    """Hyper-specific query: rolling window density pooling simulation

    Old: scatter snippets
    New: compute a fake density score and return media coordinates
    """
    # Old baseline: random scattered snippets from other nodes
    old_nodes = [n for n in G.nodes() if G.nodes[n].get("layer") == 0]
    import random
    old_sample = random.sample(old_nodes, 3)
    old_payload = "\n".join([f"{n}: {G.nodes[n]['title']}" for n in old_sample])

    # New: simulate density pooling by assigning higher score to the target node if it exists
    if target_id not in G.nodes():
        return {"old_nodes": old_sample, "old_payload": old_payload, "new_nodes": [], "new_edges": [], "new_payload": "Target not found"}

    # Simulate density: nodes with adjacent 'requires' or high confidence part_of get lower/higher scores
    def score_node(nid: str) -> float:
        base = 0.1
        for u, v, d in G.in_edges(nid, data=True):
            if d.get("type") == "requires":
                base += 0.4
            if d.get("type") == "part_of":
                base += 0.2
        return base

    scores = {n: score_node(n) for n in G.nodes()}
    peak_node = max(scores.items(), key=lambda kv: kv[1])[0]

    # Fake playback window centered on peak node's start/end
    start = G.nodes[peak_node].get("start_ts", 0)
    end = G.nodes[peak_node].get("end_ts", start + 30)
    new_payload = f"Playback Window: {start:.0f}s - {end:.0f}s" 

    return {"old_nodes": old_sample, "old_payload": old_payload, "new_nodes": [peak_node], "new_edges": [], "new_payload": new_payload}


# -- Streamlit app

st.set_page_config(page_title="Graph_v1 Interactive Demo", layout="wide")
st.title("Knowledge Graph: Hierarchical Demo")

# Build graph (real CS189 audio data by default if available).
default_audio_path = "/Users/akkaedoodle/Downloads/audio"
default_ocr_path = "/Users/akkaedoodle/Downloads/OCR"
demo = load_graph_data(default_audio_path, default_ocr_path)
G = build_nx_graph(demo)

# Left column: controls
col1, col2 = st.columns([1, 2])
with col1:
    st.header("Controls")
    if os.path.isdir(default_audio_path) and os.path.isdir(default_ocr_path):
        st.caption("Graph source: combined audio + OCR lecture corpora")
    elif os.path.isdir(default_audio_path):
        st.caption("Graph source: audio lecture corpus")
    else:
        st.caption("Graph source: mock sample graph")
    max_layer = max([int(G.nodes[n].get("layer", 0)) for n in G.nodes()]) if G.number_of_nodes() else 0
    st.caption(f"Loaded nodes={G.number_of_nodes()} edges={G.number_of_edges()} layers={max_layer + 1}")
    uploaded_questions_file = st.file_uploader("Upload student question JSON", type=["json"])
    uploaded_question_records: List[Dict[str, Any]] = []
    uploaded_questions: List[str] = []
    if uploaded_questions_file is not None:
        try:
            payload = json.loads(uploaded_questions_file.getvalue().decode("utf-8"))
            uploaded_question_records = extract_question_records_from_json(payload)
            uploaded_questions = [row["query"] for row in uploaded_question_records]
            ts_count = len([row for row in uploaded_question_records if row.get("timestamp_epoch") is not None])
            st.caption(
                f"Loaded {len(uploaded_questions)} candidate question(s) from JSON (preferring 'user_input'); "
                f"{ts_count} include parseable timestamps."
            )
        except Exception as exc:
            st.error(f"Could not parse JSON: {exc}")

    if "query_seed" not in st.session_state:
        st.session_state["query_seed"] = "Give me a high-level overview of linear and classification concepts."
    if "query_timestamp_epoch" not in st.session_state:
        st.session_state["query_timestamp_epoch"] = None

    if uploaded_questions and st.button("Sample Random Uploaded Question"):
        sampled = random.choice(uploaded_question_records)
        st.session_state["query_seed"] = sampled["query"]
        st.session_state["query_timestamp_epoch"] = sampled.get("timestamp_epoch")

    query = st.text_area("Student query", value=st.session_state["query_seed"], height=90)
    run = st.button("Run Query")

    uploaded_time_epochs = [row["timestamp_epoch"] for row in uploaded_question_records if row.get("timestamp_epoch") is not None]
    question_time_bounds: Tuple[float, float] | None = None
    if uploaded_time_epochs:
        question_time_bounds = (min(uploaded_time_epochs), max(uploaded_time_epochs))

    if uploaded_question_records:
        timestamp_lookup = {
            row["query"]: row.get("timestamp_epoch")
            for row in uploaded_question_records
            if row.get("timestamp_epoch") is not None
        }
        if query in timestamp_lookup:
            st.session_state["query_timestamp_epoch"] = timestamp_lookup[query]

    if st.session_state.get("query_timestamp_epoch") is not None:
        st.caption("Timestamp context is active for this query and heavily weighted in retrieval.")

    st.markdown("---")
    st.markdown("**Legend**")
    st.markdown(
        "- Background graph: muted grey\n"
        "- Low-level traversal nodes: Bright pink (`#FF4FC3`)\n"
        "- Dijkstra bridge nodes/edges: Forest green (`#228B22`)\n"
        "- Inferred target node: Hot pink (`#FF00B8`)\n"
        "- Non-traversed edges/nodes: faded grey\n"
        "- Edge width ~ confidence"
    )
    st.markdown(
        "**Hyperedge meaning**\n"
        "Hyperedges represent semantic membership relationships: they group several fine lecture chunks into one higher-level concept block, so the graph can show topic containment and abstraction rather than only pairwise links."
    )

# Right column: visualization and outputs
with col2:
    st.header("Graph Visualization")
    html = None
    if run:
        trace = compute_retrieval_trace(
            G,
            demo,
            query,
            query_timestamp_epoch=st.session_state.get("query_timestamp_epoch"),
            question_time_bounds=question_time_bounds,
        )
        target = trace["target"]
        mst_nodes = trace["mst_nodes"]
        mst_edges = trace["mst_edges"]
        dijkstra_nodes = trace["dijkstra_nodes"]
        dijkstra_edges = trace["dijkstra_edges"]
        moments = trace["moments"]
        moment_nodes = trace["moment_nodes"]

        st.subheader("Student Query")
        st.text_area("Raw student input", value=query, height=80, disabled=True)
        if trace.get("ood"):
            st.warning("No confident match in this lecture graph for the query terms.")
            st.caption(f"Best lexical target score was low ({trace.get('target_score', 0.0):.3f}).")
        else:
            st.caption(
                f"Inferred initial embedding target: {target} ({G.nodes[target].get('title', target)}), "
                f"score={trace.get('target_score', 0.0):.3f}"
            )
        final_response = synthesize_final_response(query, G, trace)

        t1, t2, t3, t4, t5 = st.tabs([
            "Combined Traversals",
            "MST Traversal",
            "Dijkstra Bridge",
            "Moment Extraction",
            "Final Response",
        ])

        with t1:
            html = render_pyvis(
                G,
                target_node=target,
                mst_nodes=mst_nodes,
                mst_edges=mst_edges,
                dijkstra_nodes=dijkstra_nodes,
                dijkstra_edges=dijkstra_edges,
                moment_nodes=moment_nodes,
            )
            st.caption("Combined view: pink=MST/moment nodes, green=Dijkstra bridge.")
            components.html(html, height=760, scrolling=True)

        with t2:
            mst_html = render_pyvis(
                G,
                target_node=target,
                mst_nodes=mst_nodes,
                mst_edges=mst_edges,
            )
            st.caption("Local MST backbone around the inferred lecture concept.")
            components.html(mst_html, height=760, scrolling=True)

        with t3:
            d_html = render_pyvis(
                G,
                target_node=target,
                dijkstra_nodes=dijkstra_nodes,
                dijkstra_edges=dijkstra_edges,
            )
            bridge_text = " -> ".join([G.nodes[n].get("title", n) for n in dijkstra_nodes]) if dijkstra_nodes else "No bridge path found."
            st.text(bridge_text)
            components.html(d_html, height=760, scrolling=True)

        with t4:
            m_html = render_pyvis(
                G,
                target_node=target,
                moment_nodes=moment_nodes,
            )
            for m in moments:
                st.markdown(
                    f"- **{m['title']}** @ {m['timestamp']:.1f}s (score={m['score']:.3f})\n"
                    f"  - snippet: {m['text']}"
                )
            components.html(m_html, height=760, scrolling=True)

        with t5:
            st.subheader("Model Final Response")
            st.write("Structured breakdown of how the model combined traversal evidence.")
            st.markdown("**Local Evidence**")
            st.info(final_response["local_evidence"])
            st.markdown("**Abstract Bridge**")
            st.info(final_response["abstract_bridge"])
            st.markdown("**Moments**")
            st.info(final_response["moments"])
            st.markdown("**Final Answer**")
            st.success(final_response["final_answer"])
    else:
        # Render the static graph initially
        html = render_pyvis(G)
        st.write("Enter a query and run it to inspect the lecture graph.")
        components.html(html, height=760, scrolling=True)
