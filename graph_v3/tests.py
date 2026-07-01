from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_v1.models import Span

from graph_v3.builder import (
    _compute_window_candidates,
    _repair_segments,
    _segments_from_boundaries,
    _select_boundaries_constrained,
    build_graph_v3,
)
from graph_v3.llm import GraphV3LLMClient, resolve_env


def test_embedding_signal_generates_peak_candidates() -> None:
    spans = []
    t = 0.0
    for i in range(6):
        spans.append(Span(span_id=f"a_{i}", lecture_id="lec", timestamp=t, text="dataframe join keys merge", modality="audio"))
        t += 12.0
    for i in range(6, 12):
        spans.append(Span(span_id=f"a_{i}", lecture_id="lec", timestamp=t, text="regular expressions token pattern matching", modality="audio"))
        t += 12.0

    llm = GraphV3LLMClient(prefer_remote=False)
    rows, cands = _compute_window_candidates(
        spans=spans,
        llm=llm,
        window_id="w_test",
        smooth_width=3,
        candidate_peak_percentile=70,
    )
    assert rows, "expected non-empty score rows"
    assert cands, "expected at least one candidate around the topic shift"
    near_shift = [c for c in cands if 55.0 <= float(c["timestamp"]) <= 90.0]
    assert near_shift, f"expected candidate near shift boundary, got={cands}"


def test_constrained_selector_respects_duration_bounds() -> None:
    spans = [Span(span_id=f"a_{i}", lecture_id="lec", timestamp=float(i * 20), text=f"x{i}", modality="audio") for i in range(20)]
    candidates = [
        {"timestamp": 80.0, "score": 0.8},
        {"timestamp": 140.0, "score": 0.85},
        {"timestamp": 220.0, "score": 0.88},
        {"timestamp": 280.0, "score": 0.82},
    ]
    selected, _meta = _select_boundaries_constrained(
        start_ts=0.0,
        end_ts=380.0,
        candidates=candidates,
        spans=spans,
        min_segment_sec=60.0,
        max_segment_sec=200.0,
        target_nodes=4,
        duration_penalty=0.25,
    )
    segments = _segments_from_boundaries(0.0, 380.0, selected)
    assert segments, "selector should produce at least one segment"
    for left, right in segments:
        dur = right - left
        assert dur >= 60.0 - 1e-6, f"segment too short: {dur}"
        assert dur <= 200.0 + 1e-6, f"segment too long: {dur}"


def test_repair_segments_eliminates_gaps_and_overlaps() -> None:
    # Deliberately includes a gap and an overlap.
    segments = [(0.0, 80.0), (90.0, 170.0), (160.0, 240.0)]
    repaired, actions = _repair_segments(
        segments=segments,
        container_start=0.0,
        container_end=240.0,
        min_segment_sec=30.0,
        max_segment_sec=120.0,
    )
    assert actions, "expected repair actions"
    assert repaired[0][0] == 0.0
    assert repaired[-1][1] == 240.0
    for (l0, r0), (l1, r1) in zip(repaired, repaired[1:]):
        assert abs(r0 - l1) <= 1e-6, f"expected contiguous boundary, got {r0} != {l1}"
        assert r0 > l0 and r1 > l1


def test_env_resolution_precedence() -> None:
    old = dict(os.environ)
    try:
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://primary-endpoint.example"
        os.environ["azure_endpoint"] = "https://fallback-endpoint.example"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-08-01"
        os.environ["api_version"] = "2023-01-01"
        os.environ["AZURE_OPENAI_API_KEY"] = "azure-key"
        os.environ["OPENAI_API_KEY"] = "openai-key"
        env = resolve_env()
        assert env["endpoint"] == "https://primary-endpoint.example"
        assert env["endpoint_source"] == "AZURE_OPENAI_ENDPOINT"
        assert env["api_version"] == "2024-08-01"
        assert env["api_version_source"] == "AZURE_OPENAI_API_VERSION"
        assert env["api_key"] == "azure-key"
        assert env["api_key_source"] == "AZURE_OPENAI_API_KEY"
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_smoke_build_lecture6_outputs_and_coverage() -> None:
    repo = Path(__file__).resolve().parents[1]
    audio = repo / "graph_data" / "source_lectures" / "data100_sp26" / "audio" / "Lecture 6 (Feb 5).json"
    video = repo / "graph_data" / "source_lectures" / "data100_sp26" / "video" / "Lecture 6 (Feb 5).json"
    out_dir = repo / "graph_v3" / "builds" / "data100_sp26" / "lecture6_smoke"
    out_graph = out_dir / "graph_v3.json"

    if out_dir.exists():
        shutil.rmtree(out_dir)

    try:
        graph, trace, validation, _weak = build_graph_v3(
            audio_path=audio,
            video_path=video,
            output_path=out_graph,
            prefer_remote_llm=False,
            verbose=False,
        )

        assert graph.nodes, "smoke build should produce nodes"
        assert out_graph.exists(), "graph output missing"
        assert (out_dir / "graph_v3_trace.json").exists(), "trace output missing"
        assert (out_dir / "graph_v3_validation.json").exists(), "validation output missing"
        assert trace.get("coarse") is not None

        coverage_error_codes = {
            "fine_gap_at_coarse_start",
            "fine_gap_at_coarse_end",
            "fine_gap_within_coarse",
            "fine_overlap_within_coarse",
            "fine_coverage_gap",
            "fine_coverage_gap_at_start",
            "fine_coverage_gap_at_end",
        }
        seen = {e.get("code") for e in validation.get("errors", [])}
        assert not (seen & coverage_error_codes), f"unexpected coverage errors: {seen & coverage_error_codes}"
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir)


def test_outputs_live_under_graph_v3_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    target = repo / "graph_v3" / "builds" / "_path_check" / "graph_v3.json"
    assert "graph_v3" in target.parts, target
    assert target.parent.parent.name == "builds", target.parent.parent
    assert target.parent.parent.parent.name == "graph_v3", target.parent.parent.parent


def test_builder_rejects_non_graph_v3_output_path() -> None:
    repo = Path(__file__).resolve().parents[1]
    audio = repo / "graph_data" / "source_lectures" / "data100_sp26" / "audio" / "Lecture 6 (Feb 5).json"
    bad_output = repo / "tmp" / "graph_v3.json"
    try:
        build_graph_v3(
            audio_path=audio,
            video_path=None,
            output_path=bad_output,
            prefer_remote_llm=False,
            verbose=False,
        )
    except ValueError as exc:
        assert "graph_v3 outputs must live under a graph_v3 path" in str(exc)
        return
    raise AssertionError("expected ValueError for output path outside graph_v3")


if __name__ == "__main__":
    test_embedding_signal_generates_peak_candidates()
    test_constrained_selector_respects_duration_bounds()
    test_repair_segments_eliminates_gaps_and_overlaps()
    test_env_resolution_precedence()
    test_smoke_build_lecture6_outputs_and_coverage()
    test_outputs_live_under_graph_v3_path()
    test_builder_rejects_non_graph_v3_output_path()
    print(json.dumps({"ok": True}))
