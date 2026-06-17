from __future__ import annotations

import argparse
import json
from pathlib import Path

from .builder import build_graph_for_lecture
from .models import TutorSessionState
from .tutor import TutorRuntime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end single lecture graph + QA trace")
    p.add_argument("--audio", required=True)
    p.add_argument("--video")
    p.add_argument("--output-dir", default="graph_v1/builds/data100_sp26/demo")
    p.add_argument("--question", default="What are the key concepts in this part of lecture?")
    p.add_argument("--window-sec", type=int, default=75)
    p.add_argument("--overlap-sec", type=int, default=15)
    p.add_argument("--merge-every-n", type=int, default=10)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-hops", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "graph.json"

    graph, merges = build_graph_for_lecture(
        audio_path=Path(args.audio),
        video_path=Path(args.video) if args.video else None,
        output_path=graph_path,
        window_sec=args.window_sec,
        overlap_sec=args.overlap_sec,
        merge_every_n_windows=args.merge_every_n,
    )

    runtime = TutorRuntime(graph)
    state = TutorSessionState(session_id="demo_session")
    qa = runtime.run_guided_answer(args.question, state, top_k=args.top_k, max_hops=args.max_hops)

    trace_path = out_dir / "qa_trace.json"
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(qa, f, indent=2)

    print(
        json.dumps(
            {
                "graph_path": str(graph_path),
                "merge_log_path": str(graph_path.with_name(graph_path.stem + "_merge_log.json")),
                "qa_trace_path": str(trace_path),
                "lecture_id": graph.lecture_id,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "merges": len(merges),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
