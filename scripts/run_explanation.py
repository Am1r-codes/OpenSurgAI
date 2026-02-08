#!/usr/bin/env python3
"""Generate natural-language explanations from unified scene data.

Reads SceneState JSONL files (from ``run_scene_assembly.py``), sends
them to the Nemotron API for grounded explanation generation, and saves
per-frame explanations as JSON Lines under ``experiments/explanation/``.

Requires the ``NEMOTRON_API_KEY`` (or ``NVIDIA_API_KEY``) environment
variable to be set.

Usage examples:

    # Single video
    python scripts/run_explanation.py \
        --video video01

    # All videos with scene outputs
    python scripts/run_explanation.py --all

    # Limit to 50 frames per video (cost control / testing)
    python scripts/run_explanation.py --all --max-frames 50

    # Custom model and chunk size
    python scripts/run_explanation.py --all \
        --model nvidia/llama-3.1-nemotron-70b-instruct \
        --chunk-size 10 --temperature 0.1

    # Custom scene directory
    python scripts/run_explanation.py --all \
        --scene-dir experiments/scene
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.explanation.pipeline import ExplanationPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def project_root() -> Path:
    return _PROJECT_ROOT


def default_experiments_dir() -> Path:
    return project_root() / "experiments"


def default_data_dir() -> Path:
    return project_root() / "data" / "cholec80"


def discover_scene_files(
    args: argparse.Namespace,
    scene_dir: Path,
) -> list[tuple[str, Path]]:
    """Discover (video_id, scene_path) pairs."""
    if args.video:
        p = scene_dir / f"{args.video}_scene.jsonl"
        if not p.exists():
            log.error("Scene file not found: %s", p)
            sys.exit(1)
        return [(args.video, p)]

    if args.all:
        if not scene_dir.is_dir():
            log.error("Scene directory not found: %s", scene_dir)
            sys.exit(1)
        pairs: list[tuple[str, Path]] = []
        for f in sorted(scene_dir.iterdir()):
            if f.name.endswith("_scene.jsonl"):
                vid = f.name[: -len("_scene.jsonl")]
                pairs.append((vid, f))

        if not pairs:
            log.error("No scene JSONL files found in %s", scene_dir)
            sys.exit(1)

        # Optionally filter by split
        if args.split:
            data_dir = Path(args.data_dir)
            split_file = data_dir / "splits" / f"{args.split}.txt"
            if not split_file.exists():
                log.error("Split file not found: %s", split_file)
                sys.exit(1)
            allowed = set(split_file.read_text().strip().splitlines())
            pairs = [(v, p) for v, p in pairs if v in allowed]

        return pairs

    log.error("Specify --video VIDEO_ID or --all")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate natural-language explanations from scene data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── input selection ───────────────────────────────────────────────
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--video", type=str, help="Video ID (e.g. 'video01')")
    grp.add_argument("--all", action="store_true", help="Process all scene files")

    p.add_argument(
        "--data-dir", type=Path, default=default_data_dir(),
        help="Cholec80 data directory for split filtering",
    )
    p.add_argument(
        "--split", type=str, choices=["train", "test"], default=None,
        help="Restrict --all to a specific split",
    )

    # ── scene input ───────────────────────────────────────────────────
    p.add_argument(
        "--scene-dir", type=Path,
        default=default_experiments_dir() / "scene",
        help="Directory with scene JSONL files (default: experiments/scene/)",
    )

    # ── API / model ───────────────────────────────────────────────────
    p.add_argument(
        "--api-key", type=str, default=None,
        help="Nemotron API key (default: NEMOTRON_API_KEY env var)",
    )
    p.add_argument(
        "--base-url", type=str,
        default="https://integrate.api.nvidia.com/v1",
        help="API base URL",
    )
    p.add_argument(
        "--model", type=str,
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        help="Nemotron model identifier",
    )
    p.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    p.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max completion tokens per API call (default: 512)",
    )
    p.add_argument(
        "--chunk-size", type=int, default=5,
        help="Frames per API call (default: 5)",
    )

    # ── cost control ──────────────────────────────────────────────────
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Max frames to process per video (default: all)",
    )

    # ── output ────────────────────────────────────────────────────────
    p.add_argument(
        "--output-dir", type=Path,
        default=default_experiments_dir() / "explanation",
        help="Output directory (default: experiments/explanation/)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)

    pairs = discover_scene_files(args, scene_dir)
    log.info("Videos to explain: %d", len(pairs))
    log.info("Model     : %s", args.model)
    log.info("Chunk size: %d", args.chunk_size)
    log.info("Max frames: %s", args.max_frames or "all")
    log.info("Output    : %s", output_dir)

    pipeline = ExplanationPipeline(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict] = []
    t0 = time.perf_counter()

    for video_id, scene_path in pairs:
        explanation_path = output_dir / f"{video_id}_explanations.jsonl"

        summary = pipeline.process_scene_file(
            scene_path=scene_path,
            output_path=explanation_path,
            max_frames=args.max_frames,
        )
        all_summaries.append(summary)

    total_elapsed = time.perf_counter() - t0
    total_frames = sum(s["total_frames"] for s in all_summaries)
    total_grounded = sum(s["grounded_frames"] for s in all_summaries)
    total_prompt_tokens = sum(s["total_prompt_tokens"] for s in all_summaries)
    total_completion_tokens = sum(s["total_completion_tokens"] for s in all_summaries)

    # ── write run summary ─────────────────────────────────────────────
    run_summary = {
        "model": args.model,
        "temperature": args.temperature,
        "chunk_size": args.chunk_size,
        "max_frames_per_video": args.max_frames,
        "num_videos": len(pairs),
        "total_frames": total_frames,
        "total_grounded": total_grounded,
        "grounded_pct": round(total_grounded / total_frames * 100, 1) if total_frames > 0 else 0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_elapsed_sec": round(total_elapsed, 2),
        "per_video": all_summaries,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))

    log.info("=" * 60)
    log.info("Explanation generation complete.")
    log.info("  Videos     : %d", len(pairs))
    log.info("  Frames     : %d", total_frames)
    log.info("  Grounded   : %d (%.1f%%)", total_grounded, run_summary["grounded_pct"])
    log.info("  Tokens     : %d prompt + %d completion",
             total_prompt_tokens, total_completion_tokens)
    log.info("  Elapsed    : %.1fs", total_elapsed)
    log.info("  Summary    : %s", summary_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
