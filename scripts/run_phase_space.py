#!/usr/bin/env python3
"""Generate a 3D procedural phase-space visualisation from scene data.

Reads a scene JSONL file and produces an interactive 3D HTML plot
(or static PNG fallback) showing the surgical trajectory through
phase space.

This represents procedural progression and activity,
NOT anatomical reconstruction.

Usage examples:

    # Single video (interactive HTML)
    python scripts/run_phase_space.py --video video49

    # Custom scene directory and output
    python scripts/run_phase_space.py --video video49 \
        --scene-dir experiments/scene \
        --output-dir experiments/analysis

    # Use confidence only for Z axis
    python scripts/run_phase_space.py --video video49 \
        --activity confidence

    # Process all available scene files
    python scripts/run_phase_space.py --all

    # Downsample for large videos (every 10th frame)
    python scripts/run_phase_space.py --video video49 --downsample 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.phase_space_3d import (  # noqa: E402
    build_phase_space,
    get_phase_segments,
    get_transition_points,
    plot_phase_space_3d,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate 3D procedural phase-space visualisation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video", type=str, help="Video ID (e.g. 'video49')")
    grp.add_argument("--all", action="store_true", help="Process all scene files")

    p.add_argument(
        "--scene-dir", type=Path,
        default=_PROJECT_ROOT / "experiments" / "scene",
        help="Directory with scene JSONL files (default: experiments/scene/)",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "experiments" / "analysis",
        help="Output directory (default: experiments/analysis/)",
    )
    p.add_argument(
        "--activity", type=str, default="combined",
        choices=["combined", "instrument_count", "confidence"],
        help="Z-axis activity metric (default: combined)",
    )
    p.add_argument(
        "--downsample", type=int, default=1,
        help="Plot every Nth point (default: 1 = all)",
    )
    p.add_argument(
        "--point-size", type=int, default=3,
        help="Trajectory point size (default: 3)",
    )

    return p.parse_args()


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

    return pairs


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_scene_files(args, scene_dir)
    log.info("Videos to process: %d", len(pairs))

    for video_id, scene_path in pairs:
        log.info("Processing %s ...", video_id)

        # Build phase space
        space = build_phase_space(scene_path, activity_mode=args.activity)

        # Extract segments and transitions for summary
        segments = get_phase_segments(space)
        transitions = get_transition_points(space)

        log.info("  %d frames, %d segments, %d transitions",
                 len(space["time"]), len(segments), len(transitions))

        for tr in transitions:
            log.info(
                "  Transition at %.1fs: %s -> %s (conf=%.3f)",
                tr["time"], tr["from_phase"], tr["to_phase"],
                tr["confidence_at_transition"],
            )

        # Visualise
        out_path = output_dir / f"{video_id}_phase_space_3d.html"
        saved = plot_phase_space_3d(
            space,
            output_path=out_path,
            downsample=args.downsample,
            point_size=args.point_size,
        )
        log.info("  Saved: %s", saved)

        # Save summary JSON alongside
        summary = {
            "video_id": video_id,
            "total_frames": len(space["time"]),
            "duration_sec": float(space["time"][-1] - space["time"][0]),
            "activity_mode": args.activity,
            "num_segments": len(segments),
            "num_transitions": len(transitions),
            "segments": segments,
            "transitions": transitions,
        }
        summary_path = output_dir / f"{video_id}_phase_space_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info("  Summary: %s", summary_path)

    log.info("Done. Processed %d video(s).", len(pairs))


if __name__ == "__main__":
    main()
