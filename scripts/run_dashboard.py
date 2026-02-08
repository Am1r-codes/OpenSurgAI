#!/usr/bin/env python3
"""Record annotated demo videos with full pipeline overlays.

Reads the original surgical video alongside scene, explanation, and
segmentation outputs, and produces a smooth annotated MP4 with:
- Detection bounding boxes with class labels and confidence
- Semi-transparent segmentation mask overlay
- Phase label with confidence bar
- Nemotron explanation text panel
- Frame counter and timestamp

Usage examples:

    # Single video with all overlays
    python scripts/run_dashboard.py \
        --video data/cholec80/videos/video01.mp4

    # Specify scene and explanation files explicitly
    python scripts/run_dashboard.py \
        --video data/cholec80/videos/video01.mp4 \
        --scene experiments/scene/video01_scene.jsonl \
        --explanation experiments/explanation/video01_explanations.jsonl

    # All videos, auto-discover upstream outputs
    python scripts/run_dashboard.py --all

    # Record only frames 1000-2000 for a highlight clip
    python scripts/run_dashboard.py \
        --video data/cholec80/videos/video01.mp4 \
        --start-frame 1000 --end-frame 2000

    # Adjust overlay appearance
    python scripts/run_dashboard.py --all \
        --mask-alpha 0.4 --bbox-thickness 3 --font-scale 0.7

    # Process every 2nd frame for faster rendering
    python scripts/run_dashboard.py --all --stride 2
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

from src.dashboard.recorder import DemoRecorder  # noqa: E402
from src.dashboard.renderer import OverlayRenderer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def project_root() -> Path:
    return _PROJECT_ROOT


def default_data_dir() -> Path:
    return project_root() / "data" / "cholec80"


def default_experiments_dir() -> Path:
    return project_root() / "experiments"


def default_output_dir() -> Path:
    return project_root() / "experiments" / "dashboard"


def collect_jobs(args: argparse.Namespace) -> list[dict]:
    """Build a list of recording jobs.

    Each job is a dict with keys: video_path, video_id, scene_path,
    explanation_path.
    """
    exp = Path(args.experiments_dir)

    if args.video:
        vp = Path(args.video)
        if not vp.exists():
            log.error("Video not found: %s", vp)
            sys.exit(1)
        vid = vp.stem

        scene_path = Path(args.scene) if args.scene else exp / "scene" / f"{vid}_scene.jsonl"
        expl_path = Path(args.explanation) if args.explanation else exp / "explanation" / f"{vid}_explanations.jsonl"

        return [{
            "video_path": vp,
            "video_id": vid,
            "scene_path": scene_path if scene_path.exists() else None,
            "explanation_path": expl_path if expl_path.exists() else None,
        }]

    if args.all:
        videos_dir = Path(args.data_dir) / "videos"
        if not videos_dir.is_dir():
            log.error("Videos directory not found: %s", videos_dir)
            sys.exit(1)
        videos = sorted(videos_dir.glob("video*.*"))
        if not videos:
            log.error("No video files found in %s", videos_dir)
            sys.exit(1)

        # Filter by split
        if args.split:
            split_file = Path(args.data_dir) / "splits" / f"{args.split}.txt"
            if not split_file.exists():
                log.error("Split file not found: %s", split_file)
                sys.exit(1)
            allowed = set(split_file.read_text().strip().splitlines())
            videos = [v for v in videos if v.stem in allowed]

        jobs: list[dict] = []
        for vp in videos:
            vid = vp.stem
            scene_path = exp / "scene" / f"{vid}_scene.jsonl"
            expl_path = exp / "explanation" / f"{vid}_explanations.jsonl"
            jobs.append({
                "video_path": vp,
                "video_id": vid,
                "scene_path": scene_path if scene_path.exists() else None,
                "explanation_path": expl_path if expl_path.exists() else None,
            })
        return jobs

    log.error("Specify --video PATH or --all")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record annotated demo videos with full pipeline overlays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── input selection ───────────────────────────────────────────────
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--video", type=str, help="Path to a single video file")
    grp.add_argument("--all", action="store_true", help="Process all videos")

    p.add_argument(
        "--data-dir", type=Path, default=default_data_dir(),
        help="Cholec80 data directory",
    )
    p.add_argument(
        "--split", type=str, choices=["train", "test"], default=None,
        help="Restrict --all to a specific split",
    )

    # ── upstream data (auto-discovered by default) ────────────────────
    p.add_argument(
        "--experiments-dir", type=Path, default=default_experiments_dir(),
        help="Experiments root for auto-discovering upstream outputs",
    )
    p.add_argument("--scene", type=str, default=None, help="Scene JSONL path (overrides auto)")
    p.add_argument("--explanation", type=str, default=None, help="Explanation JSONL path (overrides auto)")
    p.add_argument(
        "--mask-dir", type=Path, default=None,
        help="Segmentation mask directory (default: experiments/segmentation/)",
    )

    # ── frame range ───────────────────────────────────────────────────
    p.add_argument("--start-frame", type=int, default=0, help="First frame to include")
    p.add_argument("--end-frame", type=int, default=None, help="Last frame (exclusive)")
    p.add_argument("--stride", type=int, default=1, help="Frame stride (default: 1)")

    # ── rendering options ─────────────────────────────────────────────
    p.add_argument("--mask-alpha", type=float, default=0.35, help="Segmentation overlay opacity (default: 0.35)")
    p.add_argument("--bbox-thickness", type=int, default=2, help="Bbox line thickness (default: 2)")
    p.add_argument("--font-scale", type=float, default=0.6, help="Base font scale (default: 0.6)")
    p.add_argument("--no-confidence", action="store_true", help="Hide confidence values")

    # ── output ────────────────────────────────────────────────────────
    p.add_argument("--codec", type=str, default="mp4v", help="FourCC codec (default: mp4v)")
    p.add_argument("--output-fps", type=float, default=None, help="Output FPS (default: source FPS)")
    p.add_argument(
        "--output-dir", type=Path, default=default_output_dir(),
        help="Output directory (default: experiments/dashboard/)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    jobs = collect_jobs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Videos to record: %d", len(jobs))
    log.info("Output dir      : %s", output_dir)

    # Build renderer
    renderer = OverlayRenderer(
        mask_alpha=args.mask_alpha,
        bbox_thickness=args.bbox_thickness,
        font_scale=args.font_scale,
        show_confidence=not args.no_confidence,
    )

    # Resolve mask directory
    mask_dir = args.mask_dir
    if mask_dir is None:
        candidate = Path(args.experiments_dir) / "segmentation"
        if candidate.is_dir():
            mask_dir = candidate

    recorder = DemoRecorder(
        renderer=renderer,
        codec=args.codec,
        output_fps=args.output_fps,
        mask_dir=mask_dir,
    )

    all_summaries: list[dict] = []
    t0 = time.perf_counter()

    for job in jobs:
        vid = job["video_id"]
        out_path = output_dir / f"{vid}_demo.mp4"

        # Log data availability
        has_scene = job["scene_path"] is not None
        has_expl = job["explanation_path"] is not None
        has_masks = mask_dir is not None
        log.info(
            "%s: scene=%s, explanation=%s, masks=%s",
            vid,
            "yes" if has_scene else "no",
            "yes" if has_expl else "no",
            "yes" if has_masks else "no",
        )

        summary = recorder.record(
            video_path=job["video_path"],
            output_path=out_path,
            scene_path=job["scene_path"],
            explanation_path=job["explanation_path"],
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stride=args.stride,
        )
        all_summaries.append(summary)

    total_elapsed = time.perf_counter() - t0
    total_frames = sum(s["frames_written"] for s in all_summaries)

    # ── write run summary ─────────────────────────────────────────────
    run_summary = {
        "codec": args.codec,
        "mask_alpha": args.mask_alpha,
        "stride": args.stride,
        "num_videos": len(jobs),
        "total_frames": total_frames,
        "total_elapsed_sec": round(total_elapsed, 2),
        "overall_render_fps": round(total_frames / total_elapsed, 1) if total_elapsed > 0 else 0,
        "per_video": all_summaries,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))

    log.info("=" * 60)
    log.info("Dashboard recording complete.")
    log.info("  Videos     : %d", len(jobs))
    log.info("  Frames     : %d", total_frames)
    log.info("  Elapsed    : %.1fs", total_elapsed)
    log.info("  Render FPS : %.1f", run_summary["overall_render_fps"])
    log.info("  Output     : %s", output_dir)
    log.info("  Summary    : %s", summary_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
