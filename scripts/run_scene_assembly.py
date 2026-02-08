#!/usr/bin/env python3
"""Assemble unified surgical scene representations from upstream outputs.

Reads the JSONL outputs from detection, segmentation, and phase
recognition pipelines and combines them into a single per-frame scene
JSON suitable as the sole input to downstream explanation modules.

Expects the standard output layout produced by the upstream scripts::

    experiments/
        detection/
            video01_detections.jsonl
        segmentation/
            video01_masks.jsonl
        phase/
            video01_phases.jsonl

Produces::

    experiments/
        scene/
            video01_scene.jsonl      <- unified per-frame scene
            run_summary.json

Usage examples:

    # Single video (auto-discovers upstream files)
    python scripts/run_scene_assembly.py --video video01

    # All videos in the dataset
    python scripts/run_scene_assembly.py --all

    # Custom experiment directories
    python scripts/run_scene_assembly.py --all \
        --detection-dir experiments/detection \
        --segmentation-dir experiments/segmentation \
        --phase-dir experiments/phase

    # Only videos that have at least detection + phase outputs
    python scripts/run_scene_assembly.py --all --require detection phase
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

from src.scene.assembler import SceneAssembler  # noqa: E402

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


def discover_video_ids(
    args: argparse.Namespace,
    det_dir: Path,
    seg_dir: Path,
    phase_dir: Path,
) -> list[str]:
    """Discover video IDs from upstream output files or dataset splits."""
    if args.video:
        return [args.video]

    if args.all:
        # Collect video IDs from all upstream directories
        ids: set[str] = set()
        for d, suffix in [(det_dir, "_detections.jsonl"),
                          (seg_dir, "_masks.jsonl"),
                          (phase_dir, "_phases.jsonl")]:
            if d.is_dir():
                for f in d.iterdir():
                    if f.name.endswith(suffix):
                        vid = f.name[: -len(suffix)]
                        ids.add(vid)

        if not ids:
            log.error(
                "No upstream outputs found in:\n  %s\n  %s\n  %s",
                det_dir, seg_dir, phase_dir,
            )
            sys.exit(1)

        # Optionally filter by split
        if args.split:
            data_dir = Path(args.data_dir)
            split_file = data_dir / "splits" / f"{args.split}.txt"
            if not split_file.exists():
                log.error("Split file not found: %s", split_file)
                sys.exit(1)
            allowed = set(split_file.read_text().strip().splitlines())
            ids = ids & allowed

        return sorted(ids)

    log.error("Specify --video VIDEO_ID or --all")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assemble unified scene representations from upstream outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── input selection ───────────────────────────────────────────────
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--video", type=str,
        help="Video ID to assemble (e.g. 'video01')",
    )
    grp.add_argument(
        "--all", action="store_true",
        help="Discover and assemble all videos with upstream outputs",
    )

    p.add_argument(
        "--data-dir", type=Path, default=default_data_dir(),
        help="Cholec80 data directory for split filtering (default: data/cholec80/)",
    )
    p.add_argument(
        "--split", type=str, choices=["train", "test"], default=None,
        help="Restrict --all to a specific split",
    )

    # ── upstream directories ──────────────────────────────────────────
    exp = default_experiments_dir()
    p.add_argument(
        "--detection-dir", type=Path, default=exp / "detection",
        help="Directory with detection JSONL files",
    )
    p.add_argument(
        "--segmentation-dir", type=Path, default=exp / "segmentation",
        help="Directory with segmentation metadata JSONL files",
    )
    p.add_argument(
        "--phase-dir", type=Path, default=exp / "phase",
        help="Directory with phase recognition JSONL files",
    )

    # ── requirements ──────────────────────────────────────────────────
    p.add_argument(
        "--require", nargs="+",
        choices=["detection", "segmentation", "phase"],
        default=[],
        help="Only process videos that have ALL of the specified sources",
    )

    # ── output ────────────────────────────────────────────────────────
    p.add_argument(
        "--output-dir", type=Path, default=exp / "scene",
        help="Output directory for scene JSONL files (default: experiments/scene/)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    det_dir = Path(args.detection_dir)
    seg_dir = Path(args.segmentation_dir)
    phase_dir = Path(args.phase_dir)
    output_dir = Path(args.output_dir)

    video_ids = discover_video_ids(args, det_dir, seg_dir, phase_dir)
    log.info("Videos discovered: %d", len(video_ids))
    log.info("Detection dir    : %s", det_dir)
    log.info("Segmentation dir : %s", seg_dir)
    log.info("Phase dir        : %s", phase_dir)
    log.info("Output dir       : %s", output_dir)

    assembler = SceneAssembler()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict] = []
    skipped: list[str] = []
    t0 = time.perf_counter()

    for video_id in video_ids:
        # Resolve upstream file paths (may or may not exist)
        det_path = det_dir / f"{video_id}_detections.jsonl"
        seg_path = seg_dir / f"{video_id}_masks.jsonl"
        phase_path = phase_dir / f"{video_id}_phases.jsonl"

        # Check requirements
        if args.require:
            source_map = {
                "detection": det_path,
                "segmentation": seg_path,
                "phase": phase_path,
            }
            missing = [
                src for src in args.require
                if not source_map[src].exists()
            ]
            if missing:
                log.warning(
                    "Skipping %s – missing required source(s): %s",
                    video_id, ", ".join(missing),
                )
                skipped.append(video_id)
                continue

        scene_path = output_dir / f"{video_id}_scene.jsonl"

        summary = assembler.assemble_video_to_jsonl(
            video_id=video_id,
            output_path=scene_path,
            detection_path=det_path if det_path.exists() else None,
            segmentation_path=seg_path if seg_path.exists() else None,
            phase_path=phase_path if phase_path.exists() else None,
        )
        all_summaries.append(summary)

    total_elapsed = time.perf_counter() - t0
    total_frames = sum(s["total_frames"] for s in all_summaries)
    total_complete = sum(s["frames_complete"] for s in all_summaries)

    # ── write run summary ─────────────────────────────────────────────
    run_summary = {
        "detection_dir": str(det_dir),
        "segmentation_dir": str(seg_dir),
        "phase_dir": str(phase_dir),
        "num_videos_processed": len(all_summaries),
        "num_videos_skipped": len(skipped),
        "skipped_videos": skipped,
        "total_frames": total_frames,
        "total_complete_frames": total_complete,
        "overall_completeness_pct": (
            round(total_complete / total_frames * 100, 1)
            if total_frames > 0 else 0
        ),
        "total_elapsed_sec": round(total_elapsed, 2),
        "per_video": all_summaries,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))

    log.info("=" * 60)
    log.info("Scene assembly complete.")
    log.info("  Processed     : %d video(s)", len(all_summaries))
    if skipped:
        log.info("  Skipped       : %d video(s)", len(skipped))
    log.info("  Total frames  : %d", total_frames)
    log.info("  Complete (3/3): %d (%.1f%%)",
             total_complete, run_summary["overall_completeness_pct"])
    log.info("  Elapsed       : %.1fs", total_elapsed)
    log.info("  Summary       : %s", summary_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
