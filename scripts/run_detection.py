#!/usr/bin/env python3
"""Run YOLOv8 instrument detection on Cholec80 videos.

Processes one or more surgical videos through the detection pipeline and
saves per-frame bounding-box results as JSON Lines files under
``experiments/detection/``.

Usage examples:

    # Single video
    python scripts/run_detection.py --video data/cholec80/videos/video01.mp4

    # All videos in the dataset
    python scripts/run_detection.py --all

    # Custom model with FP16 on batch size 32
    python scripts/run_detection.py --all --model weights/yolo_cholec.pt \
        --batch-size 32 --half

    # Process every 3rd frame only (for faster passes)
    python scripts/run_detection.py --all --stride 3
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

from src.detection.pipeline import DetectionPipeline  # noqa: E402

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


def default_output_dir() -> Path:
    return project_root() / "experiments" / "detection"


def collect_videos(args: argparse.Namespace) -> list[Path]:
    """Build the list of video paths to process."""
    if args.video:
        vp = Path(args.video)
        if not vp.exists():
            log.error("Video not found: %s", vp)
            sys.exit(1)
        return [vp]

    if args.all:
        videos_dir = Path(args.data_dir) / "videos"
        if not videos_dir.is_dir():
            log.error("Videos directory not found: %s", videos_dir)
            sys.exit(1)
        videos = sorted(videos_dir.glob("video*.*"))
        if not videos:
            log.error("No video files found in %s", videos_dir)
            sys.exit(1)
        # Optionally filter to a split
        if args.split:
            split_file = Path(args.data_dir) / "splits" / f"{args.split}.txt"
            if not split_file.exists():
                log.error("Split file not found: %s", split_file)
                sys.exit(1)
            allowed = set(split_file.read_text().strip().splitlines())
            videos = [v for v in videos if v.stem in allowed]
        return videos

    log.error("Specify --video PATH or --all")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run instrument detection on Cholec80 videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── input selection ───────────────────────────────────────────────
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--video", type=str, help="Path to a single video file")
    grp.add_argument("--all", action="store_true", help="Process all videos in --data-dir")

    p.add_argument(
        "--data-dir", type=Path, default=default_data_dir(),
        help="Cholec80 data directory (default: data/cholec80/)",
    )
    p.add_argument(
        "--split", type=str, choices=["train", "test"], default=None,
        help="Restrict --all to a specific split",
    )

    # ── model / inference ─────────────────────────────────────────────
    p.add_argument(
        "--model", type=str, default="yolov8s.pt",
        help="YOLO weights file or Ultralytics model name (default: yolov8s.pt)",
    )
    p.add_argument("--device", type=str, default=None, help="PyTorch device (default: auto)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    p.add_argument("--img-size", type=int, default=640, help="Input image size (default: 640)")
    p.add_argument("--half", action="store_true", help="Use FP16 inference")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    p.add_argument("--stride", type=int, default=1, help="Frame stride (default: 1 = every frame)")

    # ── output ────────────────────────────────────────────────────────
    p.add_argument(
        "--output-dir", type=Path, default=default_output_dir(),
        help="Output directory for JSONL files (default: experiments/detection/)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    videos = collect_videos(args)

    log.info("Videos to process: %d", len(videos))
    log.info("Model : %s", args.model)
    log.info("Device: %s", args.device or "auto")
    log.info("Output: %s", args.output_dir)

    pipeline = DetectionPipeline(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        img_size=args.img_size,
        half=args.half,
        batch_size=args.batch_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict] = []
    t0 = time.perf_counter()

    for video_path in videos:
        video_id = video_path.stem
        jsonl_path = output_dir / f"{video_id}_detections.jsonl"

        summary = pipeline.process_video_to_jsonl(
            video_path=video_path,
            output_path=jsonl_path,
            video_id=video_id,
            stride=args.stride,
        )
        all_summaries.append(summary)

    total_elapsed = time.perf_counter() - t0
    total_frames = sum(s["total_frames"] for s in all_summaries)
    total_dets = sum(s["total_detections"] for s in all_summaries)

    # ── write run summary ─────────────────────────────────────────────
    run_summary = {
        "model": args.model,
        "device": pipeline.device,
        "half": pipeline.half,
        "batch_size": pipeline.batch_size,
        "img_size": pipeline.img_size,
        "conf_threshold": pipeline.conf_threshold,
        "iou_threshold": pipeline.iou_threshold,
        "stride": args.stride,
        "num_videos": len(videos),
        "total_frames": total_frames,
        "total_detections": total_dets,
        "total_elapsed_sec": round(total_elapsed, 2),
        "overall_fps": round(total_frames / total_elapsed, 1) if total_elapsed > 0 else 0,
        "per_video": all_summaries,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))

    log.info("=" * 60)
    log.info("Run complete.")
    log.info("  Videos     : %d", len(videos))
    log.info("  Frames     : %d", total_frames)
    log.info("  Detections : %d", total_dets)
    log.info("  Elapsed    : %.1fs", total_elapsed)
    log.info("  Overall FPS: %.1f", run_summary["overall_fps"])
    log.info("  Summary    : %s", summary_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
