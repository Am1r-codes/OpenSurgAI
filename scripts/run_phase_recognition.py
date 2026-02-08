#!/usr/bin/env python3
"""Run surgical phase recognition on Cholec80 videos.

Classifies each video frame into one of the 7 Cholec80 surgical phases,
applies temporal smoothing, and saves per-frame predictions as JSON
Lines under ``experiments/phase/``.  Optionally evaluates against
ground-truth annotations and escalates if accuracy is insufficient.

Usage examples:

    # Single video (integration-test mode — no fine-tuned weights)
    python scripts/run_phase_recognition.py \
        --video data/cholec80/videos/video01.mp4

    # All videos with fine-tuned checkpoint + evaluation
    python scripts/run_phase_recognition.py --all --half \
        --model-weights weights/phase_resnet50.pt --evaluate

    # Adjust temporal smoothing
    python scripts/run_phase_recognition.py --all --half \
        --model-weights weights/phase_resnet50.pt \
        --smooth-window 51 --min-duration 500

    # Test split, every 5th frame, raise threshold
    python scripts/run_phase_recognition.py --all --split test \
        --stride 5 --half --evaluate --accuracy-threshold 0.80
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

from src.phase.pipeline import PhaseRecognitionPipeline  # noqa: E402

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
    return project_root() / "experiments" / "phase"


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
        description="Run surgical phase recognition on Cholec80 videos.",
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
        "--model-weights", type=str, default=None,
        help="Path to a fine-tuned .pt checkpoint (default: ImageNet + random head)",
    )
    p.add_argument("--device", type=str, default=None, help="PyTorch device (default: auto)")
    p.add_argument("--half", action="store_true", help="Use FP16 inference")
    p.add_argument("--img-size", type=int, default=224, help="Input crop size (default: 224)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--stride", type=int, default=1, help="Frame stride (default: 1 = every frame)")

    # ── temporal smoothing ────────────────────────────────────────────
    p.add_argument(
        "--smooth-window", type=int, default=25,
        help="Majority-vote sliding window size in frames (default: 25, 0=off)",
    )
    p.add_argument(
        "--min-duration", type=int, default=250,
        help="Minimum phase segment duration in frames (default: 250, 0=off)",
    )

    # ── evaluation / escalation ───────────────────────────────────────
    p.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate against ground-truth phase annotations",
    )
    p.add_argument(
        "--accuracy-threshold", type=float, default=0.70,
        help="Accuracy threshold below which to escalate (default: 0.70)",
    )

    # ── output ────────────────────────────────────────────────────────
    p.add_argument(
        "--output-dir", type=Path, default=default_output_dir(),
        help="Output directory for JSONL files (default: experiments/phase/)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    videos = collect_videos(args)

    log.info("Videos to process: %d", len(videos))
    log.info("Weights  : %s", args.model_weights or "none (integration-test)")
    log.info("Device   : %s", args.device or "auto")
    log.info("Smoothing: window=%d, min_duration=%d", args.smooth_window, args.min_duration)
    log.info("Evaluate : %s (threshold=%.0f%%)", args.evaluate, args.accuracy_threshold * 100)
    log.info("Output   : %s", args.output_dir)

    pipeline = PhaseRecognitionPipeline(
        model_weights=args.model_weights,
        device=args.device,
        half=args.half,
        img_size=args.img_size,
        batch_size=args.batch_size,
        smooth_window=args.smooth_window,
        min_phase_duration=args.min_duration,
        accuracy_threshold=args.accuracy_threshold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    all_summaries: list[dict] = []
    escalated_videos: list[str] = []
    t0 = time.perf_counter()

    for video_path in videos:
        video_id = video_path.stem
        jsonl_path = output_dir / f"{video_id}_phases.jsonl"

        annotation_path = None
        if args.evaluate:
            annotation_path = data_dir / "phase_annotations" / f"{video_id}-phase.txt"

        summary = pipeline.process_video_to_jsonl(
            video_path=video_path,
            output_path=jsonl_path,
            video_id=video_id,
            stride=args.stride,
            annotation_path=annotation_path,
        )
        all_summaries.append(summary)
        if summary.get("escalate"):
            escalated_videos.append(video_id)

    total_elapsed = time.perf_counter() - t0
    total_frames = sum(s["total_frames"] for s in all_summaries)

    # ── aggregate evaluation ──────────────────────────────────────────
    eval_summaries = [s for s in all_summaries if "evaluation" in s]
    if eval_summaries:
        total_evaluated = sum(s["evaluation"]["num_evaluated"] for s in eval_summaries)
        total_correct = sum(
            int(s["evaluation"]["accuracy"] * s["evaluation"]["num_evaluated"])
            for s in eval_summaries
        )
        overall_acc = total_correct / total_evaluated if total_evaluated > 0 else 0.0
    else:
        overall_acc = None

    # ── write run summary ─────────────────────────────────────────────
    run_summary: dict = {
        "model_weights": args.model_weights,
        "device": pipeline.device,
        "half": pipeline.half,
        "batch_size": pipeline.batch_size,
        "img_size": pipeline.img_size,
        "smooth_window": pipeline.smooth_window,
        "min_phase_duration": pipeline.min_phase_duration,
        "accuracy_threshold": pipeline.accuracy_threshold,
        "stride": args.stride,
        "num_videos": len(videos),
        "total_frames": total_frames,
        "total_elapsed_sec": round(total_elapsed, 2),
        "overall_fps": round(total_frames / total_elapsed, 1) if total_elapsed > 0 else 0,
        "per_video": all_summaries,
    }
    if overall_acc is not None:
        run_summary["overall_accuracy"] = round(overall_acc, 4)
        run_summary["escalated_videos"] = escalated_videos

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2))

    log.info("=" * 60)
    log.info("Run complete.")
    log.info("  Videos     : %d", len(videos))
    log.info("  Frames     : %d", total_frames)
    log.info("  Elapsed    : %.1fs", total_elapsed)
    log.info("  Overall FPS: %.1f", run_summary["overall_fps"])
    if overall_acc is not None:
        log.info("  Accuracy   : %.1f%%", overall_acc * 100)
        if escalated_videos:
            log.warning(
                "  ESCALATED  : %d video(s) below %.0f%% threshold: %s",
                len(escalated_videos), args.accuracy_threshold * 100,
                ", ".join(escalated_videos),
            )
    log.info("  Summary    : %s", summary_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
