#!/usr/bin/env python3
"""Cholec80 dataset preparation script.

Downloads are NOT automated — the dataset is gated and requires manual
approval from CAMMA (https://camma.unistra.fr/datasets/).

This script:
  1. Locates downloaded archives in  data/cholec80/raw/
  2. Extracts them
  3. Organises videos, phase annotations, and tool annotations into a
     clean directory layout under  data/cholec80/
  4. Extracts frames at 1 fps using ffmpeg (optional, requires ffmpeg)
  5. Generates train/test split files (standard 40/40 split)

Usage:
    python scripts/prepare_cholec80.py [--raw-dir RAW] [--data-dir DATA]
                                       [--extract-frames] [--fps 1]
                                       [--num-workers 4]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Cholec80 constants ─────────────────────────────────────────────────
NUM_VIDEOS = 80
PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]
TOOLS = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]
# Standard 40-train / 40-test split (videos 01–40 train, 41–80 test)
TRAIN_IDS = list(range(1, 41))
TEST_IDS = list(range(41, 81))


def project_root() -> Path:
    """Return the project root (parent of ``scripts/``)."""
    return Path(__file__).resolve().parent.parent


def default_data_dir() -> Path:
    return project_root() / "data" / "cholec80"


def default_raw_dir() -> Path:
    return default_data_dir() / "raw"


# ── Archive extraction ──────────────────────────────────────────────────

def extract_archives(raw_dir: Path, data_dir: Path) -> None:
    """Extract any .zip archives found in *raw_dir*."""
    archives = sorted(raw_dir.glob("*.zip"))
    if not archives:
        log.warning("No .zip archives found in %s", raw_dir)
        return
    for archive in archives:
        log.info("Extracting %s ...", archive.name)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(data_dir / "_extracted")
    log.info("Extraction complete.")


# ── File organisation ───────────────────────────────────────────────────

VIDEO_RE = re.compile(r"video(\d{2})\.(mp4|avi|mkv)", re.IGNORECASE)
PHASE_RE = re.compile(r"video(\d{2})-phase\.txt", re.IGNORECASE)
TOOL_RE = re.compile(r"video(\d{2})-tool\.txt", re.IGNORECASE)


def _collect_files(search_root: Path) -> dict:
    """Walk *search_root* and bucket files by type."""
    found: dict[str, dict[int, Path]] = {
        "videos": {},
        "phases": {},
        "tools": {},
    }
    for path in search_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        m = VIDEO_RE.match(name)
        if m:
            found["videos"][int(m.group(1))] = path
            continue
        m = PHASE_RE.match(name)
        if m:
            found["phases"][int(m.group(1))] = path
            continue
        m = TOOL_RE.match(name)
        if m:
            found["tools"][int(m.group(1))] = path
    return found


def organise(data_dir: Path) -> None:
    """Move / copy discovered files into the canonical layout."""
    # Search in _extracted first, fall back to raw
    search_dirs = [data_dir / "_extracted", data_dir / "raw"]
    found: dict[str, dict[int, Path]] = {"videos": {}, "phases": {}, "tools": {}}
    for sd in search_dirs:
        if sd.is_dir():
            partial = _collect_files(sd)
            for k in found:
                for vid_id, p in partial[k].items():
                    found[k].setdefault(vid_id, p)

    # Also scan data_dir itself (user may have placed files directly)
    partial = _collect_files(data_dir)
    for k in found:
        for vid_id, p in partial[k].items():
            found[k].setdefault(vid_id, p)

    videos_dir = data_dir / "videos"
    phases_dir = data_dir / "phase_annotations"
    tools_dir = data_dir / "tool_annotations"
    videos_dir.mkdir(exist_ok=True)
    phases_dir.mkdir(exist_ok=True)
    tools_dir.mkdir(exist_ok=True)

    # Copy videos
    for vid_id, src in sorted(found["videos"].items()):
        ext = src.suffix
        dst = videos_dir / f"video{vid_id:02d}{ext}"
        if not dst.exists():
            log.info("  video%02d%s -> videos/", vid_id, ext)
            shutil.copy2(src, dst)

    # Copy phase annotations
    for vid_id, src in sorted(found["phases"].items()):
        dst = phases_dir / f"video{vid_id:02d}-phase.txt"
        if not dst.exists():
            log.info("  video%02d-phase.txt -> phase_annotations/", vid_id)
            shutil.copy2(src, dst)

    # Copy tool annotations
    for vid_id, src in sorted(found["tools"].items()):
        dst = tools_dir / f"video{vid_id:02d}-tool.txt"
        if not dst.exists():
            log.info("  video%02d-tool.txt -> tool_annotations/", vid_id)
            shutil.copy2(src, dst)

    log.info(
        "Organised %d videos, %d phase annotations, %d tool annotations.",
        len(found["videos"]),
        len(found["phases"]),
        len(found["tools"]),
    )


# ── Frame extraction ────────────────────────────────────────────────────

def _extract_frames_single(
    video_path: Path, out_dir: Path, fps: int
) -> tuple[int, int]:
    """Extract frames from a single video at *fps* using ffmpeg.

    Returns (video_id, num_frames_extracted).
    """
    vid_id = int(re.search(r"video(\d{2})", video_path.stem).group(1))  # type: ignore[union-attr]
    frame_dir = out_dir / f"video{vid_id:02d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(frame_dir / f"video{vid_id:02d}_%06d.png")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
        "-hide_banner", "-loglevel", "error", "-y",
    ]
    subprocess.run(cmd, check=True)
    n_frames = len(list(frame_dir.glob("*.png")))
    return vid_id, n_frames


def extract_frames(
    data_dir: Path, fps: int = 1, num_workers: int = 4
) -> None:
    """Extract frames from all videos in ``data_dir/videos/``."""
    videos_dir = data_dir / "videos"
    frames_dir = data_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    video_files = sorted(videos_dir.glob("video*.*"))
    if not video_files:
        log.warning("No video files found in %s", videos_dir)
        return

    if shutil.which("ffmpeg") is None:
        log.error(
            "ffmpeg is not installed or not on PATH. "
            "Install it to extract frames: https://ffmpeg.org/download.html"
        )
        sys.exit(1)

    log.info(
        "Extracting frames at %d fps from %d videos (workers=%d) ...",
        fps, len(video_files), num_workers,
    )

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_extract_frames_single, vp, frames_dir, fps): vp
            for vp in video_files
        }
        for future in as_completed(futures):
            vid_id, n = future.result()
            log.info("  video%02d: %d frames", vid_id, n)

    log.info("Frame extraction complete.")


# ── Train / test split ──────────────────────────────────────────────────

def write_splits(data_dir: Path) -> None:
    """Write the standard 40/40 train-test split files."""
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    train_path = splits_dir / "train.txt"
    test_path = splits_dir / "test.txt"

    with open(train_path, "w") as f:
        for vid_id in TRAIN_IDS:
            f.write(f"video{vid_id:02d}\n")

    with open(test_path, "w") as f:
        for vid_id in TEST_IDS:
            f.write(f"video{vid_id:02d}\n")

    log.info(
        "Wrote splits: %d train, %d test  ->  %s",
        len(TRAIN_IDS), len(TEST_IDS), splits_dir,
    )


# ── CLI entry point ────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare the Cholec80 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw_dir(),
        help="Directory containing downloaded Cholec80 archives (default: data/cholec80/raw/)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Root output directory (default: data/cholec80/)",
    )
    p.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract video frames at --fps using ffmpeg",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second for extraction (default: 1)",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Parallel workers for frame extraction (default: 4)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir.resolve()
    data_dir: Path = args.data_dir.resolve()

    log.info("Raw dir : %s", raw_dir)
    log.info("Data dir: %s", data_dir)

    if not raw_dir.is_dir():
        log.error(
            "Raw directory does not exist: %s\n"
            "Please download the Cholec80 dataset from "
            "https://camma.unistra.fr/datasets/ and place the "
            "archive(s) in that directory.",
            raw_dir,
        )
        sys.exit(1)

    # Step 1 – extract
    extract_archives(raw_dir, data_dir)

    # Step 2 – organise
    organise(data_dir)

    # Step 3 – splits
    write_splits(data_dir)

    # Step 4 – frames (optional)
    if args.extract_frames:
        extract_frames(data_dir, fps=args.fps, num_workers=args.num_workers)

    log.info("Done. Run `python scripts/verify_cholec80.py` for a sanity check.")


if __name__ == "__main__":
    main()
