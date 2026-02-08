#!/usr/bin/env python3
"""Cholec80 dataset integrity verification and sanity-check.

Checks:
  1. Expected directory structure exists
  2. All 80 videos are present
  3. All 80 phase annotation files are present and well-formed
  4. All 80 tool annotation files are present and well-formed
  5. Train/test split files match the standard 40/40 split
  6. (Optional) Frames directory spot-check: at least one frame per video
  7. Sanity-check sample: prints a small preview of annotations for
     a randomly selected video

Usage:
    python scripts/verify_cholec80.py [--data-dir DATA] [--check-frames]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# ── Cholec80 constants ─────────────────────────────────────────────────
NUM_VIDEOS = 80
VALID_PHASES = {
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
}
NUM_TOOLS = 7
TRAIN_IDS = set(range(1, 41))
TEST_IDS = set(range(41, 81))


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_data_dir() -> Path:
    return project_root() / "data" / "cholec80"


# ── Helpers ─────────────────────────────────────────────────────────────

class Checker:
    """Accumulates pass/warn/fail counts and messages."""

    def __init__(self) -> None:
        self.passed = 0
        self.warnings = 0
        self.failures = 0

    def ok(self, msg: str) -> None:
        self.passed += 1
        print(f"  [PASS] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings += 1
        print(f"  [WARN] {msg}")

    def fail(self, msg: str) -> None:
        self.failures += 1
        print(f"  [FAIL] {msg}")

    def summary(self) -> int:
        total = self.passed + self.warnings + self.failures
        print(f"\n{'='*60}")
        print(
            f"Results: {self.passed}/{total} passed, "
            f"{self.warnings} warnings, {self.failures} failures"
        )
        if self.failures:
            print("STATUS: FAILED")
            return 1
        if self.warnings:
            print("STATUS: PASSED with warnings")
            return 0
        print("STATUS: ALL CHECKS PASSED")
        return 0


# ── Individual checks ──────────────────────────────────────────────────

def check_directories(data_dir: Path, c: Checker) -> None:
    print("\n--- Directory structure ---")
    for subdir in ["videos", "phase_annotations", "tool_annotations", "splits"]:
        p = data_dir / subdir
        if p.is_dir():
            c.ok(f"{subdir}/ exists")
        else:
            c.fail(f"{subdir}/ missing")


def check_videos(data_dir: Path, c: Checker) -> None:
    print("\n--- Videos ---")
    videos_dir = data_dir / "videos"
    if not videos_dir.is_dir():
        c.fail("videos/ directory not found, skipping video checks")
        return

    found = set()
    for f in videos_dir.iterdir():
        if f.is_file() and f.stem.startswith("video"):
            try:
                vid_id = int(f.stem.replace("video", ""))
                found.add(vid_id)
            except ValueError:
                pass

    if len(found) == NUM_VIDEOS:
        c.ok(f"All {NUM_VIDEOS} videos present")
    else:
        missing = set(range(1, NUM_VIDEOS + 1)) - found
        if missing:
            c.fail(
                f"Found {len(found)}/{NUM_VIDEOS} videos. "
                f"Missing: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
            )
        else:
            c.warn(f"Found {len(found)} videos (expected {NUM_VIDEOS})")


def check_phase_annotations(data_dir: Path, c: Checker) -> None:
    print("\n--- Phase annotations ---")
    ann_dir = data_dir / "phase_annotations"
    if not ann_dir.is_dir():
        c.fail("phase_annotations/ directory not found")
        return

    found = sorted(ann_dir.glob("video*-phase.txt"))
    if len(found) == NUM_VIDEOS:
        c.ok(f"All {NUM_VIDEOS} phase annotation files present")
    else:
        c.fail(f"Found {len(found)}/{NUM_VIDEOS} phase annotation files")

    # Validate content of each file
    bad_files = []
    for fpath in found:
        try:
            lines = fpath.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                bad_files.append((fpath.name, "empty file"))
                continue
            # Check a sample of lines for valid phase labels
            for line in lines[1:min(len(lines), 20)]:  # skip possible header
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    parts = line.strip().split()
                if len(parts) >= 2:
                    phase = parts[-1]
                    if phase not in VALID_PHASES:
                        bad_files.append((fpath.name, f"unknown phase '{phase}'"))
                        break
        except Exception as e:
            bad_files.append((fpath.name, str(e)))

    if not bad_files:
        c.ok("Phase annotation content looks valid (sample check)")
    else:
        for name, reason in bad_files[:5]:
            c.warn(f"{name}: {reason}")


def check_tool_annotations(data_dir: Path, c: Checker) -> None:
    print("\n--- Tool annotations ---")
    ann_dir = data_dir / "tool_annotations"
    if not ann_dir.is_dir():
        c.fail("tool_annotations/ directory not found")
        return

    found = sorted(ann_dir.glob("video*-tool.txt"))
    if len(found) == NUM_VIDEOS:
        c.ok(f"All {NUM_VIDEOS} tool annotation files present")
    else:
        c.fail(f"Found {len(found)}/{NUM_VIDEOS} tool annotation files")

    # Validate: each row should have frame + 7 binary values
    bad_files = []
    for fpath in found:
        try:
            lines = fpath.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                bad_files.append((fpath.name, "empty file"))
                continue
            for line in lines[1:min(len(lines), 20)]:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    parts = line.strip().split()
                # Expect frame_number + 7 tool indicators
                tool_vals = parts[1:] if len(parts) > 1 else parts
                if len(tool_vals) < NUM_TOOLS:
                    bad_files.append(
                        (fpath.name, f"expected {NUM_TOOLS} tool columns, got {len(tool_vals)}")
                    )
                    break
                for v in tool_vals[:NUM_TOOLS]:
                    if v not in ("0", "1"):
                        bad_files.append((fpath.name, f"non-binary value '{v}'"))
                        break
        except Exception as e:
            bad_files.append((fpath.name, str(e)))

    if not bad_files:
        c.ok("Tool annotation content looks valid (sample check)")
    else:
        for name, reason in bad_files[:5]:
            c.warn(f"{name}: {reason}")


def check_splits(data_dir: Path, c: Checker) -> None:
    print("\n--- Train/test splits ---")
    splits_dir = data_dir / "splits"
    if not splits_dir.is_dir():
        c.fail("splits/ directory not found")
        return

    for split_name, expected_ids in [("train", TRAIN_IDS), ("test", TEST_IDS)]:
        fpath = splits_dir / f"{split_name}.txt"
        if not fpath.exists():
            c.fail(f"{split_name}.txt missing")
            continue
        lines = fpath.read_text().strip().splitlines()
        ids = set()
        for line in lines:
            try:
                ids.add(int(line.strip().replace("video", "")))
            except ValueError:
                pass
        if ids == expected_ids:
            c.ok(f"{split_name}.txt has correct {len(expected_ids)} video IDs")
        else:
            c.fail(
                f"{split_name}.txt mismatch: got {len(ids)} IDs, expected {len(expected_ids)}"
            )


def check_frames(data_dir: Path, c: Checker) -> None:
    print("\n--- Frames (spot-check) ---")
    frames_dir = data_dir / "frames"
    if not frames_dir.is_dir():
        c.warn("frames/ directory not found (run prepare with --extract-frames)")
        return

    dirs = sorted(d for d in frames_dir.iterdir() if d.is_dir())
    if not dirs:
        c.warn("No frame subdirectories found")
        return

    empty_dirs = []
    for d in dirs:
        pngs = list(d.glob("*.png"))
        if not pngs:
            empty_dirs.append(d.name)

    if not empty_dirs:
        c.ok(f"All {len(dirs)} frame directories contain PNG files")
    else:
        c.warn(f"{len(empty_dirs)} frame directories are empty: {empty_dirs[:5]}")


# ── Sanity-check sample ────────────────────────────────────────────────

def sanity_check_sample(data_dir: Path) -> None:
    """Print a small preview of annotations for one randomly selected video."""
    print("\n--- Sanity-check sample ---")
    phase_dir = data_dir / "phase_annotations"
    tool_dir = data_dir / "tool_annotations"

    phase_files = sorted(phase_dir.glob("video*-phase.txt")) if phase_dir.is_dir() else []
    if not phase_files:
        print("  No phase annotations available for sample preview.")
        return

    sample_file = random.choice(phase_files)
    vid_name = sample_file.stem.replace("-phase", "")
    print(f"  Randomly selected: {vid_name}")

    # Phase preview
    lines = sample_file.read_text(encoding="utf-8").strip().splitlines()
    print(f"\n  Phase annotations ({len(lines)} lines total):")
    print(f"    Header / first line: {lines[0]}")
    for line in lines[1:6]:
        print(f"    {line}")
    print(f"    ... ({len(lines) - 6} more lines)")

    # Tool preview
    tool_file = tool_dir / f"{vid_name}-tool.txt"
    if tool_file.exists():
        tlines = tool_file.read_text(encoding="utf-8").strip().splitlines()
        print(f"\n  Tool annotations ({len(tlines)} lines total):")
        print(f"    Header / first line: {tlines[0]}")
        for line in tlines[1:6]:
            print(f"    {line}")
        print(f"    ... ({len(tlines) - 6} more lines)")

    # Frame count
    frame_dir = data_dir / "frames" / vid_name
    if frame_dir.is_dir():
        n_frames = len(list(frame_dir.glob("*.png")))
        print(f"\n  Extracted frames: {n_frames}")
    else:
        print(f"\n  Frames not yet extracted for {vid_name}.")


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify Cholec80 dataset integrity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Cholec80 data root (default: data/cholec80/)",
    )
    p.add_argument(
        "--check-frames",
        action="store_true",
        help="Also verify extracted frames",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir: Path = args.data_dir.resolve()
    print(f"Verifying Cholec80 dataset at: {data_dir}\n")

    c = Checker()

    check_directories(data_dir, c)
    check_videos(data_dir, c)
    check_phase_annotations(data_dir, c)
    check_tool_annotations(data_dir, c)
    check_splits(data_dir, c)

    if args.check_frames:
        check_frames(data_dir, c)

    sanity_check_sample(data_dir)

    rc = c.summary()
    sys.exit(rc)


if __name__ == "__main__":
    main()
