#!/usr/bin/env python3
"""Prepare Cholec80 surgical video for EndoGaussian training.

Converts Cholec80 video format to EndoGaussian's expected structure.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def extract_frames(video_path: Path, output_dir: Path, fps: int = 1, max_frames: int = None):
    """Extract frames from video at specified FPS.

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        fps: Frames per second to extract (1 = 1 frame per second)
        max_frames: Maximum number of frames to extract
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(video_fps / fps)

    print(f"📹 Video: {video_path.name}")
    print(f"   FPS: {video_fps:.2f}")
    print(f"   Total frames: {total_frames:,}")
    print(f"   Extracting every {frame_interval} frames ({fps} FPS)")

    frame_idx = 0
    extracted = 0
    pbar = tqdm(total=min(max_frames, total_frames // frame_interval) if max_frames else total_frames // frame_interval)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at specified interval
        if frame_idx % frame_interval == 0:
            # Save frame
            frame_path = output_dir / f"frame_{extracted:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted += 1
            pbar.update(1)

            if max_frames and extracted >= max_frames:
                break

        frame_idx += 1

    cap.release()
    pbar.close()

    print(f"✅ Extracted {extracted} frames to {output_dir}")
    return extracted


def estimate_camera_params(frames_dir: Path, output_path: Path):
    """Estimate camera parameters for EndoGaussian.

    For surgical videos, we'll use a simplified camera model since
    laparoscopic cameras have relatively stable intrinsics.

    Args:
        frames_dir: Directory containing extracted frames
        output_path: Path to save camera parameters JSON
    """
    # Get first frame to determine resolution
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        raise ValueError(f"No frames found in {frames_dir}")

    first_frame = cv2.imread(str(frames[0]))
    height, width = first_frame.shape[:2]

    print(f"📐 Image resolution: {width}x{height}")

    # Estimate camera intrinsics for laparoscopic camera
    # Typical laparoscopic FOV is 65-90 degrees
    # We'll use a conservative 75-degree FOV
    fov_degrees = 75
    focal_length = (width / 2) / np.tan(np.radians(fov_degrees / 2))

    # Camera intrinsic matrix
    K = [
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ]

    # Create camera parameters for each frame
    # For surgical video, we assume camera is relatively static
    cameras = []
    for i, frame_path in enumerate(frames):
        # Identity rotation (camera looking straight ahead)
        R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # Small translation to simulate camera movement
        # (In reality, surgical camera moves slightly during procedure)
        t = [0.0, 0.0, 0.1 * (i / len(frames))]  # Slight forward motion

        cameras.append({
            "id": i,
            "img_name": frame_path.name,
            "width": width,
            "height": height,
            "position": t,  # Camera position (translation)
            "rotation": R,  # Camera rotation matrix
            "fx": focal_length,
            "fy": focal_length,
            "cx": width / 2,
            "cy": height / 2
        })

    # Save camera parameters
    camera_data = {
        "camera_model": "PINHOLE",
        "cameras": cameras
    }

    with open(output_path, 'w') as f:
        json.dump(camera_data, f, indent=2)

    print(f"✅ Saved camera parameters to {output_path}")
    print(f"   Intrinsics: f={focal_length:.2f}, FOV={fov_degrees}°")


def create_config(video_id: str, output_dir: Path):
    """Create training configuration for EndoGaussian.

    Args:
        video_id: Video identifier (e.g., 'video49')
        output_dir: Directory containing prepared data
    """
    config = {
        "name": f"cholec80_{video_id}",
        "data_dir": str(output_dir),
        "images": "images",
        "cameras": "cameras.json",

        # Training parameters (optimized for surgical video)
        "iterations": 7000,  # EndoGaussian default
        "resolution": -1,  # Use full resolution
        "white_background": False,  # Surgical video has dark background

        # Deformation parameters (for moving tissue)
        "deform": True,
        "is_blender": False,
        "is_6dof": False,

        # Output
        "model_path": f"output/{video_id}",
        "test_iterations": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        "save_iterations": [7000],
    }

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Created training config: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Cholec80 video for EndoGaussian 3D reconstruction"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Video ID (e.g., video49)"
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/cholec80/videos"),
        help="Directory containing Cholec80 videos"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: external/EndoGaussian/data/{video_id})"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frames per second to extract (default: 1)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum frames to extract (default: 300, ~5 min at 1 FPS)"
    )

    args = parser.parse_args()

    print("""
    🎬 CHOLEC80 → ENDOGAUSSIAN DATA PREPARATION
    ============================================
    """)

    # Locate video file
    video_path = args.video_dir / f"{args.video}.mp4"
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        print(f"   Looking for alternatives...")
        # Try other extensions
        for ext in ['.avi', '.mkv', '.mov']:
            alt_path = video_path.with_suffix(ext)
            if alt_path.exists():
                video_path = alt_path
                print(f"✅ Found: {video_path}")
                break
        else:
            raise FileNotFoundError(f"Cannot find video: {args.video}")

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = Path("external/EndoGaussian/data") / args.video

    print(f"📁 Output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    print("\n🎞️  STEP 1: Extracting frames...")
    images_dir = args.output_dir / "images"
    num_frames = extract_frames(
        video_path,
        images_dir,
        fps=args.fps,
        max_frames=args.max_frames
    )

    # Estimate camera parameters
    print("\n📷 STEP 2: Estimating camera parameters...")
    cameras_path = args.output_dir / "cameras.json"
    estimate_camera_params(images_dir, cameras_path)

    # Create config
    print("\n⚙️  STEP 3: Creating training configuration...")
    create_config(args.video, args.output_dir)

    print(f"""

    ✅✅✅ DATA PREPARATION COMPLETE! ✅✅✅
    =======================================

    📊 Summary:
    ───────────
    Video:        {args.video}
    Frames:       {num_frames}
    Resolution:   Detected from frames
    Output:       {args.output_dir}

    🚀 Next Step: TRAINING!
    ────────────────────────

    Run this command to train EndoGaussian (takes ~2 minutes):

        cd external/EndoGaussian
        conda activate endogaussian
        python train.py -s ../../{args.output_dir.relative_to(Path.cwd()).parent} -m {args.video}

    After training, render the 3D scene:

        python render.py -m {args.video}

    LET'S GO! 🔥
    """)


if __name__ == "__main__":
    main()
