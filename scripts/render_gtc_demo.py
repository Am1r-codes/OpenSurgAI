#!/usr/bin/env python3
"""Render cinematic GTC 2026 demo video with all features showcased.

Creates a 60-90 second highlight reel showing:
1. Detection pipeline (TensorRT inference)
2. 3D reconstruction visualization
3. Nemotron AI reasoning
4. Dashboard interactivity

Perfect for GTC Golden Ticket submission!
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def create_title_card(text: str, width: int = 1920, height: int = 1080) -> np.ndarray:
    """Create a professional title card frame.

    Args:
        text: Title text to display
        width: Frame width
        height: Frame height

    Returns:
        Title card frame (BGR)
    """
    # Dark navy background matching dashboard
    frame = np.full((height, width, 3), (32, 17, 10), dtype=np.uint8)

    # Add gradient overlay
    gradient = np.linspace(0, 1, height)[:, np.newaxis]
    teal_overlay = np.array([209, 206, 0])  # Teal in BGR
    for i in range(height):
        alpha = gradient[i, 0] * 0.15
        frame[i] = frame[i] * (1 - alpha) + teal_overlay * alpha

    # Add title text
    font = cv2.FONT_HERSHEY_BOLD
    font_scale = 2.5
    thickness = 4

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Center text
    x = (width - text_width) // 2
    y = (height + text_height) // 2

    # Add glow effect (multiple layers)
    for offset in [8, 6, 4, 2]:
        cv2.putText(
            frame, text, (x, y),
            font, font_scale,
            (209, 206, 0),  # Teal glow
            thickness + offset,
            cv2.LINE_AA
        )

    # Main text
    cv2.putText(
        frame, text, (x, y),
        font, font_scale,
        (255, 255, 255),  # White
        thickness,
        cv2.LINE_AA
    )

    # Add subtitle
    subtitle = "NVIDIA GTC 2026 | AI-Powered Surgical Intelligence"
    font_scale_sub = 0.8
    thickness_sub = 2

    (sub_width, sub_height), _ = cv2.getTextSize(subtitle, font, font_scale_sub, thickness_sub)
    x_sub = (width - sub_width) // 2
    y_sub = y + 80

    cv2.putText(
        frame, subtitle, (x_sub, y_sub),
        font, font_scale_sub,
        (128, 128, 128),  # Gray
        thickness_sub,
        cv2.LINE_AA
    )

    return frame


def create_feature_card(
    title: str,
    metrics: list[tuple[str, str]],
    width: int = 1920,
    height: int = 1080
) -> np.ndarray:
    """Create a feature highlight card.

    Args:
        title: Feature title
        metrics: List of (metric_name, metric_value) tuples
        width: Frame width
        height: Frame height

    Returns:
        Feature card frame (BGR)
    """
    # Dark background
    frame = np.full((height, width, 3), (32, 17, 10), dtype=np.uint8)

    # Title
    font = cv2.FONT_HERSHEY_BOLD
    cv2.putText(
        frame, title,
        (100, 150),
        font, 2.0,
        (209, 206, 0),  # Teal
        3,
        cv2.LINE_AA
    )

    # Metrics
    y_offset = 300
    for metric_name, metric_value in metrics:
        # Metric name
        cv2.putText(
            frame, metric_name,
            (150, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            (200, 200, 200),
            2,
            cv2.LINE_AA
        )

        # Metric value (larger, teal)
        cv2.putText(
            frame, metric_value,
            (150, y_offset + 60),
            font, 1.8,
            (209, 206, 0),
            3,
            cv2.LINE_AA
        )

        y_offset += 150

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Render cinematic GTC 2026 demo video"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="video49",
        help="Video ID to use for demo"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (default: experiments/gtc_demo.mp4)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video FPS (default: 30)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=90,
        help="Target duration in seconds (default: 90)"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    video_id = args.video

    if args.output is None:
        args.output = project_root / "experiments" / "gtc_demo.mp4"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"""
    ==================================================
      GTC 2026 DEMO VIDEO RENDERER
    ==================================================

    Creating cinematic highlight reel...

    Video: {video_id}
    Output: {args.output}
    Duration: {args.duration}s @ {args.fps} FPS
    """)

    # Video writer setup
    width, height = 1920, 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(args.output), fourcc, args.fps, (width, height))

    total_frames = args.duration * args.fps

    # ══════════════════════════════════════════════════════════════
    # Scene 1: Opening Title (3 seconds)
    # ══════════════════════════════════════════════════════════════
    print("\n[1/6] Rendering opening title...")
    title_frame = create_title_card("OpenSurgAI")

    for _ in tqdm(range(3 * args.fps), desc="Title"):
        out.write(title_frame)

    # ══════════════════════════════════════════════════════════════
    # Scene 2: TensorRT Feature Card (5 seconds)
    # ══════════════════════════════════════════════════════════════
    print("\n[2/6] Rendering TensorRT feature card...")
    tensorrt_card = create_feature_card(
        "TensorRT Acceleration",
        [
            ("Inference Speed", "1,300 FPS"),
            ("Model", "Cholec80 Tool Classifier"),
            ("Precision", "FP16 Optimized"),
            ("Speedup", "26x vs PyTorch")
        ]
    )

    for _ in tqdm(range(5 * args.fps), desc="TensorRT"):
        out.write(tensorrt_card)

    # ══════════════════════════════════════════════════════════════
    # Scene 3: Annotated Video Clip (30 seconds)
    # ══════════════════════════════════════════════════════════════
    print("\n[3/6] Adding annotated video clip...")

    # Load annotated video with HUD overlay
    overlay_video = project_root / "experiments" / "dashboard" / f"{video_id}_demo.mp4"

    if overlay_video.exists():
        cap = cv2.VideoCapture(str(overlay_video))
        frames_to_take = 30 * args.fps
        frame_count = 0

        pbar = tqdm(total=frames_to_take, desc="Video")

        while frame_count < frames_to_take:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop
                continue

            # Resize to 1920x1080 if needed
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))

            out.write(frame)
            frame_count += 1
            pbar.update(1)

        cap.release()
        pbar.close()
    else:
        print(f"    [!] Overlay video not found: {overlay_video}")
        print("    [*] Using placeholder...")
        placeholder = create_title_card("Run render_demo_video.py first")
        for _ in tqdm(range(30 * args.fps), desc="Placeholder"):
            out.write(placeholder)

    # ══════════════════════════════════════════════════════════════
    # Scene 4: 3D Reconstruction Card (10 seconds)
    # ══════════════════════════════════════════════════════════════
    print("\n[4/6] Rendering 3D reconstruction feature...")
    gaussian_card = create_feature_card(
        "EndoGaussian 3D Reconstruction",
        [
            ("Rendering Speed", "195 FPS Real-time"),
            ("Training Time", "2 Minutes"),
            ("Quality", "Photorealistic (37.8 PSNR)"),
            ("Features", "Interactive Click-to-Explore")
        ]
    )

    for _ in tqdm(range(10 * args.fps), desc="3D Features"):
        out.write(gaussian_card)

    # ══════════════════════════════════════════════════════════════
    # Scene 5: Nemotron AI Card (7 seconds)
    # ══════════════════════════════════════════════════════════════
    print("\n[5/6] Rendering Nemotron AI feature...")
    nemotron_card = create_feature_card(
        "Nemotron 70B Reasoning",
        [
            ("Model", "NVIDIA Nemotron-70B"),
            ("Capability", "Surgical Context Understanding"),
            ("Features", "Q&A + Explanations"),
            ("Integration", "Real-time Phase Analysis")
        ]
    )

    for _ in tqdm(range(7 * args.fps), desc="Nemotron"):
        out.write(nemotron_card)

    # ══════════════════════════════════════════════════════════════
    # Scene 6: Closing Title (5 seconds)
    # ══════════════════════════════════════════════════════════════
    print("\n[6/6] Rendering closing title...")
    closing_frame = create_title_card("Powered by NVIDIA")

    for _ in tqdm(range(5 * args.fps), desc="Closing"):
        out.write(closing_frame)

    # ══════════════════════════════════════════════════════════════
    # Finalize
    # ══════════════════════════════════════════════════════════════
    out.release()

    print(f"""

    ===================================
      DEMO VIDEO COMPLETE!
    ===================================

    Output: {args.output}
    Duration: {args.duration}s
    Resolution: 1920x1080 @ {args.fps} FPS

    Ready for GTC 2026 submission!

    Next steps:
    1. Review the video
    2. Upload to platform
    3. Submit with #NVIDIAGTC
    """)


if __name__ == "__main__":
    main()
