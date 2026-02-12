#!/usr/bin/env python3
"""Render an animated 3D trajectory video of the surgical workflow space.

Creates a video showing the surgery's path being traced through the
3D Semantic Surgical Workflow Space, with camera orbit and phase
colour coding.

Usage:
    python scripts/render_trajectory_video.py --video video49
    python scripts/render_trajectory_video.py --video video49 --duration 30 --fps 30
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.phase_space_3d import build_semantic_phase_space

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Phase colours (RGB for matplotlib)
PHASE_COLORS_RGB = [
    (0.67, 0.67, 0.67),   # 0 Preparation – silver
    (1.00, 0.78, 0.20),   # 1 CalotTriangleDissection – amber
    (0.94, 0.24, 0.24),   # 2 ClippingCutting – red
    (0.31, 0.86, 0.31),   # 3 GallbladderDissection – green
    (0.20, 0.71, 1.00),   # 4 GallbladderPackaging – sky blue
    (0.86, 0.31, 0.86),   # 5 CleaningCoagulation – magenta
    (0.90, 0.90, 0.20),   # 6 GallbladderRetraction – yellow
]

PHASE_NAMES = [
    "Preparation", "CalotTriangleDissection", "ClippingCutting",
    "GallbladderDissection", "GallbladderPackaging",
    "CleaningCoagulation", "GallbladderRetraction",
]

# HUD accent colour
ACCENT_RGB = (0.0, 0.81, 0.82)  # teal


def render_trajectory_video(
    space: dict,
    output_path: Path,
    duration: float = 30.0,
    fps: int = 30,
    resolution: tuple[int, int] = (1280, 720),
    downsample: int = 50,
) -> dict:
    """Render the animated 3D trajectory video.

    Parameters
    ----------
    space : dict
        Output from build_semantic_phase_space().
    output_path : Path
        Output MP4 path.
    duration : float
        Video duration in seconds.
    fps : int
        Frames per second.
    resolution : tuple
        (width, height) of output video.
    downsample : int
        Sample every Nth point from the space data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    w, h = resolution
    dpi = 100
    fig_w, fig_h = w / dpi, h / dpi

    # Extract and downsample data
    x_full = space["phase_progress"]
    y_full = space["phase_idx"].astype(float)
    z_full = space["activity"]
    phases = space["phase_idx"]

    # Downsample
    idx = np.arange(0, len(x_full), downsample)
    x = x_full[idx]
    y = y_full[idx]
    z = z_full[idx]
    p = phases[idx]
    n_points = len(x)

    # Point colours
    colors = np.array([PHASE_COLORS_RGB[pi % len(PHASE_COLORS_RGB)] for pi in p])

    total_frames = int(duration * fps)
    log.info(
        "Rendering trajectory: %d space points, %d video frames (%.1fs @ %dfps)",
        n_points, total_frames, duration, fps,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    t0 = time.perf_counter()

    for frame_i in range(total_frames):
        progress = frame_i / max(total_frames - 1, 1)

        # How many points to show (ease-in curve for build-up)
        # First 70% of video: building up points
        # Last 30%: all points shown, camera orbits
        if progress < 0.70:
            build_progress = progress / 0.70
            # Ease-in-out
            t = build_progress
            show_n = max(1, int(n_points * (t * t * (3 - 2 * t))))
        else:
            show_n = n_points

        # Camera orbit
        elev = 20 + 10 * np.sin(progress * np.pi)
        azim = -60 + 360 * progress

        # Create figure
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#0a1120")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0a1120")

        # Style axes
        ax.set_xlabel("Phase Progression", color="#888", fontsize=8, labelpad=6)
        ax.set_ylabel("Phase Identity", color="#888", fontsize=8, labelpad=6)
        ax.set_zlabel("Activity", color="#888", fontsize=8, labelpad=6)

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, 6.5)
        ax.set_zlim(0, max(1.0, z.max() * 1.1))

        # Tick colours
        ax.tick_params(colors="#555", labelsize=6)
        for spine in ax.xaxis.get_ticklines():
            spine.set_color("#555")

        # Grid
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#222")
        ax.yaxis.pane.set_edgecolor("#222")
        ax.zaxis.pane.set_edgecolor("#222")
        ax.grid(True, alpha=0.15, color="#444")

        # Phase labels on Y axis
        ax.set_yticks(range(7))
        ax.set_yticklabels(
            ["PREP", "CALOT", "CLIP", "DISSECT", "PKG", "CLEAN", "RETRACT"],
            fontsize=5, color="#888",
        )

        # Plot trail points (faded)
        if show_n > 1:
            # Faded history
            trail_alpha = np.linspace(0.15, 0.7, show_n)
            ax.scatter(
                x[:show_n], y[:show_n], z[:show_n],
                c=colors[:show_n],
                s=8,
                alpha=0.5,
                depthshade=True,
            )

        # Trajectory line
        if show_n > 1:
            ax.plot(
                x[:show_n], y[:show_n], z[:show_n],
                color=(*ACCENT_RGB, 0.3),
                linewidth=0.8,
            )

        # Current point (bright, larger)
        if show_n > 0:
            ci = show_n - 1
            ax.scatter(
                [x[ci]], [y[ci]], [z[ci]],
                c=[colors[ci]],
                s=60,
                alpha=1.0,
                edgecolors="white",
                linewidth=1.5,
                zorder=10,
            )

        # Camera
        ax.view_init(elev=elev, azim=azim)

        # Title
        video_id = space.get("video_id", "")
        fig.text(
            0.05, 0.95, f"OpenSurgAI",
            fontsize=14, color=ACCENT_RGB, fontweight="bold",
            fontfamily="monospace", va="top",
        )
        fig.text(
            0.05, 0.91, f"3D Semantic Surgical Workflow Space  |  {video_id}",
            fontsize=8, color="#888", fontfamily="monospace", va="top",
        )

        # Point counter
        fig.text(
            0.95, 0.95,
            f"{show_n:,} / {n_points:,} frames",
            fontsize=8, color="#666", fontfamily="monospace",
            ha="right", va="top",
        )

        # Phase legend (bottom right)
        for i, (pname, pcolor) in enumerate(zip(
            ["PREP", "CALOT", "CLIP", "DISSECT", "PKG", "CLEAN", "RETRACT"],
            PHASE_COLORS_RGB,
        )):
            fig.text(
                0.92, 0.30 - i * 0.035, f"  {pname}",
                fontsize=6, color=pcolor, fontfamily="monospace",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#0a1120", edgecolor=pcolor, alpha=0.6, linewidth=0.5),
            )

        plt.tight_layout(pad=1.5)

        # Render to numpy array
        fig.canvas.draw()
        # Modern matplotlib: buffer_rgba() instead of tostring_rgb()
        img_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_rgba = img_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # RGBA -> BGR for OpenCV (drop alpha channel)
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

        # Resize if needed
        if img_bgr.shape[1] != w or img_bgr.shape[0] != h:
            img_bgr = cv2.resize(img_bgr, (w, h))

        writer.write(img_bgr)
        plt.close(fig)

        if (frame_i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            fps_actual = (frame_i + 1) / elapsed
            log.info("  %d/%d frames (%.1f FPS)", frame_i + 1, total_frames, fps_actual)

    writer.release()
    elapsed = time.perf_counter() - t0

    summary = {
        "video_id": space.get("video_id", ""),
        "output": str(output_path),
        "space_points": n_points,
        "video_frames": total_frames,
        "duration_sec": duration,
        "elapsed_sec": round(elapsed, 1),
        "render_fps": round(total_frames / elapsed, 1),
    }
    log.info("Trajectory video saved: %s (%.1fs render)", output_path, elapsed)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Render animated 3D trajectory video")
    parser.add_argument("--video", type=str, required=True, help="Video ID (e.g. video49)")
    parser.add_argument("--scene-dir", type=Path, default=_PROJECT_ROOT / "experiments" / "scene")
    parser.add_argument("--output-dir", type=Path, default=_PROJECT_ROOT / "experiments" / "analysis")
    parser.add_argument("--duration", type=float, default=30.0, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--downsample", type=int, default=50, help="Sample every Nth frame")
    args = parser.parse_args()

    scene_path = args.scene_dir / f"{args.video}_scene.jsonl"
    if not scene_path.exists():
        log.error("Scene file not found: %s", scene_path)
        sys.exit(1)

    log.info("Building phase space from %s", scene_path)
    space = build_semantic_phase_space(scene_path)

    output_path = args.output_dir / f"{args.video}_trajectory.mp4"
    summary = render_trajectory_video(
        space, output_path,
        duration=args.duration,
        fps=args.fps,
        downsample=args.downsample,
    )
    log.info("Done: %s", summary)


if __name__ == "__main__":
    main()
