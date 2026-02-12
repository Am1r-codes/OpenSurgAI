#!/usr/bin/env python3
"""OpenSurgAI - Complete GTC 2026 Demo Pipeline.

One-command automation for the full OpenSurgAI workflow:
1. Run detection pipeline (TensorRT tool classification)
2. Train EndoGaussian 3D reconstruction
3. Generate Nemotron analysis
4. Render demo video with HUD overlay
5. Launch interactive dashboard

Perfect for GTC 2026 Golden Ticket submission!
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def run_step(cmd: str, description: str, timeout: int = 600) -> bool:
    """Run a pipeline step and return success status.

    Args:
        cmd: Command to execute
        description: Human-readable description
        timeout: Maximum execution time in seconds

    Returns:
        True if successful, False otherwise
    """
    print(f"[*] {description}...")
    print(f"    Command: {cmd}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"[+] SUCCESS ({duration:.1f}s)")
            if result.stdout:
                print(f"    Output: {result.stdout[:200]}...")
            return True
        else:
            print(f"[!] FAILED (return code: {result.returncode})")
            if result.stderr:
                print(f"    Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"[!] TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"[!] ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run complete OpenSurgAI demo pipeline for GTC 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline on video49
  python scripts/run_full_demo.py --video video49

  # Skip 3D training (if already trained)
  python scripts/run_full_demo.py --video video49 --skip-3d

  # Only run dashboard (everything else done)
  python scripts/run_full_demo.py --video video49 --dashboard-only
        """
    )

    parser.add_argument(
        "--video",
        type=str,
        default="video49",
        help="Video ID to process (default: video49)"
    )

    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip detection pipeline (use existing results)"
    )

    parser.add_argument(
        "--skip-3d",
        action="store_true",
        help="Skip 3D reconstruction training"
    )

    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Skip HUD overlay video rendering"
    )

    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Only launch dashboard (skip all processing)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit dashboard port (default: 8501)"
    )

    args = parser.parse_args()

    # Project paths
    project_root = Path(__file__).parent.parent
    video_id = args.video

    print("""
    ================================================
      OpenSurgAI - GTC 2026 GOLDEN TICKET DEMO
    ================================================

    Full NVIDIA-powered surgical AI pipeline:
    - TensorRT 1,300 FPS tool classification
    - Nemotron 70B surgical reasoning
    - EndoGaussian 3D reconstruction (195 FPS)
    - Interactive dashboard with HUD overlay

    Video: {video}
    """.format(video=video_id))

    # Pipeline steps tracking
    steps_completed = 0
    steps_total = 4

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Run Detection Pipeline (TensorRT + Nemotron)
    # ══════════════════════════════════════════════════════════════
    if not args.dashboard_only and not args.skip_detection:
        print_section("STEP 1/4: Running Detection Pipeline")

        cmd = f"python scripts/run_detection.py --video {video_id}"
        if run_step(cmd, "TensorRT tool classification + phase detection", timeout=1200):
            steps_completed += 1
            print("\n[+] Detection complete! Scene data saved.")
        else:
            print("\n[!] Detection failed! Check TensorRT model and video file.")
            if not input("Continue anyway? (y/n): ").lower().startswith('y'):
                sys.exit(1)
    else:
        print_section("STEP 1/4: Skipping Detection Pipeline")
        steps_completed += 1

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Train EndoGaussian 3D Reconstruction
    # ══════════════════════════════════════════════════════════════
    if not args.dashboard_only and not args.skip_3d:
        print_section("STEP 2/4: 3D Reconstruction Training")

        # Check if data is prepared
        data_dir = project_root / "external" / "EndoGaussian" / "data" / video_id
        if not data_dir.exists():
            print(f"[*] Preparing data for EndoGaussian...")
            prep_cmd = f"python scripts/prepare_cholec80_for_gaussian.py --video {video_id}"
            if not run_step(prep_cmd, "Extract frames and estimate camera parameters", timeout=600):
                print("[!] Data preparation failed!")
                if not input("Continue anyway? (y/n): ").lower().startswith('y'):
                    sys.exit(1)

        # Train model (takes ~2 minutes!)
        print(f"\n[*] Training EndoGaussian (this will be FAST - only 2 minutes!)...")
        endogaussian_dir = project_root / "external" / "EndoGaussian"

        if endogaussian_dir.exists():
            train_cmd = f'cd "{endogaussian_dir}" && conda activate endogaussian && python train.py -s ../../data/{video_id} -m {video_id}'
            if run_step(train_cmd, "Train Gaussian Splatting model", timeout=300):
                steps_completed += 1
                print("\n[+] 3D reconstruction trained!")
            else:
                print("\n[!] Training failed! Check EndoGaussian installation.")
                if not input("Continue anyway? (y/n): ").lower().startswith('y'):
                    sys.exit(1)
        else:
            print("[!] EndoGaussian not installed! Run: python scripts/setup_endogaussian.py")
            steps_completed += 1  # Skip gracefully
    else:
        print_section("STEP 2/4: Skipping 3D Reconstruction")
        steps_completed += 1

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Render HUD Overlay Video
    # ══════════════════════════════════════════════════════════════
    if not args.dashboard_only and not args.skip_render:
        print_section("STEP 3/4: Rendering HUD Overlay Video")

        cmd = f"python scripts/render_demo_video.py --video {video_id}"
        if run_step(cmd, "Render professional surgical HUD overlay", timeout=1800):
            steps_completed += 1
            print("\n[+] Demo video rendered!")
        else:
            print("\n[!] Rendering failed!")
            if not input("Continue anyway? (y/n): ").lower().startswith('y'):
                sys.exit(1)
    else:
        print_section("STEP 3/4: Skipping HUD Rendering")
        steps_completed += 1

    # ══════════════════════════════════════════════════════════════
    # STEP 4: Launch Interactive Dashboard
    # ══════════════════════════════════════════════════════════════
    print_section("STEP 4/4: Launching Interactive Dashboard")

    print(f"""
    [+] Pipeline Complete! ({steps_completed}/{steps_total} steps)

    Starting Streamlit dashboard...

    Dashboard Features:
    -------------------
    - Overview: Video playback with HUD overlay
    - 3D Scene: Interactive EndoGaussian reconstruction
    - AI Analysis: Nemotron-powered surgical Q&A
    - Compare: Multi-surgery phase analysis

    Access: http://localhost:{args.port}
    """)

    # Launch dashboard
    dashboard_cmd = f"streamlit run scripts/app_dashboard.py --server.port {args.port}"

    print(f"[*] Running: {dashboard_cmd}\n")
    print("="*70)
    print("Press Ctrl+C to stop the dashboard")
    print("="*70)

    try:
        subprocess.run(dashboard_cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print("\n\n[*] Dashboard stopped. Demo complete!")
    except Exception as e:
        print(f"\n[!] Dashboard error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
