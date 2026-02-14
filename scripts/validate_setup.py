#!/usr/bin/env python3
"""Validate OpenSurgAI setup and dependencies.

Checks all required components for GTC 2026 demo:
- Python dependencies
- CUDA availability
- TensorRT models
- Data files
- API keys

Run this before attempting the full demo!
"""

import importlib
import subprocess
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def check_pass(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}[+]{Colors.RESET} {message}")


def check_fail(message: str, hint: str = ""):
    """Print failure message with optional hint."""
    print(f"{Colors.RED}[X]{Colors.RESET} {message}")
    if hint:
        print(f"    {Colors.YELLOW}Hint: {hint}{Colors.RESET}")


def check_warning(message: str, hint: str = ""):
    """Print warning message."""
    print(f"{Colors.YELLOW}[!]{Colors.RESET} {message}")
    if hint:
        print(f"    {Colors.YELLOW}Hint: {hint}{Colors.RESET}")


def check_info(message: str):
    """Print info message."""
    print(f"{Colors.BLUE}[i]{Colors.RESET} {message}")


def main():
    print(f"""
{Colors.BOLD}{Colors.CYAN}
    ================================================
      OpenSurgAI Setup Validation
      Multi-NIM Surgical Intelligence Platform
    ================================================
{Colors.RESET}
    Checking all dependencies for GTC 2026 demo...
    """)

    project_root = Path(__file__).parent.parent
    all_checks_passed = True

    # ══════════════════════════════════════════════════════════════
    # 1. Python Environment
    # ══════════════════════════════════════════════════════════════
    print_header("1. Python Environment")

    python_version = sys.version_info
    if python_version >= (3, 9):
        check_pass(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        check_fail(
            f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
            "Python 3.9+ recommended"
        )
        all_checks_passed = False

    # ══════════════════════════════════════════════════════════════
    # 2. Core Dependencies
    # ══════════════════════════════════════════════════════════════
    print_header("2. Core Python Dependencies")

    required_packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "numpy": "NumPy",
        "cv2": "OpenCV",
        "streamlit": "Streamlit",
        "plotly": "Plotly",
        "tqdm": "TQDM",
        "httpx": "HTTPX (NIM API client)",
    }

    for module_name, display_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            check_pass(f"{display_name} installed")
        except ImportError:
            check_fail(
                f"{display_name} NOT installed",
                f"pip install {module_name if module_name != 'cv2' else 'opencv-python'}"
            )
            all_checks_passed = False

    # ══════════════════════════════════════════════════════════════
    # 3. CUDA and GPU
    # ══════════════════════════════════════════════════════════════
    print_header("3. CUDA and GPU")

    try:
        import torch
        if torch.cuda.is_available():
            check_pass(f"CUDA available (version {torch.version.cuda})")
            check_info(f"   GPU: {torch.cuda.get_device_name(0)}")
            check_info(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        else:
            check_warning(
                "CUDA NOT available",
                "GPU acceleration disabled. Demo will run slower."
            )
    except ImportError:
        check_fail("PyTorch not installed", "Cannot check CUDA")
        all_checks_passed = False

    # ══════════════════════════════════════════════════════════════
    # 4. TensorRT Models
    # ══════════════════════════════════════════════════════════════
    print_header("4. TensorRT Models")

    weights_dir = project_root / "weights"
    trt_dir = weights_dir / "tensorrt"

    # Check for tool classifier weights
    tool_weights = weights_dir / "tool_resnet50.pt"
    if tool_weights.exists():
        size_mb = tool_weights.stat().st_size / 1e6
        check_pass(f"Tool classifier weights found ({size_mb:.1f} MB)")
    else:
        check_warning("Tool classifier weights not found", "Place tool_resnet50.pt in weights/")

    # Check for TensorRT compiled model
    tool_trt = trt_dir / "tool_resnet50_trt.ts"
    if tool_trt.exists():
        size_mb = tool_trt.stat().st_size / 1e6
        check_pass(f"TensorRT FP16 model found ({size_mb:.1f} MB)")
    else:
        check_info("TensorRT compiled model not found (will use PyTorch fallback)")

    # Phase recognition weights
    phase_weights = weights_dir / "phase_resnet50.pt"
    if phase_weights.exists():
        check_pass("Phase recognition weights found")
    else:
        check_warning("Phase recognition weights not found")

    # ══════════════════════════════════════════════════════════════
    # 5. NIM API Services
    # ══════════════════════════════════════════════════════════════
    print_header("5. NVIDIA NIM API Services")

    import os

    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NEMOTRON_API_KEY")
    if api_key:
        check_pass("NVIDIA API key configured")
        check_info("   Nemotron 49B: Ready (text reasoning)")
        check_info("   Nemotron VL: Ready (visual analysis)")
    else:
        check_warning(
            "NVIDIA API key not set",
            "Set NVIDIA_API_KEY env var for Nemotron NIM services"
        )

    # Check VLM client can be imported
    try:
        from src.explanation.vlm_client import VLMClient
        check_pass("VLMClient module available")
    except ImportError as e:
        check_fail(f"VLMClient import failed: {e}")

    # Check frame extractor
    try:
        from src.explanation.frame_extractor import extract_frame_at_time
        check_pass("Frame extractor module available")
    except ImportError as e:
        check_fail(f"Frame extractor import failed: {e}")

    # ══════════════════════════════════════════════════════════════
    # 6. Data Files
    # ══════════════════════════════════════════════════════════════
    print_header("6. Data Files")

    # Check for video files
    videos_dir = project_root / "data" / "cholec80" / "videos"
    if videos_dir.exists():
        videos = list(videos_dir.glob("*.mp4"))
        if videos:
            check_pass(f"Found {len(videos)} video file(s)")
            for video in videos[:3]:
                check_info(f"   - {video.name}")
            if len(videos) > 3:
                check_info(f"   ... and {len(videos) - 3} more")
        else:
            check_warning("No video files found in data/cholec80/videos/")
    else:
        check_warning(
            "Videos directory not found",
            "Download Cholec80 videos and place in data/cholec80/videos/"
        )

    # Check for scene data
    scenes_dir = project_root / "experiments" / "scene"
    if scenes_dir.exists():
        scenes = list(scenes_dir.glob("*_scene.jsonl"))
        if scenes:
            check_pass(f"Found {len(scenes)} processed scene file(s)")
        else:
            check_info("No scene files yet (will be created by detection pipeline)")
    else:
        check_info("Scene directory will be created on first run")

    # ══════════════════════════════════════════════════════════════
    # 7. Dashboard Configuration
    # ══════════════════════════════════════════════════════════════
    print_header("7. Dashboard Configuration")

    streamlit_config = project_root / ".streamlit" / "config.toml"
    if streamlit_config.exists():
        check_pass("Streamlit config found")
    else:
        check_info("Streamlit config not found (will use defaults)")

    # Check dashboard can be imported
    dashboard_file = project_root / "scripts" / "app_dashboard.py"
    if dashboard_file.exists():
        check_pass("Dashboard script found")
    else:
        check_fail("Dashboard script NOT found")
        all_checks_passed = False

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print_header("Validation Summary")

    if all_checks_passed:
        print(f"""
{Colors.GREEN}{Colors.BOLD}
    [+] ALL CRITICAL CHECKS PASSED!
{Colors.RESET}
    Your OpenSurgAI Multi-NIM platform is ready for GTC 2026!

    Next steps:
    1. Launch dashboard: streamlit run scripts/app_dashboard.py
    2. Or full demo: python scripts/run_full_demo.py --video video49
        """)
        return 0
    else:
        print(f"""
{Colors.RED}{Colors.BOLD}
    [X] SOME CHECKS FAILED
{Colors.RESET}
    Please fix the issues above before running the demo.

    Common fixes:
    - Install dependencies: pip install -r requirements.txt
    - Set API key: export NVIDIA_API_KEY=nvapi-...
    - Download Cholec80 videos to data/cholec80/videos/
        """)
        return 1


if __name__ == "__main__":
    sys.exit(main())
