#!/usr/bin/env python3
"""Validate OpenSurgAI setup and dependencies.

Checks all required components for GTC 2026 demo:
- Python dependencies
- CUDA availability
- TensorRT models
- EndoGaussian installation
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
        "plyfile": "PLYFile (for 3D viewer)",
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
            check_info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            check_warning(
                "CUDA NOT available",
                "GPU acceleration disabled. Demo will run slower."
            )
    except ImportError:
        check_fail("PyTorch not installed", "Cannot check CUDA")
        all_checks_passed = False

    # Check nvcc
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
            if version_line:
                check_pass(f"nvcc compiler found ({version_line[0].strip()})")
        else:
            check_warning("nvcc compiler not found", "Needed for EndoGaussian CUDA extensions")
    except FileNotFoundError:
        check_warning("nvcc compiler not found", "Needed for EndoGaussian CUDA extensions")

    # ══════════════════════════════════════════════════════════════
    # 4. TensorRT Models
    # ══════════════════════════════════════════════════════════════
    print_header("4. TensorRT Models")

    models_dir = project_root / "models"
    trt_model = models_dir / "cholec80_resnet50_trt_fp16.engine"

    if trt_model.exists():
        size_mb = trt_model.stat().st_size / 1e6
        check_pass(f"TensorRT model found ({size_mb:.1f} MB)")
    else:
        check_fail(
            "TensorRT model NOT found",
            "Run: python scripts/export_tensorrt.py"
        )
        all_checks_passed = False

    # Check PyTorch fallback
    pt_model = models_dir / "cholec80_resnet50.pth"
    if pt_model.exists():
        check_pass("PyTorch model found (fallback available)")
    else:
        check_warning("PyTorch model NOT found", "No fallback if TensorRT fails")

    # ══════════════════════════════════════════════════════════════
    # 5. EndoGaussian Installation
    # ══════════════════════════════════════════════════════════════
    print_header("5. EndoGaussian 3D Reconstruction")

    endogaussian_dir = project_root / "external" / "EndoGaussian"

    if endogaussian_dir.exists():
        check_pass("EndoGaussian repository cloned")

        # Check for submodules
        diff_gauss = endogaussian_dir / "submodules" / "diff-gaussian-rasterization"
        simple_knn = endogaussian_dir / "submodules" / "simple-knn"

        if diff_gauss.exists() and simple_knn.exists():
            check_pass("Submodules initialized")
        else:
            check_warning(
                "Submodules NOT initialized",
                "cd external/EndoGaussian && git submodule update --init --recursive"
            )

        # Check for conda environment
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=False
        )
        if "endogaussian" in result.stdout:
            check_pass("Conda environment 'endogaussian' exists")
        else:
            check_warning(
                "Conda environment NOT created",
                "Run: python scripts/setup_endogaussian.py --yes"
            )
    else:
        check_fail(
            "EndoGaussian NOT installed",
            "Run: python scripts/setup_endogaussian.py --yes"
        )

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
            for video in videos[:3]:  # Show first 3
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
    scenes_dir = project_root / "experiments" / "scenes"
    if scenes_dir.exists():
        scenes = list(scenes_dir.glob("*_scene.jsonl"))
        if scenes:
            check_pass(f"Found {len(scenes)} processed scene file(s)")
        else:
            check_info("No scene files yet (will be created by detection pipeline)")
    else:
        check_info("Scenes directory will be created on first run")

    # ══════════════════════════════════════════════════════════════
    # 7. API Keys
    # ══════════════════════════════════════════════════════════════
    print_header("7. API Keys (Optional)")

    import os

    nemotron_key = os.getenv("NEMOTRON_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if nemotron_key:
        check_pass("Nemotron API key configured")
    else:
        check_info("Nemotron API key not set (Q&A features will be limited)")
        check_info("   Set NEMOTRON_API_KEY or NVIDIA_API_KEY environment variable")

    # ══════════════════════════════════════════════════════════════
    # 8. Dashboard Configuration
    # ══════════════════════════════════════════════════════════════
    print_header("8. Dashboard Configuration")

    streamlit_config = project_root / ".streamlit" / "config.toml"
    if streamlit_config.exists():
        check_pass("Streamlit config found")
    else:
        check_info("Streamlit config not found (will use defaults)")

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    print_header("Validation Summary")

    if all_checks_passed:
        print(f"""
{Colors.GREEN}{Colors.BOLD}
    [+] ALL CRITICAL CHECKS PASSED!
{Colors.RESET}
    Your OpenSurgAI setup is ready for the GTC 2026 demo!

    Next steps:
    1. Run full demo: python scripts/run_full_demo.py --video video49
    2. Or launch dashboard: streamlit run scripts/app_dashboard.py

    Good luck with your submission!
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
    - Install EndoGaussian: scripts\setup_endogaussian.bat (Windows)
    - Export TensorRT model: python scripts/export_tensorrt.py
    - Download Cholec80 videos to data/cholec80/videos/
        """)
        return 1


if __name__ == "__main__":
    sys.exit(main())
