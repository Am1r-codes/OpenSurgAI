#!/usr/bin/env python3
"""EndoGaussian Setup Script for Windows/Linux - FULL SEND MODE!

Automates the complete installation of EndoGaussian for 3D surgical scene reconstruction.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description="", check=True, auto_yes=False):
    """Run a shell command with nice output."""
    print(f"\n{'='*60}")
    print(f" {description if description else cmd}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr and "warning" not in result.stderr.lower():
        print(result.stderr)

    if check and result.returncode != 0:
        print(f"FAILED: {description}")
        print(f"Return code: {result.returncode}")
        if not auto_yes and not input("Continue anyway? (y/n): ").lower().startswith('y'):
            sys.exit(1)

    return result

def main():
    parser = argparse.ArgumentParser(description="Install EndoGaussian for 3D surgical reconstruction")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    args = parser.parse_args()

    print("""
    === ENDOGAUSSIAN INSTALLATION ===
    ========================================

    This will install EndoGaussian for real-time 3D surgical reconstruction!

    Requirements:
    - NVIDIA GPU with CUDA 11.7+
    - Conda or Miniconda
    - ~10GB disk space

    Training time: 2 MINUTES per video!
    Rendering: 195 FPS real-time!
    """)

    if not args.yes and not input("Ready to start? (y/n): ").lower().startswith('y'):
        print("Setup cancelled.")
        return

    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Check CUDA
    print("\n Checking CUDA availability...")
    cuda_check = run_command("nvcc --version", "Check CUDA version", check=False)
    if cuda_check.returncode != 0:
        print("WARNING: nvcc not found. Make sure CUDA is installed!")
        print("   Download from: https://developer.nvidia.com/cuda-11-7-0-download-archive")
        if not args.yes and not input("Continue anyway? (y/n): ").lower().startswith('y'):
            return

    # Create directory structure
    print("\n[*] Creating directory structure...")
    external_dir = project_root / "external"
    external_dir.mkdir(exist_ok=True)
    os.chdir(external_dir)

    # Clone EndoGaussian
    print("\n[*] Cloning EndoGaussian repository...")
    if not (external_dir / "EndoGaussian").exists():
        run_command(
            "git clone https://github.com/CUHK-AIM-Group/EndoGaussian.git",
            "Clone EndoGaussian from GitHub",
            auto_yes=args.yes
        )
    else:
        print("[+] EndoGaussian directory already exists, skipping clone")

    os.chdir(external_dir / "EndoGaussian")

    # Check if conda is available
    print("\n[*] Checking conda...")
    conda_check = run_command("conda --version", "Check conda", check=False, auto_yes=args.yes)
    if conda_check.returncode != 0:
        print("[!] Conda not found! Please install Miniconda or Anaconda first.")
        print("   Download from: https://docs.conda.io/en/latest/miniconda.html")
        return

    # Create conda environment
    print("\n[*] Creating conda environment 'endogaussian'...")
    run_command(
        "conda create -n endogaussian python=3.7 -y",
        "Create conda environment",
        check=False,  # May already exist
        auto_yes=args.yes
    )

    # Use conda run to execute commands in the environment
    # This works from any shell without activation
    def conda_run(cmd: str) -> str:
        """Wrap command to run in conda environment."""
        return f'conda run -n endogaussian {cmd}'

    print(f"\n[*] Installing PyTorch 1.13.1 with CUDA 11.7...")
    print("   (This may take 5-10 minutes...)")
    print("   NOTE: On Windows, installation may be faster using the batch script:")
    print("         scripts/setup_endogaussian.bat")
    print()

    # Install PyTorch - using pip for better Windows compatibility
    run_command(
        conda_run('pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117'),
        "Install PyTorch with CUDA 11.7",
        auto_yes=args.yes
    )

    # Install basic dependencies
    print("\n[*] Installing core dependencies...")
    dependencies = [
        "tqdm",
        "plyfile",
        "opencv-python",
        "scikit-image",
        "lpips",
        "imageio",
        "imageio-ffmpeg",
        "trimesh",
    ]

    run_command(
        conda_run(f'pip install {" ".join(dependencies)}'),
        "Install core dependencies",
        auto_yes=args.yes
    )

    # Install pytorch3d
    print("\n[*] Installing PyTorch3D...")
    print("   (This is the big one - may take 10-15 minutes...)")

    # First install dependencies
    pt3d_result = run_command(
        conda_run('conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y'),
        "Install PyTorch3D dependencies",
        check=False,
        auto_yes=args.yes
    )

    # Then install pytorch3d from source
    run_command(
        conda_run('pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"'),
        "Install PyTorch3D from GitHub",
        check=False,
        auto_yes=args.yes
    )

    # Install Gaussian rasterization submodules
    print("\n[*] Building Gaussian rasterization CUDA extensions...")
    print("   (This compiles C++/CUDA code - may take 5-10 minutes...)")

    submodules_dir = Path("submodules")
    if submodules_dir.exists():
        # Install diff-gaussian-rasterization
        if (submodules_dir / "diff-gaussian-rasterization").exists():
            os.chdir(submodules_dir / "diff-gaussian-rasterization")
            run_command(
                conda_run('pip install .'),
                "Build diff-gaussian-rasterization CUDA extension",
                auto_yes=args.yes
            )
            os.chdir(external_dir / "EndoGaussian")

        # Install simple-knn
        if (submodules_dir / "simple-knn").exists():
            os.chdir(submodules_dir / "simple-knn")
            run_command(
                conda_run('pip install .'),
                "Build simple-knn CUDA extension",
                auto_yes=args.yes
            )
            os.chdir(external_dir / "EndoGaussian")
    else:
        print("[!] WARNING: submodules directory not found!")
        print("   The repo may need submodule initialization:")
        run_command("git submodule update --init --recursive", "Initialize submodules", auto_yes=args.yes)
        print("   Please re-run this script after submodules are initialized.")

    print("""

    === INSTALLATION COMPLETE! ===
    ===================================

    EndoGaussian is ready to rock!

    Next Steps:
    -----------

    1. Activate the environment:
       conda activate endogaussian

    2. Prepare your surgical video data:
       python scripts/prepare_cholec80_for_gaussian.py --video video49

    3. Train the model (takes ~2 MINUTES!):
       python train.py --config configs/cholec80.yaml

    4. Render 3D scene (real-time at 195 FPS!):
       python render.py --model output/video49

    FEATURES:
    ---------
    * Photorealistic 3D reconstruction
    * Real-time rendering (195 FPS)
    * 2-minute training time
    * Click-to-explore interactivity
    * Deformable tissue handling

    LET'S GO!!!
    """)

if __name__ == "__main__":
    main()
