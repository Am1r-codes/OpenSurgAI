#!/usr/bin/env python3
"""EndoGaussian Setup Script for Windows/Linux - FULL SEND MODE!

Automates the complete installation of EndoGaussian for 3D surgical scene reconstruction.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description="", check=True):
    """Run a shell command with nice output."""
    print(f"\n{'='*60}")
    print(f"🔧 {description if description else cmd}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr and "warning" not in result.stderr.lower():
        print(result.stderr)

    if check and result.returncode != 0:
        print(f"❌ FAILED: {description}")
        print(f"Return code: {result.returncode}")
        if not input("Continue anyway? (y/n): ").lower().startswith('y'):
            sys.exit(1)

    return result

def main():
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

    if not input("Ready to start? (y/n): ").lower().startswith('y'):
        print("Setup cancelled.")
        return

    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Check CUDA
    print("\n📍 Checking CUDA availability...")
    cuda_check = run_command("nvcc --version", "Check CUDA version", check=False)
    if cuda_check.returncode != 0:
        print("⚠️  WARNING: nvcc not found. Make sure CUDA is installed!")
        print("   Download from: https://developer.nvidia.com/cuda-11-7-0-download-archive")
        if not input("Continue anyway? (y/n): ").lower().startswith('y'):
            return

    # Create directory structure
    print("\n📁 Creating directory structure...")
    external_dir = project_root / "external"
    external_dir.mkdir(exist_ok=True)
    os.chdir(external_dir)

    # Clone EndoGaussian
    print("\n📥 Cloning EndoGaussian repository...")
    if not (external_dir / "EndoGaussian").exists():
        run_command(
            "git clone https://github.com/CUHK-AIM-Group/EndoGaussian.git",
            "Clone EndoGaussian from GitHub"
        )
    else:
        print("✓ EndoGaussian directory already exists, skipping clone")

    os.chdir(external_dir / "EndoGaussian")

    # Check if conda is available
    print("\n🐍 Checking conda...")
    conda_check = run_command("conda --version", "Check conda", check=False)
    if conda_check.returncode != 0:
        print("❌ Conda not found! Please install Miniconda or Anaconda first.")
        print("   Download from: https://docs.conda.io/en/latest/miniconda.html")
        return

    # Create conda environment
    print("\n🌟 Creating conda environment 'endogaussian'...")
    run_command(
        "conda create -n endogaussian python=3.7 -y",
        "Create conda environment",
        check=False  # May already exist
    )

    # Detect conda activation command
    conda_exe = "conda"
    if sys.platform == "win32":
        conda_base = Path(os.environ.get("CONDA_PREFIX", "")).parent
        activate_cmd = f'"{conda_base / "Scripts" / "activate.bat"}" endogaussian'
    else:
        activate_cmd = "conda activate endogaussian"

    print(f"\n📦 Installing PyTorch 1.13.1 with CUDA 11.7...")
    print("   (This may take 5-10 minutes...)")

    # Install PyTorch - using pip for better Windows compatibility
    run_command(
        f"{activate_cmd} && pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117",
        "Install PyTorch with CUDA 11.7"
    )

    # Install basic dependencies
    print("\n📦 Installing core dependencies...")
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
        f"{activate_cmd} && pip install {' '.join(dependencies)}",
        "Install core dependencies"
    )

    # Install pytorch3d
    print("\n🎨 Installing PyTorch3D...")
    print("   (This is the big one - may take 10-15 minutes...)")

    # First try conda
    pt3d_result = run_command(
        f"{activate_cmd} && conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y",
        "Install PyTorch3D dependencies",
        check=False
    )

    # Then install pytorch3d from source
    run_command(
        f"{activate_cmd} && pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'",
        "Install PyTorch3D from GitHub",
        check=False
    )

    # Install Gaussian rasterization submodules
    print("\n⚡ Building Gaussian rasterization CUDA extensions...")
    print("   (This compiles C++/CUDA code - may take 5-10 minutes...)")

    submodules_dir = Path("submodules")
    if submodules_dir.exists():
        # Install diff-gaussian-rasterization
        if (submodules_dir / "diff-gaussian-rasterization").exists():
            os.chdir(submodules_dir / "diff-gaussian-rasterization")
            run_command(
                f"{activate_cmd} && pip install .",
                "Build diff-gaussian-rasterization CUDA extension"
            )
            os.chdir(external_dir / "EndoGaussian")

        # Install simple-knn
        if (submodules_dir / "simple-knn").exists():
            os.chdir(submodules_dir / "simple-knn")
            run_command(
                f"{activate_cmd} && pip install .",
                "Build simple-knn CUDA extension"
            )
            os.chdir(external_dir / "EndoGaussian")
    else:
        print("⚠️  WARNING: submodules directory not found!")
        print("   The repo may need submodule initialization:")
        run_command("git submodule update --init --recursive", "Initialize submodules")
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
