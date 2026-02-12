#!/bin/bash
# EndoGaussian Setup Script for OpenSurgAI
# Full send mode - automated installation

set -e  # Exit on error

echo "🔥 ENDOGAUSSIAN INSTALLATION - FULL SEND MODE 🔥"
echo "================================================="

# Check CUDA version
echo "📍 Checking CUDA version..."
nvcc --version || echo "⚠️ CUDA not found in PATH"

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p external/EndoGaussian
cd external/EndoGaussian

# Clone EndoGaussian repo
echo "📥 Cloning EndoGaussian repository..."
if [ ! -d "EndoGaussian" ]; then
    git clone https://github.com/CUHK-AIM-Group/EndoGaussian.git
    cd EndoGaussian
else
    cd EndoGaussian
    git pull
fi

# Create conda environment
echo "🐍 Creating conda environment (endogaussian)..."
conda create -n endogaussian python=3.7 -y || echo "Environment may already exist"
conda activate endogaussian

# Install PyTorch with CUDA 11.7
echo "🔧 Installing PyTorch 1.13.1 with CUDA 11.7..."
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Install dependencies
echo "📦 Installing dependencies..."
pip install tqdm
pip install plyfile
pip install opencv-python
pip install scikit-image
pip install lpips
pip install --upgrade pip

# Install pytorch3d (might take a while)
echo "🎨 Installing pytorch3d..."
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install submodules for Gaussian rasterization
echo "⚡ Building Gaussian rasterization CUDA extensions..."
cd submodules
pip install ./diff-gaussian-rasterization
pip install ./simple-knn
cd ..

echo ""
echo "✅ ENDOGAUSSIAN INSTALLATION COMPLETE!"
echo "======================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: conda activate endogaussian"
echo "2. Prepare your data in data/video49/"
echo "3. Run training: python train.py --config configs/cholec80.yaml"
echo ""
echo "Training time: ~2 MINUTES! 🚀"
