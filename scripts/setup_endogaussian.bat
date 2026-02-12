@echo off
REM EndoGaussian Setup for Windows
REM Run this after initial conda environment creation

echo ====================================
echo  EndoGaussian Windows Installation
echo ====================================
echo.

cd /d "%~dp0\.."

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda not found! Please install Miniconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [*] Creating conda environment...
call conda create -n endogaussian python=3.7 -y
if %ERRORLEVEL% NEQ 0 (
    echo [!] Environment creation failed, but it might already exist. Continuing...
)

echo.
echo [*] Cloning EndoGaussian repository...
if not exist "external" mkdir external
cd external

if not exist "EndoGaussian" (
    git clone https://github.com/CUHK-AIM-Group/EndoGaussian.git
    if %ERRORLEVEL% NEQ 0 (
        echo [!] Git clone failed!
        pause
        exit /b 1
    )
) else (
    echo [+] EndoGaussian already exists
)

cd EndoGaussian

echo.
echo [*] Activating environment and installing dependencies...
echo    (This will take 15-30 minutes...)
echo.

REM Run all installations in the conda environment
call conda activate endogaussian

echo [*] Installing PyTorch 1.13.1 + CUDA 11.7...
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

echo.
echo [*] Installing core dependencies...
pip install tqdm plyfile opencv-python scikit-image lpips imageio imageio-ffmpeg trimesh

echo.
echo [*] Installing PyTorch3D...
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

echo.
echo [*] Building CUDA extensions...

REM Initialize submodules if needed
if not exist "submodules\diff-gaussian-rasterization" (
    echo [*] Initializing submodules...
    git submodule update --init --recursive
)

REM Build diff-gaussian-rasterization
if exist "submodules\diff-gaussian-rasterization" (
    cd submodules\diff-gaussian-rasterization
    pip install .
    cd ..\..
)

REM Build simple-knn
if exist "submodules\simple-knn" (
    cd submodules\simple-knn
    pip install .
    cd ..\..
)

echo.
echo ====================================
echo  Installation Complete!
echo ====================================
echo.
echo Next steps:
echo   1. conda activate endogaussian
echo   2. python scripts/prepare_cholec80_for_gaussian.py --video video49
echo   3. cd external/EndoGaussian
echo   4. python train.py -s ../../data/video49 -m video49
echo.
echo Press any key to exit...
pause >nul
