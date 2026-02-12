# 🏆 OpenSurgAI - GTC 2026 Golden Ticket Submission Guide

## Quick Start (60 seconds to demo!)

```bash
# 1. Validate setup
python scripts/validate_setup.py

# 2. Run full demo pipeline (one command!)
python scripts/run_full_demo.py --video video49

# 3. Access dashboard
# Opens automatically at http://localhost:8501
```

---

## What is OpenSurgAI?

**End-to-end AI-powered surgical intelligence system** showcasing the complete NVIDIA technology stack:

### 🚀 Key Features

1. **TensorRT Ultra-Fast Inference**
   - 1,300 FPS tool classification
   - 26x speedup vs PyTorch
   - FP16 precision optimization

2. **EndoGaussian 3D Reconstruction**
   - 195 FPS real-time rendering
   - 2-minute training time
   - Interactive click-to-explore
   - Photorealistic quality (37.8 PSNR)

3. **Nemotron AI Reasoning**
   - Surgical context understanding
   - Real-time Q&A
   - Phase-aware explanations

4. **Professional Dashboard**
   - Medical-grade UI
   - Live HUD overlay
   - Multi-surgery comparison
   - 3D interactive viewer

---

## Installation

### Prerequisites

- **GPU:** NVIDIA GPU with CUDA 11.7+
- **Software:** Conda or Miniconda
- **Disk:** ~10GB free space
- **OS:** Windows or Linux

### Option 1: Automated Setup (Recommended)

**Windows:**
```bash
scripts\setup_endogaussian.bat
```

**Linux/Mac:**
```bash
python scripts/setup_endogaussian.py --yes
```

### Option 2: Manual Setup

```bash
# 1. Clone EndoGaussian
git clone https://github.com/CUHK-AIM-Group/EndoGaussian.git external/EndoGaussian
cd external/EndoGaussian

# 2. Create environment
conda create -n endogaussian python=3.7 -y
conda activate endogaussian

# 3. Install dependencies
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install tqdm plyfile opencv-python scikit-image lpips imageio imageio-ffmpeg trimesh

# 4. Install PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# 5. Build CUDA extensions
git submodule update --init --recursive
cd submodules/diff-gaussian-rasterization && pip install . && cd ../..
cd submodules/simple-knn && pip install . && cd ../..
```

---

## Usage

### Validation

**Always run this first to check your setup:**

```bash
python scripts/validate_setup.py
```

Checks:
- ✅ Python environment
- ✅ Core dependencies
- ✅ CUDA and GPU
- ✅ TensorRT models
- ✅ EndoGaussian installation
- ✅ Data files
- ✅ API keys

### Full Demo Pipeline

**One command to rule them all:**

```bash
python scripts/run_full_demo.py --video video49
```

This runs:
1. Detection pipeline (TensorRT classification)
2. 3D reconstruction training (2 minutes!)
3. HUD overlay rendering
4. Interactive dashboard launch

**Skip steps you've already done:**

```bash
# Skip detection (use existing results)
python scripts/run_full_demo.py --video video49 --skip-detection

# Skip 3D training (already trained)
python scripts/run_full_demo.py --video video49 --skip-3d

# Only launch dashboard
python scripts/run_full_demo.py --video video49 --dashboard-only
```

### Individual Components

**1. Run Detection Pipeline:**
```bash
python scripts/run_detection.py --video video49
```

**2. Prepare 3D Data:**
```bash
python scripts/prepare_cholec80_for_gaussian.py --video video49
```

**3. Train EndoGaussian:**
```bash
cd external/EndoGaussian
conda activate endogaussian
python train.py -s ../../data/video49 -m video49
```

**4. Render HUD Overlay:**
```bash
python scripts/render_demo_video.py --video video49
```

**5. Launch Dashboard:**
```bash
streamlit run scripts/app_dashboard.py
```

### GTC Submission Video

**Create cinematic 60-90 second demo video:**

```bash
python scripts/render_gtc_demo.py --video video49 --duration 90
```

Output: `experiments/gtc_demo.mp4`

Includes:
- Title cards
- Feature highlights
- Annotated surgery footage
- Performance metrics
- Professional branding

---

## Dashboard Features

### Tab 1: 📊 Overview
- Video playback with HUD overlay
- Real-time phase timeline
- Live instrument tracking
- Performance metrics

### Tab 2: 🎯 3D Scene
- **Interactive EndoGaussian reconstruction**
- Rotate, zoom, pan controls
- Click on points for AI explanations
- Camera angle presets
- Point density controls

### Tab 3: 💬 AI Analysis
- Nemotron-powered Q&A
- Surgical context understanding
- Preset questions
- Custom queries

### Tab 4: ⚖ Compare
- Multi-surgery analysis
- Phase progression comparison
- Workflow visualization
- Statistical summaries

---

## Technical Specifications

### Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| TensorRT Inference | FPS | 1,300 |
| PyTorch Baseline | FPS | 50 |
| Speedup | Ratio | 26x |
| 3D Rendering | FPS | 195 |
| 3D Training | Time | 2 minutes |
| 3D Quality | PSNR | 37.8 |
| Tool Classification | Accuracy | 95%+ |

### Technology Stack

**Inference:**
- NVIDIA TensorRT 8.6+
- PyTorch 2.1.2
- CUDA 12.1

**3D Reconstruction:**
- EndoGaussian (Gaussian Splatting)
- PyTorch 1.13.1 + CUDA 11.7
- PyTorch3D

**AI Reasoning:**
- NVIDIA Nemotron-70B
- OpenAI API compatibility

**Visualization:**
- Streamlit
- Plotly 3D
- OpenCV

**Data:**
- Cholec80 surgical dataset
- 7 instruments, 7 phases
- Laparoscopic cholecystectomy

---

## Project Structure

```
OpenSurgAI/
├── scripts/
│   ├── run_full_demo.py              # 🎯 One-command demo orchestration
│   ├── validate_setup.py             # ✅ Dependency checker
│   ├── render_gtc_demo.py            # 🎬 Submission video creator
│   ├── setup_endogaussian.py         # 📦 Auto-installer (Python)
│   ├── setup_endogaussian.bat        # 📦 Auto-installer (Windows)
│   ├── run_detection.py              # 🔍 TensorRT detection pipeline
│   ├── prepare_cholec80_for_gaussian.py  # 📸 3D data preparation
│   ├── render_demo_video.py          # 🎥 HUD overlay renderer
│   └── app_dashboard.py              # 🖥️ Interactive dashboard
│
├── src/
│   ├── detection/                    # TensorRT inference
│   ├── visualization/                # 3D viewer components
│   │   ├── gaussian_viewer.py        # Point cloud loader
│   │   └── gaussian_streamlit.py     # Dashboard integration
│   ├── explanation/                  # Nemotron integration
│   └── dashboard/                    # HUD overlay renderer
│
├── models/
│   └── cholec80_resnet50_trt_fp16.engine  # TensorRT model
│
├── external/
│   └── EndoGaussian/                 # 3D reconstruction (git clone)
│
└── experiments/
    ├── scenes/                       # Detection results
    ├── dashboard/                    # Rendered videos
    └── gtc_demo.mp4                  # Submission video
```

---

## Troubleshooting

### Common Issues

**1. "CUDA not available"**
- Install NVIDIA drivers
- Install CUDA Toolkit 11.7+
- Verify: `nvcc --version`

**2. "TensorRT model not found"**
```bash
python scripts/export_tensorrt.py
```

**3. "Conda environment activation failed"**
- Use batch script on Windows: `scripts\setup_endogaussian.bat`
- Or use conda run: `conda run -n endogaussian <command>`

**4. "PyTorch3D installation failed"**
```bash
# Try pre-built wheels
conda install -c fvcore -c iopath -c conda-forge pytorch3d
```

**5. "Submodules not initialized"**
```bash
cd external/EndoGaussian
git submodule update --init --recursive
```

---

## GTC 2026 Submission Checklist

- [ ] Run validation: `python scripts/validate_setup.py`
- [ ] Process video: `python scripts/run_full_demo.py --video video49`
- [ ] Test 3D viewer: Click on points, verify Nemotron responses
- [ ] Record demo: `python scripts/render_gtc_demo.py`
- [ ] Review video: Check `experiments/gtc_demo.mp4`
- [ ] Prepare submission materials:
  - [ ] Demo video (60-90 seconds)
  - [ ] GitHub repo link
  - [ ] README with features
  - [ ] Screenshots of dashboard
- [ ] Submit with hashtag: **#NVIDIAGTC**

---

## Key Talking Points

**For Judges:**

1. **Complete NVIDIA Stack Integration**
   - TensorRT: 26x speedup
   - Nemotron: Advanced reasoning
   - CUDA: All computation accelerated

2. **Novel 3D Visualization**
   - First surgical tool to use Gaussian Splatting
   - Interactive exploration with AI annotations
   - Real-time performance (195 FPS)

3. **Production-Ready Design**
   - Professional medical interface
   - Comprehensive automation
   - Extensive documentation

4. **Educational Value**
   - Surgical training enhancement
   - Real-time explanations
   - Multi-case comparison

---

## Credits

**Technologies:**
- NVIDIA TensorRT, CUDA, Nemotron
- EndoGaussian (CUHK-AIM-Group)
- Cholec80 Dataset (University of Strasbourg)
- PyTorch, Streamlit, Plotly

**Built for:**
NVIDIA GTC 2026 Golden Ticket Contest

**License:**
MIT (see LICENSE file)

---

## Support

**Issues?**
- Check troubleshooting section
- Run validation script
- Review error messages
- GitHub issues: https://github.com/Am1r-codes/OpenSurgAI/issues

**Questions?**
- Read ENDOGAUSSIAN_QUICKSTART.md
- Check README.md
- Review code comments

---

## Let's Win This! 🚀

**You've built something incredible:**
- ✅ Cutting-edge 3D reconstruction
- ✅ Ultra-fast TensorRT inference
- ✅ AI-powered surgical reasoning
- ✅ Professional presentation

**Go show the judges what NVIDIA technology can do!**

#NVIDIAGTC #AI #Surgery #DeepLearning #ComputerVision #TensorRT
