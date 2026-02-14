# OpenSurgAI - GTC 2026 Golden Ticket Submission Guide

## Quick Start (60 seconds to demo!)

```bash
# 1. Validate setup
python scripts/validate_setup.py

# 2. Launch dashboard
streamlit run scripts/app_dashboard.py

# 3. Or run full pipeline (one command!)
python scripts/run_full_demo.py --video video49
```

---

## What is OpenSurgAI?

**Multi-NIM Surgical Intelligence Platform** — orchestrating 3 NVIDIA NIM services for end-to-end surgical video analysis.

### Key Features

1. **TensorRT FP16 Inference**
   - 1,335 FPS tool classification on RTX 5060 Ti
   - 26x speedup vs PyTorch
   - Real-time surgical instrument detection

2. **Nemotron 49B Text Reasoning**
   - Post-hoc surgical case analysis
   - Educational Q&A with hidden reasoning
   - Operative report generation

3. **Nemotron VL Visual Analysis**
   - Frame-level visual understanding
   - Critical View of Safety assessment
   - Instrument identification and teaching points

4. **Professional Dashboard**
   - Medical-grade Streamlit UI
   - 3D Semantic Workflow Space
   - Live HUD overlay video playback
   - Multi-NIM orchestration display

---

## Installation

### Prerequisites

- **GPU:** NVIDIA GPU with CUDA support
- **Software:** Conda or Miniconda, Python 3.9+
- **API Key:** NVIDIA API key for NIM services
- **OS:** Windows or Linux

### Setup

```bash
# 1. Create conda environment
conda create -n opensurgai python=3.10 -y
conda activate opensurgai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
export NVIDIA_API_KEY=nvapi-...

# 4. Validate
python scripts/validate_setup.py
```

---

## Usage

### Validation

**Always run this first:**

```bash
python scripts/validate_setup.py
```

Checks:
- Python environment
- Core dependencies (PyTorch, OpenCV, Streamlit, httpx)
- CUDA and GPU
- TensorRT models
- NVIDIA NIM API services (Nemotron + Nemotron VL)
- Data files
- Dashboard configuration

### Full Demo Pipeline

**One command to rule them all:**

```bash
python scripts/run_full_demo.py --video video49
```

This runs:
1. Detection pipeline (TensorRT classification)
2. HUD overlay rendering
3. Interactive dashboard launch

**Skip steps you've already done:**

```bash
# Skip detection (use existing results)
python scripts/run_full_demo.py --video video49 --skip-detection

# Only launch dashboard
python scripts/run_full_demo.py --video video49 --dashboard-only
```

### Individual Components

**1. Run Detection Pipeline:**
```bash
python scripts/run_detection.py --video video49
```

**2. Render HUD Overlay:**
```bash
python scripts/run_dashboard.py --video data/cholec80/videos/video49.mp4
```

**3. Launch Dashboard:**
```bash
streamlit run scripts/app_dashboard.py
```

### GTC Submission Video

**Create cinematic 60-90 second demo video:**

```bash
python scripts/render_gtc_demo.py --video video49 --duration 90
```

---

## Dashboard Features

### Tab 1: Overview
- Video playback with HUD overlay
- Real-time phase timeline
- Live instrument tracking with confidence bars
- Circular progress indicators

### Tab 2: AI Analysis
- **Nemotron VL Visual Frame Analysis** — select any frame, choose analysis preset, get visual understanding
- **Nemotron Text Reasoning** — post-hoc Q&A with structured procedure summary
- Preset + custom questions
- Hidden reasoning expandable

### Tab 3: 3D Workflow Space
- Interactive Plotly 3D scatter visualization
- Phase segments and transitions tables
- Interpretation guide

### Below Tabs
- **Operative Case Report** — Nemotron-generated structured report
- Download as Markdown or styled HTML (print-ready)

---

## Technical Specifications

### Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| TensorRT Inference | FPS | 1,335 |
| PyTorch Baseline | FPS | ~50 |
| Speedup | Ratio | 26x |
| Tool Classification | Accuracy | 95%+ |
| GPU | Model | RTX 5060 Ti (Blackwell) |

### Technology Stack

**NIM Service 1 — TensorRT FP16:**
- Local GPU inference
- ResNet50 tool classifier
- 7 instrument classes

**NIM Service 2 — Nemotron 49B:**
- Cloud NIM API (`integrate.api.nvidia.com`)
- Surgical text reasoning and report generation
- OpenAI-compatible chat completions

**NIM Service 3 — Nemotron VL:**
- Cloud NIM API (multimodal)
- Base64 JPEG frame + text prompt
- Visual surgical understanding

**Visualization:**
- Streamlit (medical-grade UI)
- Plotly (3D workflow space)
- OpenCV (frame extraction)

**Data:**
- Cholec80 surgical dataset
- 7 instruments, 7 phases
- Laparoscopic cholecystectomy

---

## Project Structure

```
OpenSurgAI/
├── scripts/
│   ├── app_dashboard.py              # Interactive dashboard (3 tabs)
│   ├── run_full_demo.py              # One-command demo orchestration
│   ├── validate_setup.py             # Dependency checker
│   ├── render_gtc_demo.py            # Submission video creator
│   ├── run_detection.py              # TensorRT detection pipeline
│   ├── run_dashboard.py               # HUD overlay recorder
│   ├── run_report.py                 # Operative report generator
│   └── run_posthoc_qa.py             # Post-hoc Q&A functions
│
├── src/
│   ├── detection/                    # TensorRT inference
│   ├── explanation/
│   │   ├── pipeline.py               # NemotronClient + explanation pipeline
│   │   ├── vlm_client.py             # VLMClient for Nemotron VL NIM API
│   │   └── frame_extractor.py        # Video frame extraction
│   ├── analysis/
│   │   └── phase_space_3d.py         # 3D workflow space
│   └── phase/                        # Phase recognition
│
├── weights/                          # Model weights
│   ├── tool_resnet50.pt
│   ├── phase_resnet50.pt
│   └── tensorrt/
│       └── tool_resnet50_trt.ts      # TensorRT FP16 compiled
│
├── data/cholec80/videos/             # Cholec80 surgical videos
├── experiments/
│   ├── scene/                        # Detection results (JSONL)
│   └── dashboard/                    # Rendered overlay videos
│
└── GTC_SUBMISSION_GUIDE.md           # This file
```

---

## GTC 2026 Submission Checklist

- [ ] Run validation: `python scripts/validate_setup.py`
- [ ] Process video: `python scripts/run_full_demo.py --video video49`
- [ ] Test Nemotron VL: Click "Analyze Frame with Nemotron VL" in AI Analysis tab
- [ ] Test Nemotron: Ask a question in AI Analysis tab
- [ ] Test 3D Workflow: Explore the 3D scatter in Workflow Space tab
- [ ] Generate report: Click "Generate Report" below tabs
- [ ] Record demo: `python scripts/render_gtc_demo.py`
- [ ] Review video: Check `experiments/gtc_demo.mp4`
- [ ] Submit with hashtag: **#NVIDIAGTC**

---

## Key Talking Points

**For Judges:**

1. **Multi-NIM Orchestration**
   - 3 NVIDIA NIM services working together
   - TensorRT (local GPU) + Nemotron 49B (cloud) + Nemotron VL (cloud)
   - Single API key, unified platform

2. **Real Healthcare AI Impact**
   - Surgical education and training
   - Post-hoc case review and analysis
   - Operative report automation

3. **NVIDIA Technology Stack**
   - NIM Microservices (Nemotron 49B + Nemotron VL)
   - TensorRT FP16 acceleration (1,335 FPS)
   - CUDA-powered GPU inference
   - OpenAI-compatible API orchestration

4. **Production-Ready Design**
   - Professional medical interface
   - Comprehensive automation
   - HTML report export for clinical use

---

## Credits

**Technologies:**
- NVIDIA TensorRT, CUDA, Nemotron 49B, Nemotron VL, NIM
- Cholec80 Dataset (University of Strasbourg)
- PyTorch, Streamlit, Plotly

**Built for:**
NVIDIA GTC 2026 Golden Ticket Contest

**License:**
MIT (see LICENSE file)
