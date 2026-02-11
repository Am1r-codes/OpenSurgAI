# OpenSurgAI

**AI-powered surgical workflow analysis for laparoscopic cholecystectomy.**

End-to-end pipeline combining computer vision (YOLOv8s, ResNet50) with NVIDIA Nemotron Super 49B for post-hoc surgical reasoning, multi-surgery comparison, and case review.  Built on the [Cholec80](http://camma.u-strasbg.fr/datasets) dataset.

> Submitted for the NVIDIA GTC 2026 Golden Ticket.

---

## What it does

OpenSurgAI processes surgical videos through a multi-stage vision pipeline, then maps the results into a **3D Semantic Surgical Workflow Space** — a novel representation where each frame becomes a point in a three-dimensional procedural landscape.  A Nemotron-powered assistant can then reason over the full procedure, generate case reports, and answer educational questions.

**This is not real-time narration.  This is not anatomical 3D reconstruction.**
It is structured, post-hoc surgical case review powered by NVIDIA AI.

---

## Architecture

```
Surgical Video (Cholec80)
         |
         v
    +----+----+
    |         |
    v         v
 YOLOv8s   ResNet50
(detection) (phase)
    |         |
    +----+----+
         |
         v
   Scene Assembly
   (per-frame JSONL)
         |
    +----+----+
    |         |
    v         v
 Dashboard   3D Semantic
 Recorder    Workflow Space
 (overlay      |
  video)       |
    |     +----+----+
    |     |         |
    v     v         v
 Streamlit Dashboard
 +-------+--------+---------+
 | LEFT  | CENTER |  RIGHT  |
 | Video | 3D Viz | Nemotron|
 +-------+--------+---------+
              |
              v
     Nemotron Super 49B
     (case reports, Q&A,
      multi-surgery comparison)
```

### Pipeline stages

| Stage | Model / Tool | Output |
|---|---|---|
| Instrument detection | YOLOv8s (Ultralytics) | Per-frame instrument count + metadata |
| Phase recognition | ResNet50 (fine-tuned on Cholec80) | Phase ID + confidence per frame |
| Scene assembly | `run_scene_assembly.py` | Unified per-frame JSONL |
| Explanation | Nemotron Super 49B | System commentary per phase transition |
| 3D Workflow Space | `phase_space_3d.py` | Semantic 3D point cloud + trajectory |
| Dashboard | Streamlit | Interactive case review UI |
| Case report | Nemotron Super 49B | Structured markdown report |

---

## 3D Semantic Surgical Workflow Space

Each frame of a processed surgical video maps to **one point** in a 3D semantic space:

- **X — Phase Progression** : normalised progress within the current phase segment [0 -> 1].  Resets on every phase change.
- **Y — Phase Identity** : ordinal surgical phase index (0-6 for Cholec80).
- **Z — Surgical Activity / Complexity** : composite metric combining instrument count (normalised) and confidence volatility (rolling standard deviation).

This produces:
- **Dense clusters** for stable phases
- **Scattered regions** for complex transitions
- **Trajectory lines** showing the surgical path through workflow space
- **Phase revisits** as visually distinct return clusters

**This represents procedural structure and activity, not anatomical geometry or spatial reconstruction.**

---

## NVIDIA Technology

| Component | NVIDIA Tech |
|---|---|
| Inference | PyTorch + CUDA (RTX 5060 Ti) |
| Phase recognition | ResNet50 trained with `torchvision` |
| Instrument detection | YOLOv8s (Ultralytics, CUDA) |
| Natural language reasoning | **NVIDIA Nemotron Super 49B** via NIM API |
| Post-hoc Q&A | Nemotron chat completions |
| Case report generation | Nemotron structured generation |
| Multi-surgery comparison | Nemotron comparative analysis |

---

## Quick start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA support
- Cholec80 dataset (videos + phase annotations) — **not included in this repo**; you must request access from the [CAMMA group](http://camma.u-strasbg.fr/datasets) and agree to their license terms (redistribution is prohibited)
- NVIDIA API key for Nemotron (`NEMOTRON_API_KEY` env var)

### Install

```bash
git clone https://github.com/Am1r-codes/OpenSurgAI.git
cd OpenSurgAI
pip install -r requirements.txt
```

### Run the pipeline

```bash
# 1. Prepare Cholec80 data
python scripts/prepare_cholec80.py

# 2. Run detection (instruments)
python scripts/run_detection.py --video video49

# 3. Run phase recognition
python scripts/run_phase_recognition.py --video video49

# 4. Assemble scene JSONL
python scripts/run_scene_assembly.py --video video49

# 5. Generate explanations (Nemotron)
python scripts/run_explanation.py --video video49

# 6. Build 3D workflow space
python scripts/run_phase_space.py --video video49

# 7. Record annotated overlay video
python scripts/run_dashboard.py --video video49
```

### Launch the dashboard

```bash
streamlit run scripts/app_dashboard.py
```

### Generate a case report

```bash
python scripts/run_report.py --video video49
```

### Post-hoc Q&A (CLI)

```bash
python scripts/run_posthoc_qa.py --video video49
```

### 3D workflow space (standalone HTML)

```bash
python scripts/run_phase_space.py --video video49
```

---

## Project structure

```
OpenSurgAI/
  src/
    detection/      # YOLOv8s instrument detection
    phase/          # ResNet50 phase recognition
    scene/          # Scene assembly (JSONL)
    explanation/    # Nemotron explanation pipeline
    analysis/       # 3D Semantic Workflow Space + multi-surgery comparison
    dashboard/      # Overlay renderer + video recorder
    video.py        # Video I/O utilities
  scripts/
    app_dashboard.py        # Streamlit dashboard (main UI)
    run_report.py           # Nemotron case report generator
    run_posthoc_qa.py       # CLI Q&A interface
    run_phase_space.py      # 3D visualisation (HTML export)
    run_detection.py        # Instrument detection
    run_phase_recognition.py # Phase recognition
    run_scene_assembly.py   # Scene JSONL assembly
    run_explanation.py      # Nemotron explanation generation
    run_dashboard.py        # Annotated video recorder
    train_phase.py          # Phase model training
    prepare_cholec80.py     # Dataset preparation
  data/
    cholec80/               # Cholec80 dataset (not committed)
  experiments/
    scene/                  # Scene JSONL files
    dashboard/              # Annotated overlay videos
    analysis/               # 3D visualisations
    reports/                # Generated case reports
  requirements.txt
```

---

## Cholec80 phases

| Index | Phase | Description |
|---|---|---|
| 0 | Preparation | Insufflation, trocar placement, gallbladder exposure |
| 1 | CalotTriangleDissection | Dissection of Calot triangle to identify cystic structures |
| 2 | ClippingCutting | Clipping and division of cystic duct and artery |
| 3 | GallbladderDissection | Separation of gallbladder from liver bed |
| 4 | GallbladderPackaging | Placement into retrieval bag |
| 5 | CleaningCoagulation | Hemostasis and inspection |
| 6 | GallbladderRetraction | Retraction for improved visualisation |

---

## Dashboard

The Streamlit dashboard provides a three-panel case review interface:

- **LEFT** — Pre-rendered annotated video (auto-playing) with auto-synced phase info and a **time slider** to scrub the 3D cursor to any point in the procedure
- **CENTER** — Interactive 3D Semantic Surgical Workflow Space with trajectory lines and active phase highlighting
- **RIGHT** — Nemotron Q&A with preset questions and hidden reasoning

A virtual playback clock provides approximate synchronisation between video playback and UI state.  Full case reports can be generated with one click.

### Multi-surgery comparison

The dashboard includes a **Compare Surgeries** section where you can select multiple processed videos and overlay their 3D workflow trajectories in a single figure.  Each surgery is rendered in a distinct colour so structural differences (duration, phase ordering, revisits) are immediately visible.

A dedicated **Nemotron Q&A panel** lets you ask comparative questions ("Which surgery was more complex?", "What phase timing differences stand out?") with full context of all selected procedures.

### Video upload

Upload your own surgical video directly through the sidebar.  The dashboard will run the full pipeline automatically (detection → phase recognition → scene assembly → 3D workflow space → overlay recorder) and add the result to the video selector.

---

## Key design decisions

1. **Post-hoc, not real-time**: All reasoning happens after the procedure is complete.  This is deliberate — it enables deeper analysis and avoids the liability of real-time surgical guidance.

2. **Semantic 3D, not anatomical**: The 3D space represents workflow structure (progression, identity, activity), not physical anatomy.  No claim of spatial accuracy is made.

3. **Two-part explanation model**: Part A is static, expert-written canonical phase descriptions.  Part B is dynamic Nemotron commentary.  This guarantees a correct baseline even if the LLM fails.

4. **Hidden reasoning**: Nemotron output is split into a visible answer and hidden analytical reasoning.  This keeps the interface calm and focused.

---

## Requirements

```
torch>=2.2.0
torchvision>=0.17.0
ultralytics>=8.1.0
opencv-python-headless>=4.9.0
imageio-ffmpeg>=0.5.1
httpx>=0.27.0
plotly>=5.18.0
streamlit>=1.31.0
numpy>=1.26.0
```

---

## License

MIT

---

## Acknowledgements

- **Cholec80 dataset**: CAMMA group, University of Strasbourg ([link](http://camma.u-strasbg.fr/datasets)) — released under [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you use this dataset, please cite:

  > A.P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. de Mathelin, N. Padoy. *EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos.* IEEE Transactions on Medical Imaging (TMI), 2017.

- **NVIDIA Nemotron Super 49B**: NVIDIA NIM API
- **Ultralytics YOLO**: [ultralytics.com](https://ultralytics.com)
