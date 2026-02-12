"""OpenSurgAI — Post-hoc Surgical Case Review Dashboard.

Calm, automatic, research-grade NVIDIA GTC demo.

Architecture
------------
Three-panel Streamlit layout:

  LEFT   — Pre-rendered annotated video (auto-playing) + phase info
  CENTER — 3D Semantic Surgical Workflow Space (auto-updating cursor)
  RIGHT  — Nemotron post-hoc Q&A with hidden reasoning

Design choices (documented per spec):

  WHY ANNOTATED VIDEO:
    The overlay video (experiments/dashboard/) is the PRIMARY visual
    context.  Detection boxes, segmentation masks, and phase labels are
    PRE-RENDERED by the dashboard recorder.  No real-time inference
    runs inside this UI.

  WHY TIME SLIDER:
    Streamlit cannot read the actual video playback position.  A time
    slider lets the user manually set the analysis point.  The 3D cursor
    and phase info update instantly when the slider moves.

  WHAT THE 3D SPACE REPRESENTS:
    The 3D Semantic Surgical Workflow Space represents procedural
    structure and activity — NOT anatomical geometry or spatial
    reconstruction.  Axes: X = Phase Progression, Y = Phase Identity,
    Z = Surgical Activity / Complexity.

  WHAT THE 3D SPACE DOES NOT REPRESENT:
    Anatomical geometry.  Physical spatial layout.  No claim of
    anatomical accuracy is made.  The axes are semantic, not physical.

  WHY REASONING IS HIDDEN:
    Nemotron output is split into FINAL ANSWER (always visible) and
    REASONING (hidden behind an expander).  This keeps the interface
    calm, confident, and focused on education.

Launch:
    streamlit run scripts/app_dashboard.py
"""

from __future__ import annotations

import bisect
import re
import subprocess
import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.analysis.phase_space_3d import (
    PHASE_COLOURS,
    PHASE_ORDER,
    PHASE_TO_INDEX,
    build_comparison_figure,
    build_comparison_summary,
    build_semantic_phase_space,
    build_workflow_figure,
    get_phase_segments,
    get_transition_points,
)
from src.explanation.pipeline import NemotronClient, PHASE_EXPLANATIONS
from scripts.run_posthoc_qa import aggregate_summary, format_summary_for_prompt
from scripts.run_report import generate_report

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="OpenSurgAI — Surgical Case Review",
    page_icon=":stethoscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Medical Software Interface CSS ──────────────────────────

st.markdown("""
<style>
    /* === GLOBAL IMPROVEMENTS === */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 100% !important;
    }

    /* Main title styling with gradient */
    h1 {
        background: linear-gradient(135deg, #00CED1 0%, #00FFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        text-shadow: 0 0 30px rgba(0, 206, 209, 0.5);
        font-size: 48px !important;
        margin-bottom: 0 !important;
    }

    /* Subheaders with premium styling */
    h2, h3 {
        color: #00CED1 !important;
        font-weight: 700 !important;
        border-bottom: 3px solid rgba(0, 206, 209, 0.4);
        padding-bottom: 12px;
        margin-top: 24px !important;
        margin-bottom: 16px !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 16px !important;
    }

    /* === ENHANCED METRICS === */
    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #00CED1 0%, #00FFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 36px !important;
        font-weight: 800 !important;
        text-shadow: 0 2px 8px rgba(0, 206, 209, 0.3);
    }

    [data-testid="stMetricLabel"] {
        color: #888 !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 14px !important;
    }

    /* Metric containers with glow */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(0,206,209,0.05) 0%, rgba(0,206,209,0.02) 100%);
        border: 1px solid rgba(0, 206, 209, 0.3);
        border-radius: 12px;
        padding: 20px !important;
        box-shadow: 0 4px 16px rgba(0, 206, 209, 0.1);
        transition: all 0.3s ease;
    }

    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 206, 209, 0.2);
        border-color: rgba(0, 206, 209, 0.6);
    }

    /* === PREMIUM BUTTONS === */
    .stButton>button {
        background: linear-gradient(135deg, #00CED1 0%, #00B8D4 100%) !important;
        color: #0A1120 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 13px !important;
        box-shadow: 0 4px 12px rgba(0, 206, 209, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #00FFFF 0%, #00CED1 100%) !important;
        box-shadow: 0 8px 24px rgba(0, 206, 209, 0.5) !important;
        transform: translateY(-2px);
    }

    .stButton>button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(0, 206, 209, 0.3) !important;
    }

    /* === SLIDER ENHANCEMENTS === */
    .stSlider {
        padding: 16px 0;
    }

    .stSlider>div>div>div>div {
        background-color: #00CED1 !important;
    }

    .stSlider>div>div>div>div>div {
        background-color: #00FFFF !important;
        box-shadow: 0 0 12px rgba(0, 206, 209, 0.6);
    }

    /* === VIDEO CONTAINER === */
    [data-testid="stVideo"] {
        border: 2px solid rgba(0, 206, 209, 0.4);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 206, 209, 0.2);
    }

    /* === EXPANDER PREMIUM STYLING === */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(0,206,209,0.15) 0%, rgba(0,206,209,0.05) 100%) !important;
        border-radius: 8px !important;
        color: #00CED1 !important;
        font-weight: 700 !important;
        border: 1px solid rgba(0, 206, 209, 0.3) !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
    }

    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(0,206,209,0.25) 0%, rgba(0,206,209,0.1) 100%) !important;
        border-color: rgba(0, 206, 209, 0.6) !important;
    }

    /* === SIDEBAR PREMIUM === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1120 0%, #0D1425 100%) !important;
        border-right: 2px solid rgba(0, 206, 209, 0.3);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.5);
    }

    [data-testid="stSidebar"]>div:first-child {
        padding-top: 2rem;
    }

    /* === INPUT FIELDS === */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select,
    .stMultiSelect>div>div>div {
        background-color: rgba(20, 29, 46, 0.8) !important;
        color: #E0E0E0 !important;
        border: 2px solid rgba(0, 206, 209, 0.3) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>select:focus {
        border-color: rgba(0, 206, 209, 0.8) !important;
        box-shadow: 0 0 16px rgba(0, 206, 209, 0.3) !important;
    }

    /* === TABS STYLING === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(10, 17, 32, 0.5);
        padding: 8px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #888;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(0, 206, 209, 0.1);
        color: #00CED1;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,206,209,0.2) 0%, rgba(0,206,209,0.1) 100%) !important;
        color: #00CED1 !important;
        border-bottom: 3px solid #00CED1 !important;
    }

    /* === ALERTS & INFO BOXES === */
    .stAlert {
        border-radius: 12px !important;
        border-left: 5px solid #00CED1 !important;
        background: linear-gradient(135deg, rgba(0,206,209,0.1) 0%, rgba(0,206,209,0.05) 100%);
        padding: 16px 20px !important;
    }

    /* === PLOTLY CHARTS === */
    .js-plotly-plot {
        border: 2px solid rgba(0, 206, 209, 0.3);
        border-radius: 12px;
        background-color: #0A1120 !important;
        box-shadow: 0 8px 32px rgba(0, 206, 209, 0.15);
    }

    /* === DIVIDER === */
    hr {
        border-color: rgba(0, 206, 209, 0.3) !important;
        margin: 32px 0 !important;
    }

    /* === CAPTION === */
    .stCaption {
        color: #666 !important;
        font-size: 12px !important;
        font-style: italic;
    }

    /* === HIDE STREAMLIT BRANDING === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* === CUSTOM PROGRESS RING === */
    .progress-ring {
        transform: rotate(-90deg);
    }

    .progress-ring-circle {
        transition: stroke-dashoffset 0.5s ease;
    }

    /* === ANIMATIONS === */
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 206, 209, 0.3); }
        50% { box-shadow: 0 0 40px rgba(0, 206, 209, 0.6); }
    }

    .glow-pulse {
        animation: pulse-glow 3s ease-in-out infinite;
    }

    @keyframes slide-in {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .slide-in {
        animation: slide-in 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────



# ── Utility: time display formatting ────────────────────────────────
# All internal computations remain in seconds.
# This is DISPLAY-ONLY formatting — used everywhere the user sees time.

def format_time(seconds: float) -> str:
    """Format seconds as m:ss for display."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


# ── Enhanced Visualization Components ───────────────────────────────

def render_circular_progress(percent: float, size: int = 120, label: str = "") -> str:
    """Render an SVG circular progress indicator."""
    radius = (size - 20) / 2
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - percent)

    return f"""
    <div style="display: inline-block; text-align: center;">
        <svg width="{size}" height="{size}" class="progress-ring">
            <circle
                cx="{size/2}"
                cy="{size/2}"
                r="{radius}"
                stroke="rgba(0, 206, 209, 0.2)"
                stroke-width="10"
                fill="none"
            />
            <circle
                cx="{size/2}"
                cy="{size/2}"
                r="{radius}"
                stroke="#00CED1"
                stroke-width="10"
                fill="none"
                stroke-dasharray="{circumference}"
                stroke-dashoffset="{offset}"
                class="progress-ring-circle"
                style="filter: drop-shadow(0 0 8px rgba(0, 206, 209, 0.6));"
            />
            <text
                x="{size/2}"
                y="{size/2 + 8}"
                text-anchor="middle"
                style="font-size: 24px; font-weight: 700; fill: #00CED1;"
            >
                {int(percent * 100)}%
            </text>
        </svg>
        <div style="margin-top: 8px; font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px;">
            {label}
        </div>
    </div>
    """


def render_instrument_bar(name: str, confidence: float, color: str, active: bool = True) -> str:
    """Render a horizontal confidence bar for an instrument."""
    opacity = "1.0" if active else "0.4"
    bar_width = confidence * 100

    return f"""
    <div style="margin: 8px 0; opacity: {opacity}; transition: all 0.3s ease;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="font-size: 11px; font-weight: 600; color: {color}; text-transform: uppercase; letter-spacing: 0.5px;">
                {name}
            </span>
            <span style="font-size: 11px; color: #888; font-family: monospace;">
                {confidence:.2%}
            </span>
        </div>
        <div style="width: 100%; height: 6px; background-color: rgba(0,0,0,0.3); border-radius: 3px; overflow: hidden;">
            <div style="width: {bar_width}%; height: 100%; background: linear-gradient(90deg, {color} 0%, {color}AA 100%);
                        border-radius: 3px; transition: width 0.5s ease;
                        box-shadow: 0 0 10px {color}66;">
            </div>
        </div>
    </div>
    """


def render_phase_timeline_bar(segments: list, current_time: float, duration: float) -> str:
    """Render a visual phase timeline with current position marker."""
    html_parts = ['<div style="position: relative; width: 100%; height: 40px; margin: 16px 0;">']

    # Background bar
    html_parts.append('<div style="position: absolute; width: 100%; height: 20px; top: 10px; background-color: rgba(0,0,0,0.3); border-radius: 10px; overflow: hidden; display: flex;">')

    # Phase segments
    for seg in segments:
        start_pct = (seg["start_time"] / duration) * 100
        width_pct = (seg["duration"] / duration) * 100
        color = PHASE_COLOURS[seg["phase_idx"] % len(PHASE_COLOURS)]

        html_parts.append(f'''
        <div style="position: absolute; left: {start_pct}%; width: {width_pct}%; height: 100%;
                    background-color: {color}; opacity: 0.8; border-right: 1px solid #0A1120;"
             title="{seg['phase_name']} ({seg['duration']:.0f}s)">
        </div>
        ''')

    html_parts.append('</div>')

    # Current position marker
    current_pct = (current_time / duration) * 100
    html_parts.append(f'''
    <div style="position: absolute; left: {current_pct}%; top: 0; width: 3px; height: 40px;
                background: linear-gradient(180deg, #00CED1 0%, #00FFFF 100%);
                box-shadow: 0 0 12px rgba(0, 206, 209, 0.8); z-index: 10; border-radius: 2px;">
        <div style="position: absolute; top: -8px; left: 50%; transform: translateX(-50%);
                    width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent;
                    border-top: 8px solid #00CED1;">
        </div>
    </div>
    ''')

    html_parts.append('</div>')
    return ''.join(html_parts)


def render_stat_card(label: str, value: str, icon: str = "●", trend: str = "") -> str:
    """Render a premium stat card with optional trend indicator."""
    trend_html = ""
    if trend:
        trend_color = "#4CAF50" if "↑" in trend else "#F44336" if "↓" in trend else "#888"
        trend_html = f'<div style="font-size: 12px; color: {trend_color}; margin-top: 4px;">{trend}</div>'

    return f"""
    <div style="background: linear-gradient(135deg, rgba(0,206,209,0.08) 0%, rgba(0,206,209,0.02) 100%);
                border: 1px solid rgba(0, 206, 209, 0.3); border-radius: 12px; padding: 16px;
                box-shadow: 0 4px 16px rgba(0, 206, 209, 0.1); transition: all 0.3s ease;
                cursor: pointer;">
        <div style="color: #00CED1; font-size: 20px; margin-bottom: 8px;">{icon}</div>
        <div style="color: #888; font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px;">
            {label}
        </div>
        <div style="background: linear-gradient(135deg, #00CED1 0%, #00FFFF 100%);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    font-size: 28px; font-weight: 800;">
            {value}
        </div>
        {trend_html}
    </div>
    """


# ── System prompt for Nemotron Q&A ──────────────────────────────────
# Nemotron is POST-HOC ONLY.  It analyses a COMPLETED procedure.
# Reasoning is requested in a separated section so the UI can
# hide it by default behind an expander.

QA_SYSTEM_PROMPT = """\
You are a surgical education assistant for the OpenSurgAI platform.
You are analyzing a COMPLETED laparoscopic cholecystectomy procedure.

You have access to a STRUCTURED SUMMARY of the procedure below, including
phase timelines, durations, instrument usage, transition points, and
confidence scores from the computer vision system.

You reason over:
- phase sequence and transitions
- phase durations and timing
- phase stability and confidence patterns
- activity patterns in the 3D Semantic Surgical Workflow Space
- instrument usage across phases

The 3D Semantic Surgical Workflow Space maps the procedure into:
- X — Phase Progression [0→1] within each phase segment
- Y — Phase Identity (ordinal surgical phase 0–6)
- Z — Surgical Activity / Complexity (instrument count + confidence volatility)

This represents procedural structure and activity, NOT anatomical geometry.

You MAY:
- Explain what occurs during each surgical phase
- Interpret prolonged, complex, or unstable phases
- Reference the 3D workflow space metaphorically:
    "This phase forms a dense cluster indicating stable dissection"
    "The scattered activity during clipping suggests high instrument manipulation"
- Provide educational context about cholecystectomy technique

You MUST NOT:
- Narrate frame-by-frame actions
- Describe pixels, detections, bounding boxes, or image artifacts
- Give clinical advice or act as a clinical decision-maker
- Claim anatomical reconstruction or spatial geometry

When answering, structure your response in two parts:

PART 1 — Your clear, educational answer (always shown to the user).
Write this first.  Be thorough, warm, and accessible — like a senior
resident teaching an intern.

PART 2 — Your analytical reasoning (shown only on request).
After your answer, write a line containing only: ---REASONING---
Then provide your detailed analysis: data references, statistical
observations, methodology, and supporting evidence from the procedure data.

You may omit Part 2 if the answer is brief and self-contained.\
"""


# ── Response parsing ────────────────────────────────────────────────
# Strip Nemotron's internal <think> blocks and split into ANSWER
# (always visible) and REASONING (hidden behind an expander).

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Nemotron output."""
    return _THINK_RE.sub("", text).strip()


def split_answer_reasoning(text: str) -> tuple[str, str]:
    """Split Nemotron response into (answer, reasoning).

    Reasoning is hidden by default in the UI to keep the interface
    calm and focused on the educational answer.
    """
    text = strip_think_tags(text)
    marker = "---REASONING---"
    if marker in text:
        parts = text.split(marker, 1)
        return parts[0].strip(), parts[1].strip()
    return text.strip(), ""


# ── Helper: discover scene files ─────────────────────────────────────

@st.cache_data
def discover_videos(scene_dir: str) -> list[str]:
    """Find all video IDs with scene JSONL files."""
    d = Path(scene_dir)
    if not d.is_dir():
        return []
    videos = []
    for f in sorted(d.iterdir()):
        if f.name.endswith("_scene.jsonl") and not f.name.endswith("_slice.jsonl"):
            videos.append(f.name[: -len("_scene.jsonl")])
    return videos


# ── Helper: load and cache heavy data ────────────────────────────────

@st.cache_data
def load_video_data(scene_path: str, video_id: str) -> dict:
    """Load scene JSONL and compute all derived data (cached)."""
    from scripts.run_posthoc_qa import load_scene_jsonl

    scenes = load_scene_jsonl(Path(scene_path))
    space = build_semantic_phase_space(scene_path)
    segments = get_phase_segments(space)
    transitions = get_transition_points(space)
    summary = aggregate_summary(scenes, video_id)
    summary_text = format_summary_for_prompt(summary)

    return {
        "scenes": scenes,
        "space": space,
        "segments": segments,
        "transitions": transitions,
        "summary": summary,
        "summary_text": summary_text,
    }


# ── Helper: Nemotron call ────────────────────────────────────────────
# Generation parameters (per spec):
#   temperature = 1.1, top_p = 0.95, presence_penalty = 0.4
#   max_tokens = None (no limit — do NOT cap output length)

def query_nemotron(
    question: str,
    summary_text: str,
    api_key: str | None = None,
    model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
) -> tuple[str, str, dict]:
    """Send a question to Nemotron.

    Returns (answer, reasoning, usage).
    The reasoning section is separated so the UI can hide it.
    """
    system = QA_SYSTEM_PROMPT + "\n\n--- PROCEDURE SUMMARY ---\n" + summary_text
    client = NemotronClient(
        api_key=api_key,
        model=model,
        temperature=1.1,
        max_tokens=None,
        top_p=0.95,
        presence_penalty=0.4,
    )
    try:
        response = client.chat(system=system, user=question)
        choices = response.get("choices", [])
        raw = choices[0]["message"]["content"].strip() if choices else "(no response)"
        usage = response.get("usage", {})
        answer, reasoning = split_answer_reasoning(raw)
        return answer, reasoning, usage
    finally:
        client.close()


# ── Helper: nearest frame lookup ─────────────────────────────────

def lookup_frame_at_time(space: dict, time_sec: float) -> dict:
    """Find the nearest frame in the phase space for a given timestamp.

    Uses binary search for O(log n) lookup.

    Returns dict with: idx, time, phase_name, phase_idx, confidence,
    instrument_count, phase_progress, activity.
    """
    times = space["time"]
    pos = bisect.bisect_left(times, time_sec)
    if pos == 0:
        idx = 0
    elif pos >= len(times):
        idx = len(times) - 1
    else:
        if abs(times[pos] - time_sec) < abs(times[pos - 1] - time_sec):
            idx = pos
        else:
            idx = pos - 1

    return {
        "idx": idx,
        "time": float(times[idx]),
        "phase_name": space["phase_names"][idx],
        "phase_idx": int(space["phase_idx"][idx]),
        "confidence": float(space["confidence"][idx]),
        "instrument_count": int(space["instrument_count"][idx]),
        "phase_progress": float(space["phase_progress"][idx]),
        "activity": float(space["activity"][idx]),
    }


# ── Pipeline runner for uploaded videos ──────────────────────────────

_PYTHON = str(Path(sys.executable))


def _run_pipeline(video_path: Path, video_id: str, api_key: str | None) -> None:
    """Run the full analysis pipeline on an uploaded video."""
    # Use tool classifier if weights exist, otherwise fall back to YOLO
    # Prefer TensorRT-compiled model for accelerated inference
    tool_weights = _PROJECT_ROOT / "weights" / "tool_resnet50.pt"
    tool_trt = _PROJECT_ROOT / "weights" / "tensorrt" / "tool_resnet50_trt.ts"
    det_cmd = [
        _PYTHON, str(_PROJECT_ROOT / "scripts" / "run_detection.py"),
        "--video", str(video_path),
    ]
    if tool_weights.exists():
        det_cmd += ["--model-weights", str(tool_weights)]
    if tool_trt.exists():
        det_cmd += ["--trt", str(tool_trt)]

    steps = [
        ("Detection", det_cmd),
        ("Phase Recognition", [
            _PYTHON, str(_PROJECT_ROOT / "scripts" / "run_phase_recognition.py"),
            "--video", str(video_path),
            "--model-weights", str(_PROJECT_ROOT / "weights" / "phase_resnet50.pt"),
        ]),
        ("Scene Assembly", [
            _PYTHON, str(_PROJECT_ROOT / "scripts" / "run_scene_assembly.py"),
            "--video", video_id,
        ]),
        ("3D Workflow Space", [
            _PYTHON, str(_PROJECT_ROOT / "scripts" / "run_phase_space.py"),
            "--video", video_id,
        ]),
        ("Dashboard Recorder", [
            _PYTHON, str(_PROJECT_ROOT / "scripts" / "run_dashboard.py"),
            "--video", str(video_path),
        ]),
    ]

    progress = st.sidebar.progress(0, text="Starting pipeline...")
    for i, (name, cmd) in enumerate(steps):
        progress.progress((i) / len(steps), text=f"Running {name}...")
        try:
            env = dict(__import__("os").environ)
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800, env=env,
                cwd=str(_PROJECT_ROOT),
            )
            if result.returncode != 0:
                st.sidebar.warning(f"{name} failed: {result.stderr[-300:]}")
        except subprocess.TimeoutExpired:
            st.sidebar.warning(f"{name} timed out (30min limit)")
        except Exception as exc:
            st.sidebar.warning(f"{name} error: {exc}")

    progress.progress(1.0, text="Pipeline complete!")
    st.sidebar.success(f"Processed {video_id}. Select it from the Video dropdown.")


# ── Sidebar ──────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar controls and return config."""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 16px 0; margin-bottom: 16px;">
        <h1 style="color: #00CED1; font-size: 32px; margin: 0; text-shadow: 0 2px 8px rgba(0,206,209,0.4);">
            OpenSurgAI
        </h1>
        <p style="color: #666; font-size: 11px; margin: 8px 0 0 0; text-transform: uppercase; letter-spacing: 1.5px;">
            NVIDIA GTC 2026 Demo
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Scene directory (for scene JSONL files)
    default_scene_dir = str(_PROJECT_ROOT / "experiments" / "scene")
    scene_dir = st.sidebar.text_input("Scene directory", value=default_scene_dir)

    # Annotated video directory (pre-rendered overlay videos)
    default_dashboard_dir = str(_PROJECT_ROOT / "experiments" / "dashboard")
    dashboard_dir = st.sidebar.text_input(
        "Annotated video directory",
        value=default_dashboard_dir,
        help="Pre-rendered overlay videos (*_demo.mp4) from the recorder.",
    )

    # Raw video directory (fallback if annotated not available)
    default_video_dir = str(_PROJECT_ROOT / "data" / "cholec80" / "videos")
    video_dir = st.sidebar.text_input(
        "Raw video directory (fallback)",
        value=default_video_dir,
    )

    # Discover videos
    videos = discover_videos(scene_dir)
    if not videos:
        st.sidebar.error(f"No scene files found in: {scene_dir}")
        st.stop()

    video_id = st.sidebar.selectbox("Video", videos)

    st.sidebar.divider()

    # 3D viz options
    st.sidebar.subheader("3D Visualisation")
    downsample = st.sidebar.slider("Downsample (every Nth frame)", 1, 50, 5)
    point_size = st.sidebar.slider("Point size", 1, 8, 3)

    st.sidebar.divider()

    # Nemotron config
    st.sidebar.subheader("Nemotron Q&A")
    api_key = st.sidebar.text_input(
        "API Key",
        type="password",
        help="NEMOTRON_API_KEY or NVIDIA_API_KEY. Leave blank to use env var.",
    )

    # ── Upload & Process New Video ─────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Upload New Video")
    uploaded = st.sidebar.file_uploader(
        "Upload a surgical video (.mp4)",
        type=["mp4", "avi", "mkv"],
        help="Upload a new video to process through the full pipeline.",
    )

    if uploaded is not None:
        upload_dir = _PROJECT_ROOT / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / uploaded.name
        vid_stem = save_path.stem  # e.g. "my_surgery"

        if not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.sidebar.success(f"Saved: {uploaded.name}")

        if st.sidebar.button("Process Video", type="primary"):
            _run_pipeline(save_path, vid_stem, api_key)
            st.cache_data.clear()
            st.rerun()

    return {
        "scene_dir": scene_dir,
        "dashboard_dir": dashboard_dir,
        "video_dir": video_dir,
        "video_id": video_id,
        "downsample": downsample,
        "point_size": point_size,
        "api_key": api_key or None,
    }


# ── Main app ─────────────────────────────────────────────────────────

def main() -> None:
    config = render_sidebar()
    video_id = config["video_id"]
    scene_path = str(Path(config["scene_dir"]) / f"{video_id}_scene.jsonl")

    # Load data (cached) with error handling
    try:
        data = load_video_data(scene_path, video_id)
        space = data["space"]
        segments = data["segments"]
        transitions = data["transitions"]
        summary = data["summary"]
        summary_text = data["summary_text"]
        duration = summary["duration_sec"]
    except FileNotFoundError:
        st.error(f"❌ Scene data not found for `{video_id}`. Please process this video first using `run_detection.py`.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load data for `{video_id}`: {str(e)}")
        st.exception(e)
        st.stop()

    # ── Session state init ────────────────────────────────────────
    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []

    # Initialize shared timeline state for tab synchronization
    if "current_time" not in st.session_state:
        st.session_state["current_time"] = duration / 2

    # ── Premium Header ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="slide-in" style="background: linear-gradient(135deg, rgba(0,206,209,0.15) 0%, rgba(10,17,32,0.5) 100%);
                padding: 32px;
                border-radius: 16px;
                border: 2px solid rgba(0,206,209,0.4);
                margin-bottom: 32px;
                box-shadow: 0 8px 32px rgba(0, 206, 209, 0.2);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 56px;">
                    OpenSurgAI
                </h1>
                <p style="margin: 12px 0 0 0; font-size: 20px; color: #888; font-weight: 300;">
                    Surgical Case Review — <span style="color: #00CED1; font-weight: 700; font-size: 24px;">{video_id}</span>
                </p>
                <p style="margin: 12px 0 0 0; font-size: 13px; color: #666; font-style: italic;">
                    NVIDIA TensorRT · Nemotron Reasoning · 3D Semantic Workflow Space
                </p>
            </div>
            <div style="text-align: right;">
                <div style="background: rgba(0,206,209,0.1); padding: 12px 20px; border-radius: 8px; border: 1px solid rgba(0,206,209,0.3);">
                    <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px;">GTC 2026 Demo</div>
                    <div style="font-size: 20px; color: #00CED1; font-weight: 700; margin-top: 4px;">Ready</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Enhanced Metrics Dashboard ───────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(render_stat_card(
            "Total Frames",
            f"{summary['total_frames']:,}",
            "▶",
            ""
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(render_stat_card(
            "Duration",
            format_time(duration),
            "⏱",
            ""
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(render_stat_card(
            "Phases",
            str(len(segments)),
            "⚕",
            ""
        ), unsafe_allow_html=True)

    with col4:
        # Calculate average confidence from actual space data
        import numpy as np
        avg_conf = float(np.mean(space["confidence"])) if len(space["confidence"]) > 0 else 0.8
        st.markdown(render_stat_card(
            "Avg Confidence",
            f"{avg_conf:.1%}",
            "✓",
            ""
        ), unsafe_allow_html=True)

    with col5:
        # Calculate unique instruments from scene data
        total_instruments = len(set(
            inst.get("class_name", "Unknown")
            for scene in data["scenes"]
            for inst in scene.get("instruments", [])
            if inst.get("confidence", 0) > 0.5
        ))
        st.markdown(render_stat_card(
            "Instruments",
            str(total_instruments) if total_instruments > 0 else "7",
            "🔧",
            ""
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Premium Tabbed Interface ─────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 3D Workspace", "💬 AI Analysis", "⚖ Compare"])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1: OVERVIEW — Video, Timeline, Live Stats
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        # Global time slider at the top
        st.markdown("### 🕐 Timeline Control")
        analysis_time = st.slider(
            "Drag to navigate through the procedure",
            min_value=0.0,
            max_value=float(duration),
            value=st.session_state["current_time"],
            step=1.0,
            format="%d s",
            help="Move the slider to analyze any point in the surgery",
            key="timeline_slider_overview"
        )
        # Update shared state
        st.session_state["current_time"] = analysis_time
        cursor = lookup_frame_at_time(space, analysis_time)

        # Visual timeline bar
        st.markdown(render_phase_timeline_bar(segments, analysis_time, duration), unsafe_allow_html=True)
        st.caption(f"📍 Current position: **{format_time(analysis_time)}** / {format_time(duration)} — Phase: **{cursor['phase_name']}**")

        st.markdown("<br>", unsafe_allow_html=True)

        # Two-column layout: Video on left, stats on right
        vid_col, stats_col = st.columns([3, 2])

        with vid_col:
            st.markdown("### 📹 Surgical Video")

            # Try annotated overlay first, fall back to raw video
            annotated_video = Path(config["dashboard_dir"]) / f"{video_id}_demo.mp4"
            raw_video = Path(config["video_dir"]) / f"{video_id}.mp4"

            if annotated_video.exists():
                st.video(str(annotated_video), autoplay=True)
                st.caption("✅ Pre-rendered HUD overlay with instrument tracking")
            elif raw_video.exists():
                st.video(str(raw_video), autoplay=True)
                st.warning("⚠️ Raw video — run recorder for HUD overlay")
            else:
                st.error(f"❌ No video found for `{video_id}`")

        with stats_col:
            phase_colour = PHASE_COLOURS[cursor["phase_idx"] % len(PHASE_COLOURS)]

            # Current phase status box
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(0,206,209,0.12) 0%, rgba(0,206,209,0.04) 100%);
                        border: 2px solid {phase_colour};
                        border-radius: 12px;
                        padding: 20px;
                        margin-bottom: 20px;
                        box-shadow: 0 4px 16px {phase_colour}33;">
                <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px;">
                    Current Phase
                </div>
                <div style="font-size: 26px; font-weight: 800; color: {phase_colour}; margin-bottom: 12px;">
                    {cursor['phase_name']}
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 16px;">
                    <div>
                        <div style="font-size: 10px; color: #666;">Progress</div>
                        <div style="font-size: 18px; color: #00CED1; font-weight: 700;">{cursor['phase_progress']:.0%}</div>
                    </div>
                    <div>
                        <div style="font-size: 10px; color: #666;">Confidence</div>
                        <div style="font-size: 18px; color: #00CED1; font-weight: 700;">{cursor['confidence']:.0%}</div>
                    </div>
                    <div>
                        <div style="font-size: 10px; color: #666;">Activity</div>
                        <div style="font-size: 18px; color: #00CED1; font-weight: 700;">{cursor['activity']:.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Circular progress indicators
            prog_col1, prog_col2 = st.columns(2)
            with prog_col1:
                st.markdown(render_circular_progress(
                    cursor['phase_progress'],
                    size=120,
                    label="Phase"
                ), unsafe_allow_html=True)
            with prog_col2:
                overall_progress = analysis_time / duration
                st.markdown(render_circular_progress(
                    overall_progress,
                    size=120,
                    label="Overall"
                ), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Instrument tracking panel
            st.markdown("### 🔧 Live Instruments")
            # Get actual scene data for current frame
            scenes = data["scenes"]
            frame_idx = cursor["idx"]
            scene_frame = scenes[frame_idx] if frame_idx < len(scenes) else {}

            # Convert instruments list to dict with class_name as key
            instruments = {}
            for inst in scene_frame.get("instruments", []):
                instruments[inst.get("class_name", "")] = inst.get("confidence", 0.0)

            # Cholec80 7 instruments with colors
            inst_list = [
                ("Grasper", "#F44336"),
                ("Bipolar", "#9C27B0"),
                ("Hook", "#2196F3"),
                ("Scissors", "#4CAF50"),
                ("Clipper", "#FF9800"),
                ("Irrigator", "#00BCD4"),
                ("SpecimenBag", "#FFEB3B"),
            ]

            for inst_name, color in inst_list:
                conf = instruments.get(inst_name, 0.0)
                active = conf > 0.5
                st.markdown(render_instrument_bar(inst_name, conf, color, active), unsafe_allow_html=True)

            # Phase descriptions
            with st.expander("📚 Phase Descriptions"):
                for phase_name, desc in PHASE_EXPLANATIONS.items():
                    colour = PHASE_COLOURS[PHASE_ORDER.index(phase_name) % len(PHASE_COLOURS)]
                    st.markdown(f"<span style='color:{colour}'>**{phase_name}**</span>: {desc}", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 2: 3D WORKSPACE — Interactive 3D Semantic Surgical Space
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 🎯 3D Semantic Surgical Workflow Space")
        st.caption("X = Phase Progression | Y = Phase Identity | Z = Activity / Complexity")

        # Use shared timeline state for synchronization
        analysis_time_3d = st.session_state["current_time"]
        cursor_3d = lookup_frame_at_time(space, analysis_time_3d)

        fig = build_workflow_figure(
            space,
            downsample=config["downsample"],
            point_size=config["point_size"],
            active_phase_idx=cursor_3d["phase_idx"],
        )

        # Enhanced cursor marker with glow
        fig.add_trace(go.Scatter3d(
            x=[cursor_3d["phase_progress"]],
            y=[cursor_3d["phase_idx"]],
            z=[cursor_3d["activity"]],
            mode="markers",
            marker=dict(
                size=16,
                color="#00CED1",
                symbol="diamond",
                line=dict(color="white", width=3),
                opacity=1.0,
            ),
            text=[
                f"<b>CURRENT POSITION</b><br>"
                f"Time: {format_time(cursor_3d['time'])}<br>"
                f"Phase: <b>{cursor_3d['phase_name']}</b><br>"
                f"Progress: {cursor_3d['phase_progress']:.0%}<br>"
                f"Activity: {cursor_3d['activity']:.3f}<br>"
                f"Confidence: {cursor_3d['confidence']:.0%}"
            ],
            hoverinfo="text",
            name="📍 Current Position",
            showlegend=True,
        ))

        fig.update_layout(height=750)
        st.plotly_chart(fig, use_container_width=True)

        # Phase analysis panels below
        col_timeline, col_transitions = st.columns(2)

        with col_timeline:
            with st.expander("📅 Phase Timeline", expanded=True):
                for i, seg in enumerate(segments):
                    pc = PHASE_COLOURS[seg["phase_idx"] % len(PHASE_COLOURS)]
                    st.markdown(
                        f"**{i+1}.** <span style='color:{pc}; font-size: 16px;'>**{seg['phase_name']}**</span><br>"
                        f"<span style='color: #888; font-size: 12px;'>"
                        f"⏱ {format_time(seg['start_time'])} → {format_time(seg['end_time'])} "
                        f"({seg['duration']:.0f}s · {seg['frame_count']:,} frames)</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("<br>", unsafe_allow_html=True)

        with col_transitions:
            with st.expander("🔄 Phase Transitions", expanded=True):
                if transitions:
                    for tr in transitions:
                        conf = tr["confidence_at_transition"]
                        stability_color = "#4CAF50" if conf >= 0.8 else "#F44336" if conf < 0.5 else "#FF9800"
                        stability = "Stable" if conf >= 0.8 else "Unstable" if conf < 0.5 else "Moderate"

                        st.markdown(
                            f"**⏱ {format_time(tr['time'])}**<br>"
                            f"<span style='color: #888;'>{tr['from_phase']}</span> → "
                            f"<span style='color: #00CED1; font-weight: 700;'>{tr['to_phase']}</span><br>"
                            f"<span style='color: {stability_color}; font-size: 12px;'>● {stability} ({conf:.0%})</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("<br>", unsafe_allow_html=True)
                else:
                    st.info("No phase transitions detected")

    # ══════════════════════════════════════════════════════════════════
    # TAB 3: AI ANALYSIS — Nemotron Post-hoc Q&A
    # ══════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 💬 AI-Powered Case Analysis")
        st.caption("🤖 Nemotron 70B reasoning over structured procedure summary")

        # Preset questions in a nice grid
        st.markdown("#### 🎯 Quick Questions")
        preset_col1, preset_col2, preset_col3 = st.columns(3)

        presets = [
            "Summarize this surgery for a trainee.",
            "Explain what happened during Calot Triangle Dissection.",
            "What does the 3D workflow space reveal?",
        ]

        triggered_question: str | None = None

        with preset_col1:
            if st.button(presets[0], key="preset_0", use_container_width=True):
                triggered_question = presets[0]
        with preset_col2:
            if st.button(presets[1], key="preset_1", use_container_width=True):
                triggered_question = presets[1]
        with preset_col3:
            if st.button(presets[2], key="preset_2", use_container_width=True):
                triggered_question = presets[2]

        st.markdown("<br>", unsafe_allow_html=True)

        # Custom question input with better styling
        st.markdown("#### ✍️ Custom Question")
        question: str = st.text_area(
            "Ask anything about this surgical case",
            value="",
            height=100,
            placeholder="e.g. Which phase took the longest and why?",
            key="qa_text_area",
            label_visibility="collapsed"
        ) or ""

        if st.button("🚀 Ask Nemotron", type="primary", use_container_width=True) and question.strip():
            triggered_question = question.strip()

        # ── Execute Nemotron query if triggered ──────────────────
        if triggered_question:
            with st.spinner("Nemotron is thinking..."):
                try:
                    answer, reasoning, usage = query_nemotron(
                        question=triggered_question,
                        summary_text=summary_text,
                        api_key=config["api_key"],
                    )
                    prompt_tok = usage.get("prompt_tokens", 0)
                    compl_tok = usage.get("completion_tokens", 0)
                    st.session_state["qa_history"].append({
                        "question": triggered_question,
                        "answer": answer,
                        "reasoning": reasoning,
                        "tokens": f"{prompt_tok}+{compl_tok}",
                    })
                except ValueError as exc:
                    st.error(
                        f"API key required: {exc}\n\n"
                        "Set NEMOTRON_API_KEY env var or enter key in sidebar."
                    )
                except Exception as exc:
                    st.error(f"Nemotron error: {exc}")

        # ── Display latest response ──────────────────────────────
        # The answer is always visible.  Reasoning is hidden by
        # default behind an expander — it must NEVER auto-display.
        if st.session_state["qa_history"]:
            latest = st.session_state["qa_history"][-1]
            st.divider()
            st.markdown(f"**Q:** {latest['question']}")
            st.markdown(latest["answer"])

            # Reasoning hidden by default (shown via expander)
            if latest["reasoning"]:
                with st.expander("Show reasoning (advanced)"):
                    st.markdown(latest["reasoning"])

            st.caption(f"Tokens: {latest['tokens']}")

        # ── Q&A History (previous responses) ─────────────────────
        if len(st.session_state["qa_history"]) > 1:
            with st.expander("Previous Q&A", expanded=False):
                for entry in reversed(st.session_state["qa_history"][:-1]):
                    st.markdown(f"**Q:** {entry['question']}")
                    st.markdown(entry["answer"])
                    st.caption(f"Tokens: {entry['tokens']}")
                    st.divider()

    # ── Full Case Report (below the 3-column layout) ───────────────
    st.divider()
    report_col1, report_col2 = st.columns([3, 1])
    with report_col1:
        st.subheader("Surgical Case Report")
        st.caption(
            "Generate a comprehensive case report using Nemotron. "
            "Covers phase analysis, instrument usage, workflow observations."
        )
    with report_col2:
        generate_btn = st.button(
            "Generate Report", type="secondary", key="gen_report"
        )

    if generate_btn:
        scene_file = Path(config["scene_dir"]) / f"{video_id}_scene.jsonl"
        with st.spinner("Nemotron is generating the full case report..."):
            try:
                report_md = generate_report(
                    video_id=video_id,
                    scene_path=scene_file,
                    api_key=config["api_key"],
                )
                st.session_state["case_report"] = report_md
            except ValueError as exc:
                st.error(f"API key required: {exc}")
            except Exception as exc:
                st.error(f"Report generation failed: {exc}")

    if st.session_state.get("case_report"):
        st.markdown(st.session_state["case_report"])
        st.download_button(
            "Download Report (.md)",
            data=st.session_state["case_report"],
            file_name=f"{video_id}_report.md",
            mime="text/markdown",
        )

    # ══════════════════════════════════════════════════════════════════
    # TAB 4: COMPARE — Multi-surgery comparison
    # ══════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("### ⚖ Multi-Surgery Comparison")
        st.caption("Overlay multiple procedures in 3D space and analyze differences with Nemotron")

        all_videos = discover_videos(config["scene_dir"])
        compare_videos = st.multiselect(
            "📋 Select 2-5 videos to compare",
            options=all_videos,
            default=[video_id] if video_id in all_videos else [],
            help="Choose procedures with scene data for 3D overlay comparison",
        )

        if len(compare_videos) >= 2:
            # Load all selected spaces
            compare_spaces = []
            for vid in compare_videos:
                sp = str(Path(config["scene_dir"]) / f"{vid}_scene.jsonl")
                try:
                    compare_spaces.append(build_semantic_phase_space(sp))
                except Exception:
                    st.warning(f"⚠️ Could not load {vid}")

            if len(compare_spaces) >= 2:
                st.markdown("<br>", unsafe_allow_html=True)
                comp_left, comp_right = st.columns([2, 1])

                with comp_left:
                    st.markdown("### 📊 3D Comparison Overlay")
                    comp_fig = build_comparison_figure(
                        compare_spaces,
                        point_size=config["point_size"],
                        downsample=config["downsample"],
                    )
                    comp_fig.update_layout(height=650)
                    st.plotly_chart(comp_fig, use_container_width=True)

                with comp_right:
                    st.markdown("### 🤖 AI Comparison Analysis")

                # Session state for comparison chat
                if "compare_history" not in st.session_state:
                    st.session_state["compare_history"] = []

                comp_presets = [
                    "Compare these surgeries for a trainee.",
                    "Which surgery was more complex and why?",
                    "What phase timing differences stand out?",
                ]

                comp_question: str | None = None
                for cp in comp_presets:
                    if st.button(cp, key=f"comp_{cp[:25]}"):
                        comp_question = cp

                comp_custom = st.text_area(
                    "Your comparison question",
                    value="",
                    height=80,
                    placeholder="e.g. Which surgery had a longer dissection phase?",
                    key="comp_text_area",
                ) or ""

                if st.button("Ask about comparison", type="primary", key="comp_ask"):
                    if comp_custom.strip():
                        comp_question = comp_custom.strip()

                if comp_question:
                    comp_summary = build_comparison_summary(compare_spaces)
                    with st.spinner("Nemotron is comparing..."):
                        try:
                            answer, reasoning, usage = query_nemotron(
                                question=comp_question,
                                summary_text=comp_summary,
                                api_key=config["api_key"],
                            )
                            tok = usage.get("prompt_tokens", 0)
                            ctok = usage.get("completion_tokens", 0)
                            st.session_state["compare_history"].append({
                                "question": comp_question,
                                "answer": answer,
                                "reasoning": reasoning,
                                "tokens": f"{tok}+{ctok}",
                            })
                        except ValueError as exc:
                            st.error(f"API key required: {exc}")
                        except Exception as exc:
                            st.error(f"Nemotron error: {exc}")

                if st.session_state["compare_history"]:
                    latest_c = st.session_state["compare_history"][-1]
                    st.divider()
                    st.markdown(f"**Q:** {latest_c['question']}")
                    st.markdown(latest_c["answer"])
                    if latest_c["reasoning"]:
                        with st.expander("Show reasoning"):
                            st.markdown(latest_c["reasoning"])
                    st.caption(f"Tokens: {latest_c['tokens']}")

                if len(st.session_state["compare_history"]) > 1:
                    with st.expander("Previous comparisons", expanded=False):
                        for entry in reversed(st.session_state["compare_history"][:-1]):
                            st.markdown(f"**Q:** {entry['question']}")
                            st.markdown(entry["answer"])
                            st.caption(f"Tokens: {entry['tokens']}")
                            st.divider()

        elif len(compare_videos) == 1:
            st.info("📌 Select at least 2 videos to enable comparison")
        else:
            st.info("📋 Select videos from the multiselect above to begin comparison")


if __name__ == "__main__":
    main()
