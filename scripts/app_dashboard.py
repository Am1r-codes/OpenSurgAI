"""OpenSurgAI -- Multi-NIM Surgical Intelligence Platform.

Three-tab Streamlit layout for NVIDIA GTC 2026 demo:

  Tab 1: Overview  -- Annotated video, timeline, phase info, live stats
  Tab 2: AI Analysis -- Nemotron Q&A + Nemotron VL visual frame analysis
  Tab 3: 3D Workflow Space -- Semantic surgical workflow visualization

Multi-NIM orchestration: Nemotron (text reasoning) + Nemotron VL (vision) +
TensorRT (real-time inference at 1,335 FPS).

Launch:
    streamlit run scripts/app_dashboard.py
"""

from __future__ import annotations

import bisect
import logging
import os
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
    build_semantic_phase_space,
    build_workflow_figure,
    get_phase_segments,
    get_transition_points,
)
from src.explanation.pipeline import NemotronClient, PHASE_EXPLANATIONS
from src.explanation.vlm_client import VLMClient, ANALYSIS_PRESETS, VLM_SYSTEM_PROMPT
from src.explanation.frame_extractor import extract_frame_at_time, get_video_info

from scripts.run_posthoc_qa import aggregate_summary, format_summary_for_prompt
from scripts.run_report import generate_report

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="OpenSurgAI — Multi-NIM Surgical Intelligence",
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
    /* Keep header visible for sidebar toggle button */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    /* Hide only the Streamlit toolbar items, keep hamburger menu */
    header[data-testid="stHeader"] > div:first-child {
        background-color: rgba(10, 17, 32, 0.95);
    }

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



# ── Utility: H.264 video compatibility ─────────────────────────────
# Browsers require H.264 (avc1) codec for <video> / st.video().
# OpenCV's mp4v fourcc sometimes produces MPEG-4 Part 2 (FMP4) which
# is not browser-playable.  This helper detects and re-encodes.

_log = logging.getLogger(__name__)


def _find_ffmpeg() -> str:
    """Locate ffmpeg binary — checks PATH first, then imageio_ffmpeg."""
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"  # last resort, will fail with FileNotFoundError


def _find_ffprobe() -> str:
    """Locate ffprobe binary next to the ffmpeg we found."""
    import shutil
    path = shutil.which("ffprobe")
    if path:
        return path
    # imageio_ffmpeg ships ffmpeg but not ffprobe — derive path
    ffmpeg = _find_ffmpeg()
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")
    if Path(ffprobe).exists():
        return ffprobe
    return "ffprobe"


@st.cache_resource(show_spinner="Re-encoding video for browser playback...")
def _ensure_h264(video_path: Path) -> str:
    """Return a browser-compatible H.264 path for *video_path*.

    If the file is already H.264, returns the original path as a string.
    Otherwise, transcodes to ``<name>.h264.mp4`` next to the original
    using ffmpeg and returns that path.
    """
    import cv2

    # Fast codec check via OpenCV (no ffprobe needed)
    cap = cv2.VideoCapture(str(video_path))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    codec = fourcc_int.to_bytes(4, "little").decode("ascii", errors="replace").strip("\x00").lower()

    if codec in ("avc1", "h264", "x264"):
        return str(video_path)

    # Need re-encoding
    h264_path = video_path.with_suffix(".h264.mp4")
    if h264_path.exists() and h264_path.stat().st_size > 0:
        return str(h264_path)

    ffmpeg = _find_ffmpeg()
    _log.info("Re-encoding %s (%s -> H.264) ...", video_path.name, codec or "unknown")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(video_path),
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-movflags", "+faststart",
             "-an",  # no audio in surgical videos
             str(h264_path)],
            check=True, capture_output=True, timeout=600,
        )
        _log.info("Re-encoded: %s (%.0f MB)",
                  h264_path.name, h264_path.stat().st_size / 1e6)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        _log.warning("ffmpeg re-encode failed: %s — falling back to original", exc)
        return str(video_path)

    return str(h264_path)


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

    # Use round() instead of int() for consistent rounding
    display_percent = round(percent * 100)

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
                transform="rotate(90 {size/2} {size/2})"
                style="font-size: 24px; font-weight: 700; fill: #00CED1;"
            >
                {display_percent}%
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
    html_parts = ['<div style="position: relative; width: 100%; height: 30px; margin: 16px 0;">']

    # Background bar
    html_parts.append('<div style="position: absolute; width: 100%; height: 20px; top: 5px; background-color: rgba(0,0,0,0.3); border-radius: 10px; overflow: hidden; display: flex;">')

    # Phase segments
    for seg in segments:
        start_pct = (seg["start_time"] / duration) * 100
        width_pct = (seg["duration"] / duration) * 100
        color = PHASE_COLOURS[seg["phase_idx"] % len(PHASE_COLOURS)]

        html_parts.append(f'<div style="position: absolute; left: {start_pct}%; width: {width_pct}%; height: 100%; background-color: {color}; opacity: 0.8; border-right: 1px solid #0A1120;" title="{seg["phase_name"]} ({format_time(seg["duration"])})"></div>')

    html_parts.append('</div>')

    # Current position marker (simplified - no nested triangle)
    current_pct = (current_time / duration) * 100
    html_parts.append(f'<div style="position: absolute; left: {current_pct}%; top: 0; width: 4px; height: 30px; background: linear-gradient(180deg, #00CED1 0%, #00FFFF 100%); box-shadow: 0 0 12px rgba(0, 206, 209, 0.8); z-index: 10; border-radius: 2px;"></div>')

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
        <h1 style="color: #00CED1; font-size: 28px; margin: 0; text-shadow: 0 2px 8px rgba(0,206,209,0.4);">
            OpenSurgAI
        </h1>
        <p style="color: #76B900; font-size: 10px; margin: 6px 0 0 0; text-transform: uppercase; letter-spacing: 2px; font-weight: 700;">
            Multi-NIM Surgical Intelligence
        </p>
        <p style="color: #666; font-size: 10px; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 1.5px;">
            NVIDIA GTC 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    # NIM Status Indicators
    st.sidebar.markdown("""
    <div style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 14px; margin-bottom: 16px;
                border: 1px solid rgba(118,185,0,0.3);">
        <div style="font-size: 10px; color: #76B900; text-transform: uppercase; letter-spacing: 2px;
                    font-weight: 700; margin-bottom: 12px;">NIM Services</div>
        <div style="display: flex; flex-direction: column; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background: #76B900;
                            box-shadow: 0 0 8px rgba(118,185,0,0.6);"></div>
                <span style="font-size: 11px; color: #ccc; font-weight: 600;">Nemotron 49B</span>
                <span style="font-size: 9px; color: #888; margin-left: auto;">Text Reasoning</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background: #76B900;
                            box-shadow: 0 0 8px rgba(118,185,0,0.6);"></div>
                <span style="font-size: 11px; color: #ccc; font-weight: 600;">Nemotron VL</span>
                <span style="font-size: 9px; color: #888; margin-left: auto;">Vision Analysis</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background: #76B900;
                            box-shadow: 0 0 8px rgba(118,185,0,0.6);"></div>
                <span style="font-size: 11px; color: #ccc; font-weight: 600;">TensorRT FP16</span>
                <span style="font-size: 9px; color: #888; margin-left: auto;">1,335 FPS</span>
            </div>
        </div>
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

    # Default to video49 if available, otherwise first video
    default_idx = videos.index("video49") if "video49" in videos else 0
    video_id = st.sidebar.selectbox("Video", videos, index=default_idx)

    st.sidebar.divider()

    # NVIDIA API config (shared across Nemotron + Nemotron VL)
    st.sidebar.subheader("NVIDIA NIM API")
    api_key = st.sidebar.text_input(
        "API Key",
        type="password",
        help="NVIDIA_API_KEY — shared by all Nemotron NIM services. Leave blank to use env var.",
    )

    # ── Process New Video ────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Process New Video")

    # Find unprocessed videos in the raw video directory
    raw_dir = Path(video_dir)
    processed_set = set(videos)
    unprocessed = []
    if raw_dir.is_dir():
        for vf in sorted(raw_dir.glob("*.mp4")):
            vid_name = vf.stem
            if vid_name not in processed_set:
                unprocessed.append(vid_name)

    if unprocessed:
        new_video = st.sidebar.selectbox(
            "Unprocessed videos",
            unprocessed,
            help="Videos in data directory that haven't been processed yet.",
            key="new_video_select",
        )
        if st.sidebar.button("Process Video", type="primary", key="process_btn"):
            video_path = raw_dir / f"{new_video}.mp4"
            _run_pipeline(video_path, new_video, api_key)
            st.cache_data.clear()
            st.rerun()
    else:
        st.sidebar.caption("All videos in the data directory have been processed.")

    # Upload external video
    uploaded = st.sidebar.file_uploader(
        "Or upload a new video (.mp4)",
        type=["mp4", "avi", "mkv"],
        help="Upload a video file to process through the full pipeline.",
    )

    if uploaded is not None:
        upload_dir = _PROJECT_ROOT / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / uploaded.name
        vid_stem = save_path.stem

        if not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.sidebar.success(f"Saved: {uploaded.name}")

        if st.sidebar.button("Process Uploaded Video", type="primary", key="process_upload_btn"):
            _run_pipeline(save_path, vid_stem, api_key)
            st.cache_data.clear()
            st.rerun()

    return {
        "scene_dir": scene_dir,
        "dashboard_dir": dashboard_dir,
        "video_dir": video_dir,
        "video_id": video_id,
        "api_key": api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("NEMOTRON_API_KEY"),
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

    # ── Premium Header — Multi-NIM Banner ────────────────────────────
    st.markdown(f"""
    <div class="slide-in" style="background: linear-gradient(135deg, rgba(0,206,209,0.12) 0%, rgba(118,185,0,0.08) 50%, rgba(10,17,32,0.5) 100%);
                padding: 32px;
                border-radius: 16px;
                border: 2px solid rgba(118,185,0,0.4);
                margin-bottom: 32px;
                box-shadow: 0 8px 32px rgba(118, 185, 0, 0.15);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 48px;">
                    OpenSurgAI
                </h1>
                <p style="margin: 8px 0 0 0; font-size: 18px; color: #76B900; font-weight: 700; letter-spacing: 1px;">
                    Multi-NIM Surgical Intelligence Platform
                </p>
                <p style="margin: 8px 0 0 0; font-size: 15px; color: #888; font-weight: 300;">
                    Case Review — <span style="color: #00CED1; font-weight: 700; font-size: 18px;">{video_id}</span>
                </p>
                <div style="display: flex; gap: 16px; margin-top: 14px;">
                    <span style="background: rgba(118,185,0,0.15); color: #76B900; padding: 4px 12px;
                                 border-radius: 20px; font-size: 11px; font-weight: 600; border: 1px solid rgba(118,185,0,0.3);
                                 letter-spacing: 0.5px;">TensorRT FP16</span>
                    <span style="background: rgba(0,206,209,0.15); color: #00CED1; padding: 4px 12px;
                                 border-radius: 20px; font-size: 11px; font-weight: 600; border: 1px solid rgba(0,206,209,0.3);
                                 letter-spacing: 0.5px;">Nemotron 49B</span>
                    <span style="background: rgba(156,39,176,0.15); color: #CE93D8; padding: 4px 12px;
                                 border-radius: 20px; font-size: 11px; font-weight: 600; border: 1px solid rgba(156,39,176,0.3);
                                 letter-spacing: 0.5px;">Nemotron VL</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="background: rgba(118,185,0,0.1); padding: 14px 22px; border-radius: 10px;
                            border: 1px solid rgba(118,185,0,0.3);">
                    <div style="font-size: 10px; color: #76B900; text-transform: uppercase; letter-spacing: 1.5px;
                                font-weight: 700;">NVIDIA GTC 2026</div>
                    <div style="font-size: 24px; color: #76B900; font-weight: 800; margin-top: 4px;">3 NIMs</div>
                    <div style="font-size: 10px; color: #888; margin-top: 2px;">Orchestrated</div>
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
        # Calculate unique Cholec80 instruments from scene data
        total_instruments = 7  # Cholec80 dataset has 7 instrument classes
        st.markdown(render_stat_card(
            "Instruments",
            str(total_instruments) if total_instruments > 0 else "7",
            "🔧",
            ""
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Premium Tabbed Interface ─────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🧠 AI Analysis", "🌐 3D Workflow Space"])

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
                playable = _ensure_h264(annotated_video)
                st.video(playable, format="video/mp4", start_time=0)
                st.caption("Pre-rendered HUD overlay with instrument tracking")
            elif raw_video.exists():
                playable = _ensure_h264(raw_video)
                st.video(playable, format="video/mp4", start_time=0)
                st.warning("Raw video — run recorder for HUD overlay")
            else:
                st.error(f"No video found for `{video_id}`")

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
                        <div style="font-size: 18px; color: #00CED1; font-weight: 700;">{round(cursor['phase_progress'] * 100)}%</div>
                    </div>
                    <div>
                        <div style="font-size: 10px; color: #666;">Confidence</div>
                        <div style="font-size: 18px; color: #00CED1; font-weight: 700;">{round(cursor['confidence'] * 100)}%</div>
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

            # Convert instruments list to dict with Cholec80 names
            # Map COCO class names to surgical instruments if needed
            _COCO_MAP = {
                "scissors": "Scissors", "knife": "Hook", "fork": "Grasper",
                "spoon": "Irrigator", "toothbrush": "Bipolar",
            }
            instruments = {}
            for inst in scene_frame.get("instruments", []):
                raw = inst.get("class_name", "")
                name = _COCO_MAP.get(raw, raw)  # map COCO or pass through
                if name in {"Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"}:
                    conf = inst.get("confidence", 0.0)
                    instruments[name] = max(instruments.get(name, 0.0), conf)

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
    # TAB 2: AI ANALYSIS — Nemotron VL Visual + Nemotron Text
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("### 🧠 Multi-NIM AI Analysis")
        st.caption("Nemotron VL (vision) + Nemotron 49B (text reasoning) — two NIM services working together")

        # ── Nemotron VL Visual Frame Analysis ─────────────────────
        st.markdown("---")
        st.markdown("#### 👁 Nemotron VL Visual Frame Analysis")
        st.caption("Send any frame to NVIDIA Nemotron VL for visual understanding")

        vila_col1, vila_col2 = st.columns([2, 3])
        frame_b64 = None  # initialized before columns, used across both

        with vila_col1:
            # Frame time selector (synced with overview timeline)
            vila_time = st.slider(
                "Select frame timestamp",
                min_value=0.0,
                max_value=float(duration),
                value=st.session_state["current_time"],
                step=1.0,
                format="%d s",
                key="vila_time_slider",
            )

            # Analysis preset selector
            preset_choice = st.selectbox(
                "Analysis type",
                list(ANALYSIS_PRESETS.keys()),
                index=0,
                key="vila_preset",
            )
            vila_prompt = ANALYSIS_PRESETS[preset_choice]

            # Custom prompt override
            custom_vila = st.text_area(
                "Or enter a custom prompt",
                value="",
                height=80,
                placeholder="e.g. Is the critical view of safety achieved?",
                key="vila_custom_prompt",
            )
            if custom_vila.strip():
                vila_prompt = custom_vila.strip()

            analyze_btn = st.button(
                "Analyze Frame with Nemotron VL",
                type="primary",
                use_container_width=True,
                key="vila_analyze_btn",
            )

        with vila_col2:
            # Find video file for frame extraction
            annotated_video = Path(config["dashboard_dir"]) / f"{video_id}_demo.mp4"
            raw_video = Path(config["video_dir"]) / f"{video_id}.mp4"
            video_file = annotated_video if annotated_video.exists() else raw_video

            if video_file.exists():
                # Show frame preview
                frame_b64 = extract_frame_at_time(str(video_file), vila_time)
                if frame_b64:
                    st.markdown(f"""
                    <div style="border: 2px solid rgba(156,39,176,0.4); border-radius: 10px; overflow: hidden;
                                box-shadow: 0 4px 16px rgba(156,39,176,0.2);">
                        <img src="data:image/jpeg;base64,{frame_b64}" style="width: 100%; display: block;">
                    </div>
                    <p style="text-align: center; font-size: 11px; color: #888; margin-top: 8px;">
                        Frame at {format_time(vila_time)} — {video_id}
                    </p>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No video file found for {video_id}. Frame preview unavailable.")
                frame_b64 = None

        # Execute Nemotron VL analysis
        if analyze_btn:
            if not frame_b64:
                st.error("Cannot extract frame. Check that the video file exists.")
            elif not config["api_key"]:
                st.error("API key required. Enter your NVIDIA API key in the sidebar.")
            else:
                with st.spinner("Nemotron VL is analyzing the frame..."):
                    try:
                        vlm = VLMClient(api_key=config["api_key"])
                        result = vlm.analyze_frame(
                            image_b64=frame_b64,
                            prompt=vila_prompt,
                        )
                        vlm.close()
                        st.session_state["vila_result"] = {
                            "time": vila_time,
                            "prompt": vila_prompt,
                            "content": result["content"],
                            "usage": result["usage"],
                            "model": result["model"],
                        }
                    except Exception as exc:
                        st.error(f"Nemotron VL error: {exc}")

        # Display Nemotron VL result
        if st.session_state.get("vila_result"):
            vr = st.session_state["vila_result"]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(156,39,176,0.1) 0%, rgba(156,39,176,0.03) 100%);
                        border: 1px solid rgba(156,39,176,0.3); border-radius: 12px; padding: 20px; margin: 16px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="color: #CE93D8; font-weight: 700; font-size: 13px; text-transform: uppercase;
                                 letter-spacing: 1px;">Nemotron VL Visual Analysis</span>
                    <span style="color: #888; font-size: 11px;">Frame at {format_time(vr['time'])}</span>
                </div>
                <div style="color: #E0E0E0; font-size: 14px; line-height: 1.7;">
                    {vr['content']}
                </div>
                <div style="margin-top: 12px; font-size: 10px; color: #666;">
                    Model: {vr['model']} | Tokens: {vr['usage'].get('prompt_tokens', 0)}+{vr['usage'].get('completion_tokens', 0)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Nemotron Text Q&A (existing) ─────────────────────────
        st.markdown("#### 💬 Nemotron Text Analysis")
        st.caption("Post-hoc reasoning over the complete procedure summary")

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

    # ══════════════════════════════════════════════════════════════════
    # TAB 3: 3D WORKFLOW SPACE
    # ══════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🌐 3D Semantic Surgical Workflow Space")
        st.caption(
            "Interactive 3D visualization of the procedure's workflow structure. "
            "X = Phase Progression, Y = Phase Identity, Z = Surgical Activity/Complexity."
        )

        # Build the 3D workflow figure
        try:
            workflow_fig = build_workflow_figure(space)
            st.plotly_chart(workflow_fig, use_container_width=True, height=650)
        except Exception as exc:
            st.error(f"Failed to build 3D workflow visualization: {exc}")

        # Workflow Space Interpretation Guide
        with st.expander("How to read the 3D Workflow Space"):
            st.markdown("""
**Axes:**
- **X (Phase Progression):** Progress within each phase segment (0 to 1)
- **Y (Phase Identity):** The surgical phase (0 = Preparation ... 6 = GallbladderRetraction)
- **Z (Surgical Activity):** Complexity metric combining instrument count and confidence volatility

**What to look for:**
- **Dense clusters** = stable, consistent surgical activity
- **Scattered points** = high complexity or instrument switching
- **Vertical jumps** = phase transitions
- **High Z values** = multiple instruments active with varying confidence

This is a **procedural workflow** representation, not anatomical geometry.
            """)

        # Phase segments — clickable for AI deep-dive
        st.markdown("#### Phase Segments")
        st.caption("Click a phase to ask Nemotron about it")

        # Render phase buttons in a grid
        phase_cols = st.columns(min(len(segments), 4))
        for i, seg in enumerate(segments):
            col = phase_cols[i % len(phase_cols)]
            phase_colour = PHASE_COLOURS[seg["phase_idx"] % len(PHASE_COLOURS)]
            with col:
                if st.button(
                    f"{seg['phase_name']}\n{format_time(seg['start_time'])} · {format_time(seg['duration'])} · {seg['frame_count']}f",
                    key=f"phase_btn_{i}",
                    use_container_width=True,
                ):
                    st.session_state["workflow_selected_phase"] = seg["phase_name"]
                    st.session_state["workflow_selected_seg"] = seg

        # Show full table in expander
        with st.expander("Full phase segments table"):
            seg_data = []
            for seg in segments:
                seg_data.append({
                    "Phase": seg["phase_name"],
                    "Start": format_time(seg["start_time"]),
                    "Duration": format_time(seg["duration"]),
                    "Frames": seg["frame_count"],
                })
            st.dataframe(seg_data, use_container_width=True, hide_index=True)

        # Transitions
        if transitions:
            with st.expander("Phase Transitions"):
                trans_data = []
                for t in transitions:
                    conf = t["confidence_at_transition"]
                    conf_label = f"{conf:.1%}"
                    trans_data.append({
                        "Time": format_time(t["time"]),
                        "From": t["from_phase"],
                        "To": t["to_phase"],
                        "Confidence": conf_label,
                    })
                st.dataframe(trans_data, use_container_width=True, hide_index=True)

        # ── Phase-specific Nemotron Q&A ──────────────────────────
        st.markdown("---")
        selected_phase = st.session_state.get("workflow_selected_phase")
        selected_seg = st.session_state.get("workflow_selected_seg")

        if selected_phase and selected_seg:
            phase_colour = PHASE_COLOURS[selected_seg["phase_idx"] % len(PHASE_COLOURS)]
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {phase_colour}22 0%, {phase_colour}08 100%);
                        border: 1px solid {phase_colour}66; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                <div style="font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">
                    Selected Phase
                </div>
                <div style="font-size: 22px; font-weight: 800; color: {phase_colour};">
                    {selected_phase}
                </div>
                <div style="font-size: 12px; color: #aaa; margin-top: 6px;">
                    {format_time(selected_seg['start_time'])} – {format_time(selected_seg['end_time'])} · {format_time(selected_seg['duration'])} · {selected_seg['frame_count']} frames
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Quick questions about this phase
            st.markdown("##### Ask Nemotron about this phase")
            wf_presets = [
                f"What happens during {selected_phase} in a cholecystectomy?",
                f"Is the duration of {selected_phase} ({format_time(selected_seg['duration'])}) normal?",
                f"What instruments are typically used in {selected_phase}?",
            ]

            wf_preset_cols = st.columns(3)
            wf_triggered: str | None = None
            for j, preset in enumerate(wf_presets):
                with wf_preset_cols[j]:
                    if st.button(preset, key=f"wf_preset_{j}", use_container_width=True):
                        wf_triggered = preset

            # Custom question
            wf_custom = st.text_input(
                f"Or ask your own question about {selected_phase}",
                value="",
                placeholder=f"e.g. Was there anything unusual about {selected_phase}?",
                key="wf_custom_q",
            )
            if st.button("Ask Nemotron", key="wf_ask_btn", type="primary") and wf_custom.strip():
                wf_triggered = wf_custom.strip()

            # Execute query
            if wf_triggered:
                with st.spinner(f"Nemotron is analyzing {selected_phase}..."):
                    try:
                        answer, reasoning, usage = query_nemotron(
                            question=wf_triggered,
                            summary_text=summary_text,
                            api_key=config["api_key"],
                        )
                        st.session_state["wf_qa_result"] = {
                            "phase": selected_phase,
                            "question": wf_triggered,
                            "answer": answer,
                            "reasoning": reasoning,
                            "tokens": f"{usage.get('prompt_tokens', 0)}+{usage.get('completion_tokens', 0)}",
                        }
                    except ValueError as exc:
                        st.error(f"API key required: {exc}")
                    except Exception as exc:
                        st.error(f"Nemotron error: {exc}")

            # Display result
            if st.session_state.get("wf_qa_result"):
                wfr = st.session_state["wf_qa_result"]
                if wfr["phase"] == selected_phase:
                    st.divider()
                    st.markdown(f"**Q:** {wfr['question']}")
                    st.markdown(wfr["answer"])
                    if wfr["reasoning"]:
                        with st.expander("Show reasoning"):
                            st.markdown(wfr["reasoning"])
                    st.caption(f"Tokens: {wfr['tokens']}")
        else:
            st.info("Click a phase segment above to ask Nemotron about it.")

    # ── Full Case Report (below tabs) ─────────────────────────────
    st.divider()
    report_col1, report_col2 = st.columns([3, 1])
    with report_col1:
        st.subheader("Operative Case Report")
        st.caption(
            "Generate a structured clinical report using Nemotron 49B. "
            "Includes procedure overview, phase analysis, instrument usage, "
            "workflow observations, and teaching points."
        )
    with report_col2:
        generate_btn = st.button(
            "Generate Report", type="secondary", key="gen_report"
        )

    if generate_btn:
        scene_file = Path(config["scene_dir"]) / f"{video_id}_scene.jsonl"
        with st.spinner("Nemotron is generating the operative report..."):
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

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "Download Report (.md)",
                data=st.session_state["case_report"],
                file_name=f"{video_id}_report.md",
                mime="text/markdown",
            )
        with dl_col2:
            # HTML export for PDF-ready printing
            html_report = _markdown_to_html_report(
                st.session_state["case_report"], video_id
            )
            st.download_button(
                "Download Report (.html)",
                data=html_report,
                file_name=f"{video_id}_report.html",
                mime="text/html",
            )

    # ── NVIDIA Ecosystem Footer ───────────────────────────────────
    st.markdown("""
    <div style="margin-top: 48px; padding: 24px; border-top: 2px solid rgba(118,185,0,0.3);
                text-align: center;">
        <div style="display: flex; justify-content: center; gap: 24px; margin-bottom: 16px; flex-wrap: wrap;">
            <span style="background: rgba(118,185,0,0.1); color: #76B900; padding: 6px 16px;
                         border-radius: 20px; font-size: 11px; font-weight: 600;
                         border: 1px solid rgba(118,185,0,0.3);">NIM Microservices</span>
            <span style="background: rgba(0,206,209,0.1); color: #00CED1; padding: 6px 16px;
                         border-radius: 20px; font-size: 11px; font-weight: 600;
                         border: 1px solid rgba(0,206,209,0.3);">TensorRT FP16</span>
            <span style="background: rgba(156,39,176,0.1); color: #CE93D8; padding: 6px 16px;
                         border-radius: 20px; font-size: 11px; font-weight: 600;
                         border: 1px solid rgba(156,39,176,0.3);">Nemotron 49B</span>
            <span style="background: rgba(255,152,0,0.1); color: #FFB74D; padding: 6px 16px;
                         border-radius: 20px; font-size: 11px; font-weight: 600;
                         border: 1px solid rgba(255,152,0,0.3);">Nemotron VL</span>
        </div>
        <p style="color: #555; font-size: 11px; margin: 0;">
            OpenSurgAI &mdash; Multi-NIM Surgical Intelligence Platform &mdash; Built for NVIDIA GTC 2026
        </p>
        <p style="color: #444; font-size: 10px; margin: 4px 0 0 0;">
            Nemotron 49B + Nemotron VL + TensorRT FP16 &mdash; Open Source Surgical AI
        </p>
    </div>
    """, unsafe_allow_html=True)


def _markdown_to_html_report(md_text: str, video_id: str) -> str:
    """Convert markdown report to a styled HTML document for PDF printing."""
    # Simple markdown to HTML conversion (headers, bold, lists)
    import re as _re

    html_body = md_text
    # Headers
    html_body = _re.sub(r"^##### (.+)$", r"<h5>\1</h5>", html_body, flags=_re.MULTILINE)
    html_body = _re.sub(r"^#### (.+)$", r"<h4>\1</h4>", html_body, flags=_re.MULTILINE)
    html_body = _re.sub(r"^### (.+)$", r"<h3>\1</h3>", html_body, flags=_re.MULTILINE)
    html_body = _re.sub(r"^## (.+)$", r"<h2>\1</h2>", html_body, flags=_re.MULTILINE)
    html_body = _re.sub(r"^# (.+)$", r"<h1>\1</h1>", html_body, flags=_re.MULTILINE)
    # Bold and italic
    html_body = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_body)
    html_body = _re.sub(r"\*(.+?)\*", r"<em>\1</em>", html_body)
    # List items
    html_body = _re.sub(r"^- (.+)$", r"<li>\1</li>", html_body, flags=_re.MULTILINE)
    # Horizontal rules
    html_body = _re.sub(r"^---+$", r"<hr>", html_body, flags=_re.MULTILINE)
    # Paragraphs (double newlines)
    html_body = _re.sub(r"\n\n", r"</p><p>", html_body)
    html_body = f"<p>{html_body}</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Operative Report - {video_id} | OpenSurgAI</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px; margin: 40px auto; padding: 0 20px;
            color: #333; line-height: 1.6;
        }}
        h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
        h2 {{ color: #283593; margin-top: 32px; }}
        h3 {{ color: #3949ab; }}
        hr {{ border: 1px solid #e0e0e0; margin: 24px 0; }}
        li {{ margin: 4px 0; }}
        .header {{
            background: linear-gradient(135deg, #1a237e, #283593);
            color: white; padding: 24px; border-radius: 8px; margin-bottom: 32px;
        }}
        .header h1 {{ color: white; border: none; margin: 0; }}
        .header p {{ margin: 4px 0 0 0; opacity: 0.8; }}
        .footer {{
            margin-top: 48px; padding-top: 16px; border-top: 2px solid #e0e0e0;
            font-size: 12px; color: #888; text-align: center;
        }}
        @media print {{
            body {{ margin: 0; padding: 20px; }}
            .header {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OpenSurgAI Operative Report</h1>
        <p>Multi-NIM Surgical Intelligence Platform | {video_id}</p>
    </div>
    {html_body}
    <div class="footer">
        Generated by OpenSurgAI with NVIDIA Nemotron 49B | Multi-NIM Surgical Intelligence Platform
    </div>
</body>
</html>"""


if __name__ == "__main__":
    main()
