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

  WHY AUTO-SYNC:
    A virtual playback clock (time.monotonic-based) provides approximate
    synchronisation between video playback and UI state.  Exact sync is
    not possible via st.video; approximate sync is acceptable.  The page
    auto-reruns at ~1.5 s intervals to advance the virtual clock.

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
import sys
import time
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
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
from scripts.run_posthoc_qa import aggregate_summary, format_summary_for_prompt
from scripts.run_report import generate_report

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="OpenSurgAI — Surgical Case Review",
    page_icon=":stethoscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────

# How often the page auto-reruns to advance the virtual clock.
# Lower = smoother updates but more Plotly redraws.
# Higher = less flickering but coarser time tracking.
_REFRESH_INTERVAL_SEC = 1.5


# ── Utility: time display formatting ────────────────────────────────
# All internal computations remain in seconds.
# This is DISPLAY-ONLY formatting — used everywhere the user sees time.

def format_time(seconds: float) -> str:
    """Format seconds as m:ss for display."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


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
# Split Nemotron output into ANSWER (always visible) and REASONING
# (hidden behind an expander by default).

def split_answer_reasoning(text: str) -> tuple[str, str]:
    """Split Nemotron response into (answer, reasoning).

    Reasoning is hidden by default in the UI to keep the interface
    calm and focused on the educational answer.
    """
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


# ── Sidebar ──────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar controls and return config."""
    st.sidebar.title("OpenSurgAI")
    st.sidebar.caption("Surgical Case Review — NVIDIA GTC Demo")

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

    # Load data (cached)
    data = load_video_data(scene_path, video_id)
    space = data["space"]
    segments = data["segments"]
    transitions = data["transitions"]
    summary = data["summary"]
    summary_text = data["summary_text"]
    duration = summary["duration_sec"]

    # ── Virtual playback clock ────────────────────────────────────
    # Approximate synchronisation: a monotonic clock estimates where
    # the video is.  This does NOT read actual video playback time
    # (not possible via st.video).  Approximate sync is acceptable
    # and documented.  The page auto-reruns at _REFRESH_INTERVAL_SEC
    # intervals to advance the clock.
    if "playback_start" not in st.session_state:
        st.session_state["playback_start"] = time.monotonic()
        st.session_state["playing"] = True

    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []

    elapsed = time.monotonic() - st.session_state["playback_start"]
    elapsed = min(elapsed, duration)  # clamp to video duration

    if elapsed >= duration:
        st.session_state["playing"] = False

    cursor = lookup_frame_at_time(space, elapsed)

    # ── Header ───────────────────────────────────────────────────────
    st.title(f"Surgical Case Review — {video_id}")
    st.caption(
        "3D Semantic Surgical Workflow Space: represents procedural structure "
        "and activity, **not** anatomical geometry or spatial reconstruction."
    )

    # ── Metrics row ──────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Frames", f"{summary['total_frames']:,}")
    m2.metric("Duration", format_time(duration))
    m3.metric("Phase Segments", len(segments))
    m4.metric("Transitions", len(transitions))

    # ── Three-column layout: LEFT | CENTER | RIGHT ───────────────────
    left_col, center_col, right_col = st.columns([1, 2, 1])

    # ── LEFT PANEL: Annotated Video + Auto-synced Phase Info ─────────
    # The annotated overlay video is the PRIMARY visual context.
    # Detection boxes, segmentation masks, and phase labels are
    # PRE-RENDERED by the dashboard recorder.  No real-time inference
    # runs inside this UI.
    with left_col:
        st.subheader("Annotated Video")

        # Try annotated overlay first, fall back to raw video
        annotated_video = Path(config["dashboard_dir"]) / f"{video_id}_demo.mp4"
        raw_video = Path(config["video_dir"]) / f"{video_id}.mp4"

        if annotated_video.exists():
            st.video(str(annotated_video), autoplay=True)
            st.caption(
                "Pre-rendered overlay — detections and phase labels baked in."
            )
        elif raw_video.exists():
            st.video(str(raw_video), autoplay=True)
            st.caption(
                "Raw video — run the recorder to generate annotated overlay."
            )
        else:
            st.info(
                f"No video found for `{video_id}`.\n\n"
                "Check directory paths in the sidebar."
            )

        # Current position (auto-synced to virtual playback clock)
        phase_colour = PHASE_COLOURS[cursor["phase_idx"] % len(PHASE_COLOURS)]

        st.markdown("#### Current Position")
        st.metric("Time", format_time(cursor["time"]))
        st.markdown(
            f"**Phase:** <span style='color:{phase_colour}'>"
            f"**{cursor['phase_name']}**</span>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{cursor['confidence']:.3f}")
        st.metric("Instruments", cursor["instrument_count"])
        st.metric("Progress", f"{cursor['phase_progress']:.0%}")
        st.metric("Activity", f"{cursor['activity']:.3f}")

        # Phase descriptions (collapsible)
        with st.expander("Phase Descriptions", expanded=False):
            for phase_name, desc in PHASE_EXPLANATIONS.items():
                colour = PHASE_COLOURS[
                    PHASE_ORDER.index(phase_name) % len(PHASE_COLOURS)
                ]
                st.markdown(
                    f"<span style='color:{colour}'>**{phase_name}**</span>: "
                    f"{desc}",
                    unsafe_allow_html=True,
                )

    # ── CENTER PANEL: 3D Semantic Surgical Workflow Space ─────────────
    # This visualisation represents procedural structure and activity,
    # not anatomical geometry or spatial reconstruction.
    # The cursor auto-updates based on the virtual playback clock.
    with center_col:
        st.subheader("3D Semantic Surgical Workflow Space")
        st.caption(
            "X = Phase Progression | Y = Phase Identity | "
            "Z = Activity / Complexity"
        )

        fig = build_workflow_figure(
            space,
            downsample=config["downsample"],
            point_size=config["point_size"],
            active_phase_idx=cursor["phase_idx"],
        )

        # Add cursor marker at the current estimated position
        import plotly.graph_objects as go
        fig.add_trace(go.Scatter3d(
            x=[cursor["phase_progress"]],
            y=[cursor["phase_idx"]],
            z=[cursor["activity"]],
            mode="markers",
            marker=dict(
                size=14,
                color="yellow",
                symbol="cross",
                line=dict(color="white", width=2),
            ),
            text=[
                f"NOW: {format_time(cursor['time'])}<br>"
                f"Phase: {cursor['phase_name']}<br>"
                f"Progress: {cursor['phase_progress']:.0%}<br>"
                f"Activity: {cursor['activity']:.3f}<br>"
                f"Confidence: {cursor['confidence']:.3f}"
            ],
            hoverinfo="text",
            name="Current Position",
            showlegend=True,
        ))

        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

        # Phase timeline and transitions below the 3D plot
        with st.expander("Phase Timeline", expanded=False):
            for i, seg in enumerate(segments):
                pc = PHASE_COLOURS[seg["phase_idx"] % len(PHASE_COLOURS)]
                st.markdown(
                    f"**{i+1}.** "
                    f"<span style='color:{pc}'>"
                    f"**{seg['phase_name']}**</span> — "
                    f"{format_time(seg['start_time'])} to "
                    f"{format_time(seg['end_time'])} "
                    f"({seg['duration']:.0f}s, {seg['frame_count']} frames)",
                    unsafe_allow_html=True,
                )

        with st.expander("Phase Transitions", expanded=False):
            if transitions:
                for tr in transitions:
                    conf = tr["confidence_at_transition"]
                    stability = (
                        "stable" if conf >= 0.8
                        else "unstable" if conf < 0.5
                        else "moderate"
                    )
                    st.markdown(
                        f"**{format_time(tr['time'])}** — "
                        f"{tr['from_phase']} → **{tr['to_phase']}** "
                        f"(confidence: {conf:.3f}, {stability})"
                    )
            else:
                st.info("No phase transitions detected.")

    # ── RIGHT PANEL: Nemotron Chat ───────────────────────────────────
    # Nemotron is POST-HOC ONLY.  It reasons over the structured
    # procedure summary, NOT over live video or pixels.
    # Reasoning is hidden by default to keep the interface calm.
    with right_col:
        st.subheader("Ask Nemotron")
        st.caption(
            "Post-hoc case review.  Reasoning hidden by default."
        )

        # ── Preset question buttons (functional, auto-trigger) ────
        # When clicked, these IMMEDIATELY trigger Nemotron inference
        # and display the response.  They are functional, not placeholders.
        presets = [
            "Summarize this surgery for a trainee.",
            "Explain what happened during Calot Triangle Dissection.",
            "What does the 3D workflow space reveal?",
        ]

        triggered_question: str | None = None

        for preset in presets:
            if st.button(preset, key=f"preset_{preset[:30]}"):
                triggered_question = preset

        # Custom question input
        question: str = st.text_area(
            "Your question",
            value="",
            height=80,
            placeholder="e.g. Which phase was most complex?",
            key="qa_text_area",
        ) or ""

        if st.button("Ask Nemotron", type="primary") and question.strip():
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

    # ── Auto-advance: virtual playback clock ─────────────────────────
    # The page auto-reruns at _REFRESH_INTERVAL_SEC intervals while
    # playing.  This provides approximate synchronisation between
    # the video playback and the UI state (3D cursor, phase info).
    # Approximate sync is acceptable and documented.
    if st.session_state.get("playing", False):
        time.sleep(_REFRESH_INTERVAL_SEC)
        st.rerun()


if __name__ == "__main__":
    main()
