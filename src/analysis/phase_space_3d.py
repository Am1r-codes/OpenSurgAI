"""3D Semantic Surgical Workflow Space.

WHY 3D IS USED
--------------
Surgical procedures have three independent dimensions of variation that
cannot be captured in a 2D timeline:

1. **Where within a phase** the surgeon is (progress)
2. **Which phase** is active (identity)
3. **How much activity / instability** is happening (complexity)

Collapsing these onto a timeline loses the spatial structure: stable
phases appear as dense clouds, chaotic transitions as scattered points,
and phase revisits create visually distinct return clusters.

WHAT THIS REPRESENTS
--------------------
Each frame of a processed surgical video maps to ONE POINT in a 3D
semantic workflow space:

- **X — Phase Progression** : normalised progress within the current
  phase segment [0 → 1].  Resets on every phase change.
- **Y — Phase Identity** : ordinal surgical phase index (0–6 for
  Cholec80 laparoscopic cholecystectomy).
- **Z — Surgical Activity / Complexity** : composite metric combining
  instrument count (normalised) and confidence volatility (rolling
  standard deviation).  Higher = more manipulation / instability.

WHAT THIS DOES NOT REPRESENT
-----------------------------
This visualisation represents **procedural structure and activity**,
**not** anatomical geometry or spatial reconstruction.  The axes are
semantic, not physical.  No claim of anatomical accuracy is made.

Data flow::

    scene JSONL  ──>  build_semantic_phase_space()  ──>  space dict
                                                          │
                      get_phase_segments()  <──────────────┤
                      get_transition_points()  <───────────┘
                                                          │
                      build_workflow_figure()  ────────────┘──> Plotly Figure
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ── Cholec80 phase ordering (canonical) ──────────────────────────────

PHASE_ORDER: list[str] = [
    "Preparation",               # 0
    "CalotTriangleDissection",   # 1
    "ClippingCutting",           # 2
    "GallbladderDissection",     # 3
    "GallbladderPackaging",      # 4
    "CleaningCoagulation",       # 5
    "GallbladderRetraction",     # 6
]

PHASE_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(PHASE_ORDER)}

# Colours per phase (hex)
PHASE_COLOURS: list[str] = [
    "#e6194b",  # Preparation — red
    "#3cb44b",  # CalotTriangleDissection — green
    "#4363d8",  # ClippingCutting — blue
    "#f58231",  # GallbladderDissection — orange
    "#911eb4",  # GallbladderPackaging — purple
    "#42d4f4",  # CleaningCoagulation — cyan
    "#f032e6",  # GallbladderRetraction — magenta
]


# ── Scene JSONL loading ──────────────────────────────────────────────

def _load_scenes(scene_path: Path) -> list[dict]:
    """Load all records from a scene JSONL file."""
    scenes: list[dict] = []
    with open(scene_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                scenes.append(json.loads(line))
    return scenes


# ── Metric computation ───────────────────────────────────────────────

def compute_phase_progress(phase_idx_arr: np.ndarray) -> np.ndarray:
    """Compute normalised progress [0→1] within each contiguous phase segment.

    Resets to 0 at each phase boundary.  A single-frame segment gets 0.5.
    """
    n = len(phase_idx_arr)
    progress = np.zeros(n, dtype=np.float64)

    seg_start = 0
    for i in range(1, n):
        if phase_idx_arr[i] != phase_idx_arr[seg_start]:
            seg_len = i - seg_start
            if seg_len == 1:
                progress[seg_start] = 0.5
            else:
                progress[seg_start:i] = np.linspace(0.0, 1.0, seg_len)
            seg_start = i

    # Final segment
    seg_len = n - seg_start
    if seg_len == 1:
        progress[seg_start] = 0.5
    else:
        progress[seg_start:n] = np.linspace(0.0, 1.0, seg_len)

    return progress


def compute_activity_metric(
    instrument_count: np.ndarray,
    confidence: np.ndarray,
    volatility_window: int = 25,
) -> np.ndarray:
    """Compute Z-axis surgical activity / complexity metric.

    Combines:
    - Normalised instrument count (0–1)
    - Confidence volatility: rolling standard deviation of phase
      confidence over a window.  High volatility = instability.

    Both are weighted equally and summed, then clipped to [0, 1].
    """
    n = len(instrument_count)

    # Normalise instrument count to 0–1
    ic_max = instrument_count.max() if instrument_count.max() > 0 else 1.0
    ic_norm = instrument_count / ic_max

    # Rolling std of confidence (measures instability)
    vol = np.zeros(n, dtype=np.float64)
    half_w = volatility_window // 2
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        vol[i] = np.std(confidence[lo:hi])

    # Normalise volatility to 0–1
    vol_max = vol.max() if vol.max() > 0 else 1.0
    vol_norm = vol / vol_max

    # Composite: equal weight
    activity = np.clip((ic_norm + vol_norm) / 2.0, 0.0, 1.0)
    return activity


# ── Core: build the semantic 3D workflow space ───────────────────────

def build_semantic_phase_space(
    scene_path: str | Path,
    volatility_window: int = 25,
) -> dict:
    """Build the 3D Semantic Surgical Workflow Space from a scene JSONL.

    This 3D space represents procedural structure and activity,
    not anatomical geometry or spatial reconstruction.

    Axes:
        X — Phase Progression [0→1] within each phase segment
        Y — Phase Identity (ordinal 0–6)
        Z — Surgical Activity / Complexity (composite metric)

    Parameters
    ----------
    scene_path : str | Path
        Path to the scene JSONL file.
    volatility_window : int
        Rolling window size for confidence volatility (default 25 frames).

    Returns
    -------
    dict with keys:
        video_id, time, phase_idx, phase_names, frame_indices,
        confidence, instrument_count,
        phase_progress (X), activity (Z)
    """
    scene_path = Path(scene_path)
    scenes = _load_scenes(scene_path)

    if not scenes:
        raise ValueError(f"No frames found in {scene_path}")

    n = len(scenes)
    time_arr = np.zeros(n, dtype=np.float64)
    phase_idx_arr = np.zeros(n, dtype=np.int32)
    confidence_arr = np.zeros(n, dtype=np.float64)
    instrument_arr = np.zeros(n, dtype=np.float64)
    frame_idx_arr = np.zeros(n, dtype=np.int64)
    phase_names: list[str] = []

    for i, scene in enumerate(scenes):
        time_arr[i] = scene.get("timestamp_sec", 0.0)
        frame_idx_arr[i] = scene.get("frame_idx", 0)

        phase = scene.get("phase") or {}
        pname = phase.get("phase_name", "Unknown")
        phase_names.append(pname)
        phase_idx_arr[i] = PHASE_TO_INDEX.get(pname, -1)
        confidence_arr[i] = phase.get("confidence", 0.0)
        instrument_arr[i] = scene.get("instrument_count", 0)

    # Compute semantic axes
    phase_progress = compute_phase_progress(phase_idx_arr)
    activity = compute_activity_metric(
        instrument_arr, confidence_arr, volatility_window,
    )

    video_id = scenes[0].get("video_id", scene_path.stem)

    log.info(
        "Built semantic workflow space for %s: %d points, %.1fs duration, "
        "%d unique phases",
        video_id, n, time_arr[-1] - time_arr[0],
        len(set(phase_names)),
    )

    return {
        "video_id": video_id,
        "time": time_arr,
        "phase_idx": phase_idx_arr,
        "phase_names": phase_names,
        "frame_indices": frame_idx_arr,
        "confidence": confidence_arr,
        "instrument_count": instrument_arr,
        # Semantic 3D axes
        "phase_progress": phase_progress,   # X
        "activity": activity,               # Z
    }


# ── Phase segments ───────────────────────────────────────────────────

def get_phase_segments(space: dict) -> list[dict]:
    """Extract contiguous phase segments from the workflow space.

    Returns a list of dicts with keys:
        phase_name, phase_idx, start_idx, end_idx,
        start_time, end_time, duration, frame_count
    """
    phase_idx = space["phase_idx"]
    time_arr = space["time"]
    phase_names = space["phase_names"]
    n = len(phase_idx)

    segments: list[dict] = []
    seg_start = 0

    for i in range(1, n):
        if phase_idx[i] != phase_idx[seg_start]:
            segments.append({
                "phase_name": phase_names[seg_start],
                "phase_idx": int(phase_idx[seg_start]),
                "start_idx": seg_start,
                "end_idx": i - 1,
                "start_time": float(time_arr[seg_start]),
                "end_time": float(time_arr[i - 1]),
                "duration": float(time_arr[i - 1] - time_arr[seg_start]),
                "frame_count": i - seg_start,
            })
            seg_start = i

    # Final segment
    segments.append({
        "phase_name": phase_names[seg_start],
        "phase_idx": int(phase_idx[seg_start]),
        "start_idx": seg_start,
        "end_idx": n - 1,
        "start_time": float(time_arr[seg_start]),
        "end_time": float(time_arr[n - 1]),
        "duration": float(time_arr[n - 1] - time_arr[seg_start]),
        "frame_count": n - seg_start,
    })

    return segments


# ── Transition points ────────────────────────────────────────────────

def get_transition_points(space: dict) -> list[dict]:
    """Identify phase transition points (where Y-axis jumps).

    Returns a list of dicts with keys:
        idx, time, from_phase, to_phase, from_idx, to_idx,
        confidence_at_transition
    """
    phase_idx = space["phase_idx"]
    time_arr = space["time"]
    phase_names = space["phase_names"]
    confidence = space["confidence"]
    n = len(phase_idx)

    transitions: list[dict] = []
    for i in range(1, n):
        if phase_idx[i] != phase_idx[i - 1]:
            transitions.append({
                "idx": i,
                "time": float(time_arr[i]),
                "from_phase": phase_names[i - 1],
                "to_phase": phase_names[i],
                "from_idx": int(phase_idx[i - 1]),
                "to_idx": int(phase_idx[i]),
                "confidence_at_transition": float(confidence[i]),
            })

    return transitions


# ── Plotly visualisation ─────────────────────────────────────────────

def build_workflow_figure(
    space: dict,
    title: str | None = None,
    point_size: int = 3,
    downsample: int = 1,
    active_phase_idx: int | None = None,
):
    """Build a Plotly Figure for the 3D Semantic Surgical Workflow Space.

    This visualisation represents procedural structure and activity,
    not anatomical geometry or spatial reconstruction.

    Parameters
    ----------
    space : dict
        Output from :func:`build_semantic_phase_space`.
    title : str | None
        Plot title.
    point_size : int
        Base point size for trajectory cloud.
    downsample : int
        Plot every Nth point (1 = all).
    active_phase_idx : int | None
        If set, points belonging to this phase are rendered larger and
        brighter to highlight the active phase region.
    """
    import plotly.graph_objects as go

    video_id = space["video_id"]
    if title is None:
        title = f"{video_id} — 3D Semantic Surgical Workflow Space"

    step = max(1, downsample)
    x = space["phase_progress"][::step]
    y = space["phase_idx"][::step]
    z = space["activity"][::step]
    names = space["phase_names"][::step]
    conf = space["confidence"][::step]
    inst = space["instrument_count"][::step]
    t = space["time"][::step]

    fig = go.Figure()

    # ── Trajectory lines: connect sequential frames within each phase ──
    # Rendered first (behind point clouds) as thin lines showing the
    # surgical path through workflow space.  Each contiguous phase
    # segment gets its own line trace.
    full_x = space["phase_progress"][::step]
    full_y = space["phase_idx"][::step]
    full_z = space["activity"][::step]
    full_n = len(full_x)

    if full_n > 1:
        seg_start = 0
        for i in range(1, full_n + 1):
            if i == full_n or full_y[i] != full_y[seg_start]:
                seg_len = i - seg_start
                if seg_len >= 2:
                    pidx_val = int(full_y[seg_start])
                    colour = PHASE_COLOURS[pidx_val % len(PHASE_COLOURS)]
                    line_opacity = 0.6 if (
                        active_phase_idx is not None
                        and pidx_val == active_phase_idx
                    ) else 0.2
                    fig.add_trace(go.Scatter3d(
                        x=full_x[seg_start:i],
                        y=full_y[seg_start:i],
                        z=full_z[seg_start:i],
                        mode="lines",
                        line=dict(color=colour, width=2),
                        opacity=line_opacity,
                        hoverinfo="skip",
                        showlegend=False,
                    ))
                seg_start = i

    # ── Point clouds: one trace per phase ──────────────────────────────
    unique_phases_seen = []
    for pidx, pname in enumerate(PHASE_ORDER):
        mask = y == pidx
        if not np.any(mask):
            continue
        unique_phases_seen.append(pidx)

        px = x[mask]
        py = y[mask]
        pz = z[mask]
        pt = t[mask]
        pc = conf[mask]
        pi = inst[mask]

        # Highlight active phase region
        if active_phase_idx is not None and pidx == active_phase_idx:
            opacity = 0.9
            size = point_size + 2
        else:
            opacity = 0.5
            size = point_size

        hover = [
            f"Phase: {pname}<br>"
            f"Progress: {xv:.0%}<br>"
            f"Activity: {zv:.3f}<br>"
            f"Time: {tv:.1f}s<br>"
            f"Confidence: {cv:.3f}<br>"
            f"Instruments: {int(iv)}"
            for xv, zv, tv, cv, iv in zip(px, pz, pt, pc, pi)
        ]

        colour = PHASE_COLOURS[pidx % len(PHASE_COLOURS)]

        fig.add_trace(go.Scatter3d(
            x=px, y=py, z=pz,
            mode="markers",
            marker=dict(size=size, color=colour, opacity=opacity),
            text=hover,
            hoverinfo="text",
            name=f"{pidx}: {pname}",
            legendgroup=pname,
        ))

    # Y-axis tick labels = phase names
    y_ticks = list(range(len(PHASE_ORDER)))
    y_labels = [f"{i}: {name}" for i, name in enumerate(PHASE_ORDER)]

    fig.update_layout(
        title=dict(
            text=(
                f"{title}<br>"
                "<sub>Procedural structure and activity — "
                "not anatomical geometry or spatial reconstruction</sub>"
            ),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="Phase Progression", range=[-0.05, 1.05]),
            yaxis=dict(
                title="Phase Identity",
                tickvals=y_ticks,
                ticktext=y_labels,
            ),
            zaxis=dict(title="Surgical Activity / Complexity", range=[-0.05, 1.05]),
            camera=dict(eye=dict(x=1.8, y=-1.2, z=0.9)),
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(size=10),
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        template="plotly_dark",
    )

    return fig


# ── Multi-surgery comparison figure ──────────────────────────────────

# Distinct colours per surgery (not per phase) for comparison mode.
SURGERY_COLOURS: list[str] = [
    "#00ccff",  # cyan
    "#ff6600",  # orange
    "#33ff33",  # lime
    "#ff33cc",  # pink
    "#ffff00",  # yellow
]


def build_comparison_figure(
    spaces: list[dict],
    point_size: int = 3,
    downsample: int = 5,
) -> "go.Figure":
    """Build a Plotly 3D figure overlaying multiple surgical trajectories.

    Each surgery gets a unique colour so trajectory differences are
    immediately visible.  Phase identity (Y-axis) is shared across all
    surgeries, making it easy to compare timing and activity patterns.

    Parameters
    ----------
    spaces : list[dict]
        List of space dicts from :func:`build_semantic_phase_space`.
    point_size : int
        Base marker size.
    downsample : int
        Plot every Nth point per surgery.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    step = max(1, downsample)

    for s_idx, space in enumerate(spaces):
        vid = space["video_id"]
        colour = SURGERY_COLOURS[s_idx % len(SURGERY_COLOURS)]

        x = space["phase_progress"][::step]
        y = space["phase_idx"][::step]
        z = space["activity"][::step]
        t = space["time"][::step]
        names = space["phase_names"][::step]
        conf = space["confidence"][::step]

        hover = [
            f"<b>{vid}</b><br>"
            f"Phase: {n}<br>"
            f"Progress: {xv:.0%}<br>"
            f"Activity: {zv:.3f}<br>"
            f"Time: {tv:.1f}s<br>"
            f"Confidence: {cv:.3f}"
            for n, xv, zv, tv, cv in zip(names, x, z, t, conf)
        ]

        # Trajectory line
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=colour, width=2),
            opacity=0.3,
            hoverinfo="skip",
            showlegend=False,
        ))

        # Point cloud
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=point_size, color=colour, opacity=0.7),
            text=hover,
            hoverinfo="text",
            name=vid,
            legendgroup=vid,
        ))

    # Shared axis labels
    y_ticks = list(range(len(PHASE_ORDER)))
    y_labels = [f"{i}: {name}" for i, name in enumerate(PHASE_ORDER)]

    fig.update_layout(
        title=dict(
            text=(
                "Multi-Surgery Comparison<br>"
                "<sub>Trajectories overlaid in shared workflow space</sub>"
            ),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="Phase Progression", range=[-0.05, 1.05]),
            yaxis=dict(
                title="Phase Identity",
                tickvals=y_ticks,
                ticktext=y_labels,
            ),
            zaxis=dict(title="Surgical Activity / Complexity", range=[-0.05, 1.05]),
            camera=dict(eye=dict(x=1.8, y=-1.2, z=0.9)),
        ),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        template="plotly_dark",
    )

    return fig


def build_comparison_summary(spaces: list[dict]) -> str:
    """Build a structured text summary comparing multiple surgeries.

    This is used as context for Nemotron comparison Q&A.
    """
    lines = ["MULTI-SURGERY COMPARISON DATA", "=" * 40, ""]

    for space in spaces:
        vid = space["video_id"]
        segments = get_phase_segments(space)
        transitions = get_transition_points(space)
        duration = float(space["time"][-1] - space["time"][0])
        n_frames = len(space["time"])
        unique_phases = sorted(set(space["phase_names"]))
        avg_conf = float(space["confidence"].mean())
        avg_activity = float(space["activity"].mean())

        lines.append(f"--- {vid} ---")
        lines.append(f"  Duration: {duration:.0f}s ({duration/60:.1f} min)")
        lines.append(f"  Frames: {n_frames:,}")
        lines.append(f"  Phases seen: {len(unique_phases)} ({', '.join(unique_phases)})")
        lines.append(f"  Phase segments: {len(segments)}")
        lines.append(f"  Transitions: {len(transitions)}")
        lines.append(f"  Avg confidence: {avg_conf:.3f}")
        lines.append(f"  Avg activity: {avg_activity:.3f}")

        lines.append("  Phase timeline:")
        for seg in segments:
            lines.append(
                f"    {seg['phase_name']}: "
                f"{seg['start_time']:.0f}s-{seg['end_time']:.0f}s "
                f"({seg['duration']:.0f}s, {seg['frame_count']} frames)"
            )
        lines.append("")

    return "\n".join(lines)


# ── File-saving wrapper (for CLI) ───────────────────────────────────

def plot_phase_space_3d(
    space: dict,
    output_path: str | Path | None = None,
    title: str | None = None,
    point_size: int = 3,
    transition_size: int = 12,
    downsample: int = 1,
) -> str | None:
    """Save the workflow space visualisation to an HTML file.

    Backward-compatible wrapper for CLI scripts.
    """
    video_id = space["video_id"]
    if output_path is None:
        output_path = Path(f"experiments/analysis/{video_id}_phase_space_3d.html")
    output_path = Path(output_path).with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = build_workflow_figure(space, title=title, point_size=point_size,
                                downsample=downsample)
    fig.write_html(str(output_path), include_plotlyjs=True)
    log.info("Saved 3D workflow space: %s", output_path)
    return str(output_path)


# ── Backward compatibility aliases ───────────────────────────────────
# These keep scripts/run_phase_space.py working without changes.

def build_phase_space(
    scene_path: str | Path,
    activity_mode: str = "combined",
) -> dict:
    """Backward-compatible alias for build_semantic_phase_space."""
    return build_semantic_phase_space(scene_path)


def build_plotly_figure(
    space: dict,
    title: str | None = None,
    point_size: int = 3,
    transition_size: int = 12,
    downsample: int = 1,
):
    """Backward-compatible alias for build_workflow_figure."""
    return build_workflow_figure(space, title=title, point_size=point_size,
                                 downsample=downsample)
