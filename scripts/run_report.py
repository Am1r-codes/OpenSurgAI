#!/usr/bin/env python3
"""Generate a structured surgical case report using Nemotron.

Reads the scene JSONL for a video, aggregates procedure statistics,
and uses Nemotron to generate a comprehensive markdown case report
covering: procedure overview, phase-by-phase analysis, instrument
usage, workflow observations, and educational notes.

The report is saved as a markdown file and optionally printed to
stdout.

Requires the ``NEMOTRON_API_KEY`` (or ``NVIDIA_API_KEY``) environment
variable.

Usage:
    python scripts/run_report.py --video video49

    # Custom output
    python scripts/run_report.py --video video49 \
        --output experiments/reports/video49_report.md
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.phase_space_3d import (
    build_semantic_phase_space,
    get_phase_segments,
    get_transition_points,
)
from src.explanation.pipeline import NemotronClient, PHASE_EXPLANATIONS
from scripts.run_posthoc_qa import (
    aggregate_summary,
    format_summary_for_prompt,
    load_scene_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Report system prompt ─────────────────────────────────────────────

REPORT_SYSTEM_PROMPT = """\
You are a surgical education assistant for the OpenSurgAI platform.
You are generating a STRUCTURED CASE REPORT for a completed
laparoscopic cholecystectomy procedure.

You have access to a detailed PROCEDURE SUMMARY below with phase
timelines, durations, instrument usage, transition points, confidence
scores, and 3D workflow space metrics.

Generate a professional markdown report with these sections:

## Procedure Overview
Brief summary: video ID, total duration, number of phases observed,
overall complexity assessment.

## Phase-by-Phase Analysis
For each phase that appeared in the procedure (in chronological order):
- Phase name and duration
- What typically happens during this phase (educational context)
- Observations from the data (instruments used, confidence level,
  activity patterns)
- Any notable characteristics (unusually long/short, multiple
  revisits, low confidence)

## Instrument Usage Patterns
- Which instruments dominated the procedure
- How instrument usage varied across phases
- What this tells us about surgical technique

## Workflow Observations
- Phase transition patterns (smooth vs. unstable)
- Any phase revisits or unusual ordering
- Overall procedural flow assessment
- Reference the 3D Semantic Surgical Workflow Space:
  "Dense clusters indicate stable phases, scattered regions suggest
  complexity"

## Educational Notes
- Key teaching points from this case
- What a trainee should notice about this procedure
- Comparison to typical cholecystectomy patterns

Write in a warm, educational tone.  Be thorough and data-driven.
Reference specific numbers from the summary (durations, frame counts,
confidence scores).  All timestamps are in mm:ss format — use this
format in the report as well (never raw seconds).

The 7 Cholec80 surgical instruments are: Grasper, Bipolar, Hook,
Scissors, Clipper, Irrigator, SpecimenBag.  Only reference these
instrument names.

You MUST NOT invent events not supported by the data.
You MUST NOT claim anatomical reconstruction.
The 3D space represents procedural workflow, not anatomy.\
"""


# ── Report generation ────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Format seconds as mm:ss (e.g. 125.3 -> '2:05')."""
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def build_workflow_summary(
    space: dict,
    segments: list[dict],
    transitions: list[dict],
) -> str:
    """Build additional workflow-space metrics for the report prompt."""
    lines: list[str] = []

    lines.append("=== 3D WORKFLOW SPACE METRICS ===")
    lines.append(f"Total points: {len(space['time'])}")

    import numpy as np

    # Per-phase activity statistics
    lines.append("\nPer-phase activity (Z-axis) statistics:")
    phase_idx = space["phase_idx"]
    activity = space["activity"]
    for pidx in sorted(set(phase_idx)):
        mask = phase_idx == pidx
        act = activity[mask]
        if len(act) > 0:
            pname = space["phase_names"][np.where(mask)[0][0]]
            lines.append(
                f"  {pname}: mean={act.mean():.3f}, "
                f"std={act.std():.3f}, "
                f"max={act.max():.3f}, "
                f"frames={mask.sum()}"
            )

    # Transition confidence summary
    if transitions:
        confs = [t["confidence_at_transition"] for t in transitions]
        lines.append(f"\nTransition confidence: "
                     f"mean={sum(confs)/len(confs):.3f}, "
                     f"min={min(confs):.3f}, max={max(confs):.3f}")
        low_conf = [t for t in transitions if t["confidence_at_transition"] < 0.7]
        if low_conf:
            lines.append(f"Low-confidence transitions ({len(low_conf)}):")
            for t in low_conf:
                lines.append(
                    f"  {_fmt_time(t['time'])}: {t['from_phase']} -> {t['to_phase']} "
                    f"(conf={t['confidence_at_transition']:.3f})"
                )

    return "\n".join(lines)


def generate_report(
    video_id: str,
    scene_path: Path,
    api_key: str | None = None,
    model: str = "nvidia/llama-3.3-nemotron-super-49b-v1.5",
) -> str:
    """Generate a full case report for a video. Returns markdown string."""
    log.info("Loading scene data: %s", scene_path)
    scenes = load_scene_jsonl(scene_path)
    log.info("Loaded %d frames", len(scenes))

    # Build all data
    space = build_semantic_phase_space(str(scene_path))
    segments = get_phase_segments(space)
    transitions = get_transition_points(space)
    summary = aggregate_summary(scenes, video_id)
    summary_text = format_summary_for_prompt(summary)
    workflow_text = build_workflow_summary(space, segments, transitions)

    # Combine into prompt
    full_summary = summary_text + "\n\n" + workflow_text

    log.info("Summary: %d phases, %d transitions, %.0fs duration",
             len(summary["phase_durations"]),
             summary["num_phase_transitions"],
             summary["duration_sec"])

    # Call Nemotron
    log.info("Generating report with Nemotron (%s)...", model)
    client = NemotronClient(
        api_key=api_key,
        model=model,
        temperature=1.1,
        max_tokens=None,
        top_p=0.95,
        presence_penalty=0.4,
    )

    user_prompt = (
        f"Generate a complete surgical case report for {video_id}.\n\n"
        f"--- PROCEDURE SUMMARY ---\n{full_summary}"
    )

    try:
        response = client.chat(
            system=REPORT_SYSTEM_PROMPT,
            user=user_prompt,
        )
        choices = response.get("choices", [])
        raw = choices[0]["message"]["content"].strip() if choices else ""
        # Strip Nemotron's internal <think> blocks
        report = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        usage = response.get("usage", {})

        prompt_tok = usage.get("prompt_tokens", 0)
        compl_tok = usage.get("completion_tokens", 0)
        log.info("Report generated: %d prompt + %d completion tokens",
                 prompt_tok, compl_tok)
    finally:
        client.close()

    # Add header
    header = (
        f"# Operative Case Report — {video_id}\n\n"
        f"*Generated by OpenSurgAI Multi-NIM Surgical Intelligence Platform*\n"
        f"*NVIDIA Nemotron Super 49B · NIM Inference Microservice*\n\n"
        f"---\n\n"
    )

    return header + report


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a surgical case report using Nemotron.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--video", type=str, required=True,
        help="Video ID (e.g. 'video49')",
    )
    p.add_argument(
        "--scene-dir", type=Path,
        default=_PROJECT_ROOT / "experiments" / "scene",
        help="Directory with scene JSONL files",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Output markdown file (default: experiments/reports/<video>_report.md)",
    )
    p.add_argument(
        "--api-key", type=str, default=None,
        help="Nemotron API key (default: env var)",
    )
    p.add_argument(
        "--model", type=str,
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        help="Nemotron model identifier",
    )
    p.add_argument(
        "--print", action="store_true", dest="print_report",
        help="Also print report to stdout",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    scene_path = Path(args.scene_dir) / f"{args.video}_scene.jsonl"
    if not scene_path.exists():
        log.error("Scene file not found: %s", scene_path)
        sys.exit(1)

    output_path = args.output or (
        _PROJECT_ROOT / "experiments" / "reports" / f"{args.video}_report.md"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = generate_report(
        video_id=args.video,
        scene_path=scene_path,
        api_key=args.api_key,
        model=args.model,
    )

    output_path.write_text(report, encoding="utf-8")
    log.info("Report saved: %s", output_path)

    if args.print_report:
        print("\n" + report)


if __name__ == "__main__":
    main()
