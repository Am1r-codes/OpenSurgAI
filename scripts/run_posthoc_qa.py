#!/usr/bin/env python3
"""Post-hoc interactive Q&A over a completed surgical procedure.

Loads the full scene JSONL for a video, aggregates it into a structured
surgical summary (phase timeline, durations, stability, instrument usage),
and exposes a CLI loop where the user types natural-language questions
answered by Nemotron.

Requires the ``NEMOTRON_API_KEY`` (or ``NVIDIA_API_KEY``) environment
variable to be set.

Usage examples:

    # Interactive Q&A for a single video
    python scripts/run_posthoc_qa.py --video video49

    # With a custom scene directory
    python scripts/run_posthoc_qa.py --video video49 \
        --scene-dir experiments/scene

    # Use a different model or temperature
    python scripts/run_posthoc_qa.py --video video49 \
        --model nvidia/llama-3.3-nemotron-super-49b-v1.5 \
        --temperature 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.explanation.pipeline import NemotronClient, PHASE_EXPLANATIONS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── System prompt for post-hoc reasoning ─────────────────────────────

QA_SYSTEM_PROMPT = """\
You are a friendly surgical education assistant. You help medical students
and surgical trainees understand laparoscopic cholecystectomy procedures
by explaining what happened during a completed surgery.

You have a STRUCTURED SUMMARY of the procedure below, including phase
timelines, durations, instrument usage, and confidence scores from the
computer vision system.

When answering questions:
- Use a warm, educational tone — like a senior resident teaching an intern.
- Feel free to draw on general surgical knowledge about cholecystectomy
  to give context and explain WHY each phase matters.
- Reference the actual data (durations, instruments, phase order) to
  ground your explanations.
- Give detailed, thorough answers. Use bullet points or numbered lists
  when it helps readability.
- If the user asks for a summary or overview, be comprehensive.
- If the data doesn't cover something, say so, but still share relevant
  general surgical knowledge if it helps the student learn.\
"""


# ── Scene loading and aggregation ────────────────────────────────────

def load_scene_jsonl(path: Path) -> list[dict]:
    """Load all records from a scene JSONL file."""
    scenes: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                scenes.append(json.loads(line))
    return scenes


def aggregate_summary(scenes: list[dict], video_id: str) -> dict:
    """Build a structured surgical summary from scene records.

    Returns a dict with:
    - video_id, total_frames, duration_sec
    - phase_timeline: ordered list of {phase_name, start_frame, end_frame,
      start_sec, end_sec, duration_sec, mean_confidence, frame_count}
    - instrument_summary: {instrument_name: total_frame_appearances}
    - phase_instrument_map: {phase_name: {instrument: count}}
    """
    if not scenes:
        return {"video_id": video_id, "total_frames": 0, "phases": []}

    total_frames = len(scenes)
    first_ts = scenes[0].get("timestamp_sec", 0.0)
    last_ts = scenes[-1].get("timestamp_sec", 0.0)
    duration_sec = last_ts - first_ts

    # Build phase segments (consecutive runs of the same phase)
    segments: list[dict] = []
    current_phase: str | None = None
    seg_start_frame = 0
    seg_start_sec = 0.0
    seg_confidences: list[float] = []
    seg_frame_count = 0

    def _flush_segment() -> None:
        if current_phase is not None and seg_frame_count > 0:
            mean_conf = sum(seg_confidences) / len(seg_confidences) if seg_confidences else 0.0
            segments.append({
                "phase_name": current_phase,
                "start_frame": seg_start_frame,
                "end_frame": scenes[min(seg_start_frame + seg_frame_count - 1, total_frames - 1)].get("frame_idx", 0),
                "start_sec": round(seg_start_sec, 2),
                "end_sec": round(scenes[min(seg_start_frame + seg_frame_count - 1, total_frames - 1)].get("timestamp_sec", 0.0), 2),
                "duration_sec": round(
                    scenes[min(seg_start_frame + seg_frame_count - 1, total_frames - 1)].get("timestamp_sec", 0.0) - seg_start_sec, 2
                ),
                "mean_confidence": round(mean_conf, 3),
                "frame_count": seg_frame_count,
            })

    for i, scene in enumerate(scenes):
        phase = scene.get("phase", {})
        phase_name = phase.get("phase_name", "Unknown") if phase else "Unknown"
        confidence = phase.get("confidence", 0.0) if phase else 0.0

        if phase_name != current_phase:
            _flush_segment()
            current_phase = phase_name
            seg_start_frame = i
            seg_start_sec = scene.get("timestamp_sec", 0.0)
            seg_confidences = [confidence]
            seg_frame_count = 1
        else:
            seg_confidences.append(confidence)
            seg_frame_count += 1

    _flush_segment()

    # Instrument counts (global and per-phase)
    instrument_counter: Counter = Counter()
    phase_instrument_map: dict[str, Counter] = defaultdict(Counter)

    for scene in scenes:
        phase = scene.get("phase", {})
        phase_name = phase.get("phase_name", "Unknown") if phase else "Unknown"
        instruments = scene.get("instruments", [])
        for inst in instruments:
            name = inst.get("class_name", inst.get("name", "Unknown"))
            instrument_counter[name] += 1
            phase_instrument_map[phase_name][name] += 1

    # Phase durations (aggregate across all segments of the same phase)
    phase_total_duration: Counter = Counter()
    phase_total_frames: Counter = Counter()
    for seg in segments:
        phase_total_duration[seg["phase_name"]] += seg["duration_sec"]
        phase_total_frames[seg["phase_name"]] += seg["frame_count"]

    return {
        "video_id": video_id,
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "phase_timeline": segments,
        "phase_durations": {
            k: round(v, 2) for k, v in sorted(phase_total_duration.items(), key=lambda x: -x[1])
        },
        "phase_frame_counts": dict(sorted(phase_total_frames.items(), key=lambda x: -x[1])),
        "instrument_usage": dict(instrument_counter.most_common()),
        "phase_instrument_map": {
            phase: dict(counts.most_common())
            for phase, counts in sorted(phase_instrument_map.items())
        },
        "num_phase_transitions": len(segments),
    }


def format_summary_for_prompt(summary: dict) -> str:
    """Format the structured summary as readable text for the system prompt."""
    lines: list[str] = []
    lines.append(f"Video: {summary['video_id']}")
    lines.append(f"Total frames: {summary['total_frames']}")
    lines.append(f"Duration: {summary['duration_sec']}s")
    lines.append(f"Phase transitions: {summary['num_phase_transitions']}")
    lines.append("")

    lines.append("=== PHASE TIMELINE (chronological) ===")
    for seg in summary["phase_timeline"]:
        lines.append(
            f"  {seg['phase_name']}: frames {seg['start_frame']}-{seg['end_frame']} "
            f"({seg['start_sec']}s - {seg['end_sec']}s, {seg['duration_sec']}s, "
            f"{seg['frame_count']} frames, confidence={seg['mean_confidence']})"
        )
    lines.append("")

    lines.append("=== PHASE DURATIONS (total across all segments) ===")
    for phase, dur in summary["phase_durations"].items():
        frames = summary["phase_frame_counts"].get(phase, 0)
        lines.append(f"  {phase}: {dur}s ({frames} frames)")
    lines.append("")

    lines.append("=== INSTRUMENT USAGE (total frame appearances) ===")
    if summary["instrument_usage"]:
        for inst, count in summary["instrument_usage"].items():
            lines.append(f"  {inst}: {count} frames")
    else:
        lines.append("  No instruments detected.")
    lines.append("")

    lines.append("=== INSTRUMENTS PER PHASE ===")
    for phase, instruments in summary["phase_instrument_map"].items():
        inst_str = ", ".join(f"{k}({v})" for k, v in instruments.items())
        lines.append(f"  {phase}: {inst_str}")
    lines.append("")

    lines.append("=== CANONICAL PHASE DESCRIPTIONS ===")
    for phase_name, desc in PHASE_EXPLANATIONS.items():
        lines.append(f"  {phase_name}: {desc}")

    return "\n".join(lines)


# ── Interactive Q&A loop ─────────────────────────────────────────────

def run_qa_loop(client: NemotronClient, summary_text: str) -> None:
    """Run the interactive Q&A loop."""
    system = QA_SYSTEM_PROMPT + "\n\n--- PROCEDURE SUMMARY ---\n" + summary_text

    print("\n" + "=" * 60)
    print("  Post-hoc Surgical Q&A")
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        try:
            response = client.chat(system=system, user=question)
            choices = response.get("choices", [])
            answer = choices[0]["message"]["content"].strip() if choices else "(no response)"
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            print(f"\nA: {answer}")
            print(f"   [{prompt_tokens}+{completion_tokens} tokens]\n")

        except Exception as exc:
            print(f"\n[Error] {exc}\n")


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-hoc interactive Q&A over a completed surgical procedure.",
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
        help="Directory with scene JSONL files (default: experiments/scene/)",
    )
    p.add_argument(
        "--api-key", type=str, default=None,
        help="Nemotron API key (default: NEMOTRON_API_KEY env var)",
    )
    p.add_argument(
        "--base-url", type=str,
        default="https://integrate.api.nvidia.com/v1",
        help="API base URL",
    )
    p.add_argument(
        "--model", type=str,
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        help="Nemotron model identifier",
    )
    p.add_argument(
        "--temperature", type=float, default=0.5,
        help="Sampling temperature (default: 0.5)",
    )
    p.add_argument(
        "--max-tokens", type=int,
        help="Max completion tokens per answer",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    scene_path = Path(args.scene_dir) / f"{args.video}_scene.jsonl"
    if not scene_path.exists():
        log.error("Scene file not found: %s", scene_path)
        sys.exit(1)

    log.info("Loading scene data: %s", scene_path)
    scenes = load_scene_jsonl(scene_path)
    log.info("Loaded %d frames", len(scenes))

    log.info("Aggregating surgical summary...")
    summary = aggregate_summary(scenes, args.video)
    summary_text = format_summary_for_prompt(summary)

    log.info("Summary: %d phases, %d transitions, %.1fs duration",
             len(summary["phase_durations"]), summary["num_phase_transitions"],
             summary["duration_sec"])
    log.info("Instruments: %s", ", ".join(summary["instrument_usage"].keys()) or "none detected")

    log.info("Connecting to Nemotron (%s)...", args.model)
    with NemotronClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ) as client:
        run_qa_loop(client, summary_text)


if __name__ == "__main__":
    main()
