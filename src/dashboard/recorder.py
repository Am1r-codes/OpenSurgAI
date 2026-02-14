"""Demo video recorder with full pipeline overlay.

Reads the original surgical video alongside all upstream outputs (scene
JSONL, explanation JSONL, segmentation masks), composites every overlay
using :class:`OverlayRenderer`, and writes a smooth annotated MP4
suitable for public sharing.

Output codec
------------
Uses H.264 (``avc1``) via OpenCV's VideoWriter for maximum
compatibility.  The output preserves the original video's resolution and
frame rate.

Data flow
---------
::

    Original video  ──┐
    Scene JSONL     ──┤
    Explanation JSONL──┤──> DemoRecorder ──> annotated_video01.mp4
    Mask PNGs       ──┘
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.dashboard.renderer import OverlayRenderer
from src.explanation.pipeline import PHASE_EXPLANATIONS
from src.video import VideoReader

log = logging.getLogger(__name__)


# ── JSONL helpers ─────────────────────────────────────────────────────

def _load_jsonl_index(path: Path) -> dict[int, dict]:
    """Load a JSONL file and return a dict keyed by ``frame_idx``."""
    index: dict[int, dict] = {}
    if not path.exists():
        return index
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fidx = rec.get("frame_idx")
                if fidx is not None:
                    index[int(fidx)] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return index


def _build_phase_timeline(scene_index: dict[int, dict]) -> list[dict]:
    """Build phase segments from scene data for the timeline bar.

    Returns a list of dicts:
        [{"start": int, "end": int, "phase_id": int, "phase_name": str}, ...]
    """
    if not scene_index:
        return []

    segments: list[dict] = []
    sorted_frames = sorted(scene_index.keys())
    current_pid: int | None = None
    seg_start: int = 0

    for fidx in sorted_frames:
        scene = scene_index[fidx]
        phase = scene.get("phase")
        if not phase:
            continue
        pid = phase.get("phase_id")
        pname = phase.get("phase_name", "")

        if pid != current_pid:
            if current_pid is not None:
                segments.append({
                    "start": seg_start,
                    "end": fidx,
                    "phase_id": current_pid,
                    "phase_name": _last_pname,
                })
            current_pid = pid
            _last_pname = pname
            seg_start = fidx

    # Close last segment
    if current_pid is not None and sorted_frames:
        segments.append({
            "start": seg_start,
            "end": sorted_frames[-1] + 1,
            "phase_id": current_pid,
            "phase_name": _last_pname,
        })

    return segments


# ── Demo recorder ─────────────────────────────────────────────────────

class DemoRecorder:
    """Record an annotated demo video from upstream pipeline outputs.

    Parameters
    ----------
    renderer : OverlayRenderer | None
        Custom renderer instance.  If ``None`` a default renderer is
        created.
    codec : str
        FourCC codec string for VideoWriter.
    output_fps : float | None
        Output frame rate.  ``None`` uses the source video's FPS.
    mask_dir : Path | None
        Directory containing segmentation mask PNGs.  Masks are loaded
        on demand per frame using the ``mask_file`` field from the scene
        data.  If ``None``, mask overlays are skipped.
    """

    def __init__(
        self,
        renderer: OverlayRenderer | None = None,
        codec: str = "avc1",
        output_fps: float | None = None,
        mask_dir: Path | None = None,
        title_card_frames: int = 35,
    ) -> None:
        self.renderer = renderer or OverlayRenderer()
        self.codec = codec
        self.output_fps = output_fps
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.title_card_frames = title_card_frames

    # ── mask loading ──────────────────────────────────────────────────

    def _load_mask(self, mask_file: str | None) -> np.ndarray | None:
        """Load a segmentation mask PNG from disk."""
        if mask_file is None or self.mask_dir is None:
            return None
        mask_path = self.mask_dir / mask_file
        if not mask_path.exists():
            return None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return mask

    # ── record ────────────────────────────────────────────────────────

    def record(
        self,
        video_path: Path,
        output_path: Path,
        scene_path: Path | None = None,
        explanation_path: Path | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        stride: int = 1,
    ) -> dict:
        """Record an annotated demo video.

        Parameters
        ----------
        video_path : Path
            Original surgical video.
        output_path : Path
            Output MP4 path.
        scene_path : Path | None
            Scene JSONL (from ``run_scene_assembly.py``).
        explanation_path : Path | None
            Explanation JSONL (from ``run_explanation.py``).
        start_frame : int
            First frame to include (default 0).
        end_frame : int | None
            Last frame to include (exclusive).  ``None`` = all frames.
        stride : int
            Frame stride (1 = every frame).

        Returns a summary dict.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load upstream data indexed by frame
        scene_index = _load_jsonl_index(Path(scene_path)) if scene_path else {}
        expl_index = _load_jsonl_index(Path(explanation_path)) if explanation_path else {}

        log.info(
            "Data loaded: %d scene frames, %d explanation frames",
            len(scene_index), len(expl_index),
        )

        # Pre-compute phase timeline for the HUD bottom bar
        phase_segments = _build_phase_timeline(scene_index)
        total_scene_frames = max(scene_index.keys()) + 1 if scene_index else 0
        if phase_segments:
            self.renderer.set_phase_timeline(phase_segments, total_scene_frames)
            log.info(
                "Phase timeline: %d segments over %d frames",
                len(phase_segments), total_scene_frames,
            )

        # Open source video
        with VideoReader(video_path, stride=stride) as reader:
            fps = self.output_fps or reader.fps
            w, h = reader.width, reader.height

            log.info(
                "Recording %s -> %s  (%dx%d @ %.1f fps, stride=%d)",
                video_path.name, output_path.name, w, h, fps, stride,
            )

            # Open writer
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open video writer: {output_path}")

            t0 = time.perf_counter()
            frames_written = 0
            frames_with_scene = 0
            frames_with_explanation = 0
            prev_phase_id: int | None = None

            try:
                for frame_idx, timestamp_sec, bgr_frame in reader.frames():
                    # Range filtering
                    if frame_idx < start_frame:
                        continue
                    if end_frame is not None and frame_idx >= end_frame:
                        break

                    # Look up scene and explanation for this frame
                    scene = scene_index.get(frame_idx, {})
                    expl_rec = expl_index.get(frame_idx)

                    explanation = None
                    grounded = True
                    if expl_rec:
                        explanation = expl_rec.get("explanation")
                        grounded = expl_rec.get("grounded", True)
                        frames_with_explanation += 1

                    if scene:
                        frames_with_scene += 1

                    # Inject frame metadata if scene is empty
                    if not scene:
                        scene = {
                            "video_id": video_path.stem,
                            "frame_idx": frame_idx,
                            "timestamp_sec": timestamp_sec,
                            "instruments": [],
                            "instrument_count": 0,
                            "phase": None,
                            "anatomy": None,
                        }

                    # Detect phase change -> insert title card
                    phase = scene.get("phase")
                    if phase:
                        phase_id = phase.get("phase_id")
                        phase_name = phase.get("phase_name", "")
                        if phase_id is not None and phase_id != prev_phase_id:
                            card_text = PHASE_EXPLANATIONS.get(phase_name, "")
                            if card_text:
                                card = self.renderer.render_title_card(
                                    bgr_frame, phase_name, phase_id,
                                    card_text,
                                )
                                for _ in range(self.title_card_frames):
                                    writer.write(card)
                                    frames_written += 1
                                log.info(
                                    "  Title card: %s (%d frames at #%d)",
                                    phase_name, self.title_card_frames, frame_idx,
                                )
                            prev_phase_id = phase_id

                    # Load segmentation mask
                    mask = None
                    anatomy = scene.get("anatomy")
                    if anatomy:
                        mask = self._load_mask(anatomy.get("mask_file"))

                    # Render HUD overlays (with explanation text)
                    annotated = self.renderer.render_frame(
                        frame=bgr_frame,
                        scene=scene,
                        explanation=explanation,
                        grounded=grounded,
                        mask=mask,
                    )

                    writer.write(annotated)
                    frames_written += 1

                    if frames_written % 500 == 0:
                        elapsed = time.perf_counter() - t0
                        fps_actual = frames_written / elapsed if elapsed > 0 else 0
                        log.info(
                            "  %d frames written (%.1f FPS)",
                            frames_written, fps_actual,
                        )

            finally:
                writer.release()

        elapsed = time.perf_counter() - t0
        fps_actual = frames_written / elapsed if elapsed > 0 else 0

        summary = {
            "video_id": video_path.stem,
            "input_video": str(video_path),
            "output_video": str(output_path),
            "frames_written": frames_written,
            "frames_with_scene": frames_with_scene,
            "frames_with_explanation": frames_with_explanation,
            "output_fps": fps,
            "resolution": f"{w}x{h}",
            "elapsed_sec": round(elapsed, 2),
            "render_fps": round(fps_actual, 1),
        }
        log.info(
            "Recorded %s – %d frames in %.1fs (%.1f FPS)",
            output_path.name, frames_written, elapsed, fps_actual,
        )
        return summary
