"""Frame overlay renderer for surgical video dashboard.

Draws detection bounding boxes, segmentation mask overlays, phase labels
with confidence bars, and Nemotron explanation text onto raw video frames.
All rendering is done in-place with OpenCV for zero-dependency operation.

Design principles
-----------------
- **Legibility**: white text on semi-transparent dark backgrounds; thick
  box outlines; consistent font sizing.
- **Non-destructive mask overlay**: the segmentation mask is alpha-
  blended over the frame so the underlying anatomy remains visible.
- **Instrument colour coding**: each instrument class gets a distinct
  colour from a fixed palette for visual consistency across frames.
- **Explanation panel**: Nemotron explanations are rendered in a dark
  panel at the bottom of the frame, word-wrapped to fit.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# ── Instrument colour palette (BGR for OpenCV) ───────────────────────
# One colour per Cholec80 instrument class, high-contrast on dark
# surgical backgrounds.
INSTRUMENT_COLOURS_BGR = [
    (0,   255, 0),    # Grasper     - green
    (255, 255, 0),    # Bipolar     - cyan
    (0,   165, 255),  # Hook        - orange
    (0,   0,   255),  # Scissors    - red
    (255, 0,   255),  # Clipper     - magenta
    (255, 255, 0),    # Irrigator   - cyan
    (0,   255, 255),  # SpecimenBag - yellow
]

# ── Phase colour palette (BGR) ───────────────────────────────────────
PHASE_COLOURS_BGR = [
    (180, 180, 180),  # 0 Preparation            - grey
    (0,   200, 255),  # 1 CalotTriangleDissection - orange
    (0,   0,   255),  # 2 ClippingCutting         - red
    (0,   255, 0),    # 3 GallbladderDissection   - green
    (255, 200, 0),    # 4 GallbladderPackaging     - blue-ish
    (255, 0,   255),  # 5 CleaningCoagulation     - magenta
    (0,   255, 255),  # 6 GallbladderRetraction   - yellow
]

# ── Segmentation mask palette (BGR for OpenCV blending) ──────────────
# 21-entry COCO-VOC palette converted to BGR.
SEG_PALETTE_BGR = np.array([
    [0,   0,   0],    # 0  background (transparent)
    [0,   0,   128],  # 1
    [0,   128, 0],    # 2
    [0,   128, 128],  # 3
    [128, 0,   0],    # 4
    [128, 0,   128],  # 5
    [128, 128, 0],    # 6
    [128, 128, 128],  # 7
    [0,   0,   64],   # 8
    [0,   0,   192],  # 9
    [0,   128, 64],   # 10
    [0,   128, 192],  # 11
    [128, 0,   64],   # 12
    [128, 0,   192],  # 13
    [128, 128, 64],   # 14
    [128, 128, 192],  # 15
    [0,   64,  0],    # 16
    [0,   64,  128],  # 17
    [0,   192, 0],    # 18
    [0,   192, 128],  # 19
    [128, 64,  0],    # 20
], dtype=np.uint8)


def _get_instrument_colour(class_id: int) -> tuple[int, int, int]:
    """Return a BGR colour for the given instrument class ID."""
    idx = class_id % len(INSTRUMENT_COLOURS_BGR)
    return INSTRUMENT_COLOURS_BGR[idx]


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Word-wrap text to fit within *max_chars* per line."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines


class OverlayRenderer:
    """Renders detection, segmentation, phase, and explanation overlays.

    Parameters
    ----------
    mask_alpha : float
        Opacity of the segmentation mask overlay (0.0-1.0).
    bbox_thickness : int
        Bounding box line thickness in pixels.
    font_scale : float
        Base font scale for labels.
    explanation_lines : int
        Maximum number of text lines in the explanation panel.
    show_confidence : bool
        Show confidence values alongside labels.
    """

    def __init__(
        self,
        mask_alpha: float = 0.35,
        bbox_thickness: int = 2,
        font_scale: float = 0.6,
        explanation_lines: int = 4,
        show_confidence: bool = True,
        show_masks: bool = False,
        show_class_labels: bool = False,
        show_detections: bool = False,
    ) -> None:
        self.mask_alpha = mask_alpha
        self.bbox_thickness = bbox_thickness
        self.font_scale = font_scale
        self.explanation_lines = explanation_lines
        self.show_confidence = show_confidence
        self.show_masks = show_masks
        self.show_class_labels = show_class_labels
        self.show_detections = show_detections
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._last_explanation: str | None = None

    # ── segmentation mask overlay ─────────────────────────────────────

    def draw_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Alpha-blend a class-index mask onto the frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame (H, W, 3).
        mask : np.ndarray
            Class-index mask (H, W), uint8.

        Returns the blended frame (modified in-place).
        """
        h, w = frame.shape[:2]
        mh, mw = mask.shape[:2]
        if (mh, mw) != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Build colour overlay from palette
        colour_mask = SEG_PALETTE_BGR[mask]  # (H, W, 3)

        # Only blend non-background pixels
        fg = mask > 0
        if fg.any():
            frame[fg] = cv2.addWeighted(
                frame[fg], 1.0 - self.mask_alpha,
                colour_mask[fg], self.mask_alpha,
                0,
            )
        return frame

    # ── detection bounding boxes ──────────────────────────────────────

    def draw_detections(
        self,
        frame: np.ndarray,
        instruments: list[dict],
    ) -> np.ndarray:
        """Draw instrument bounding boxes."""
        for inst in instruments:
            bbox = inst.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cls_id = inst.get("class_id", 0)
            conf = inst.get("confidence", 0.0)

            colour = _get_instrument_colour(cls_id)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, self.bbox_thickness)

            # Label: generic "Instrument" or full class name
            if self.show_class_labels:
                cls_name = inst.get("class_name", "Instrument")
                label = f"{cls_name} {conf:.0%}" if self.show_confidence else cls_name
            else:
                label = f"Instrument {conf:.0%}" if self.show_confidence else "Instrument"

            # Label background
            (tw, th), baseline = cv2.getTextSize(
                label, self._font, self.font_scale, 1
            )
            label_y = max(y1 - 6, th + 4)
            cv2.rectangle(
                frame,
                (x1, label_y - th - 4),
                (x1 + tw + 4, label_y + baseline),
                colour, -1,
            )
            cv2.putText(
                frame, label,
                (x1 + 2, label_y - 2),
                self._font, self.font_scale,
                (0, 0, 0), 1, cv2.LINE_AA,
            )

        return frame

    # ── phase label ───────────────────────────────────────────────────

    def draw_phase(
        self,
        frame: np.ndarray,
        phase: dict | None,
    ) -> np.ndarray:
        """Draw the surgical phase label and confidence bar at top-left."""
        if phase is None:
            return frame

        phase_name = phase.get("phase_name", "Unknown")
        phase_id = phase.get("phase_id", 0)
        confidence = phase.get("confidence", 0.0)

        # Phase colour
        colour = PHASE_COLOURS_BGR[phase_id % len(PHASE_COLOURS_BGR)]

        # Build label
        label = f"Phase: {phase_name}"
        if self.show_confidence:
            label += f"  ({confidence:.0%})"

        scale = self.font_scale * 1.1
        (tw, th), baseline = cv2.getTextSize(label, self._font, scale, 2)

        # Panel background
        panel_h = th + baseline + 16
        panel_w = tw + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Confidence bar
        bar_x = 8
        bar_y = panel_h - 6
        bar_w = panel_w - 16
        cv2.rectangle(frame, (bar_x, bar_y - 3), (bar_x + bar_w, bar_y), (60, 60, 60), -1)
        filled = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y - 3), (bar_x + filled, bar_y), colour, -1)

        # Text
        cv2.putText(
            frame, label,
            (8, th + 8),
            self._font, scale,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

        return frame

    # ── frame info (timestamp, frame idx) ─────────────────────────────

    def draw_frame_info(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        video_id: str = "",
    ) -> np.ndarray:
        """Draw frame number and timestamp at top-right."""
        h, w = frame.shape[:2]

        # Format timestamp as mm:ss.f
        mins = int(timestamp_sec // 60)
        secs = timestamp_sec % 60
        ts_str = f"{mins:02d}:{secs:05.2f}"

        label = f"{video_id}  #{frame_idx}  {ts_str}"
        scale = self.font_scale * 0.9
        (tw, th), baseline = cv2.getTextSize(label, self._font, scale, 1)

        # Background
        x0 = w - tw - 16
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, 0), (w, th + baseline + 12), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame, label,
            (x0 + 8, th + 6),
            self._font, scale,
            (200, 200, 200), 1, cv2.LINE_AA,
        )
        return frame

    # ── explanation text panel ─────────────────────────────────────────

    def draw_explanation(
        self,
        frame: np.ndarray,
        explanation: str | None,
        grounded: bool = True,
    ) -> np.ndarray:
        """Draw the explanation text in a panel at the bottom.

        If *explanation* is None or empty, the last non-empty explanation
        is re-used so the text stays stable across frames.
        """
        if explanation:
            self._last_explanation = explanation
        explanation = self._last_explanation
        if not explanation:
            return frame

        h, w = frame.shape[:2]
        max_chars = int(w / (self.font_scale * 18))
        lines = _wrap_text(explanation, max_chars)
        lines = lines[: self.explanation_lines]

        scale = self.font_scale * 0.85
        line_height = int(cv2.getTextSize("Tg", self._font, scale, 1)[0][1] * 1.8)
        panel_h = line_height * len(lines) + 16

        # Semi-transparent dark panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Grounding indicator
        if not grounded:
            cv2.rectangle(
                frame, (0, h - panel_h), (4, h), (0, 0, 255), -1
            )

        # Text lines
        text_colour = (220, 220, 220) if grounded else (140, 140, 255)
        y = h - panel_h + line_height
        for line in lines:
            cv2.putText(
                frame, line,
                (10, y),
                self._font, scale,
                text_colour, 1, cv2.LINE_AA,
            )
            y += line_height

        return frame

    # ── instrument count badge ────────────────────────────────────────

    def draw_instrument_count(
        self,
        frame: np.ndarray,
        count: int,
    ) -> np.ndarray:
        """Draw a small badge showing the instrument count."""
        if count == 0:
            return frame

        h, w = frame.shape[:2]
        label = f"{count} instrument{'s' if count != 1 else ''}"
        scale = self.font_scale * 0.8
        (tw, th), baseline = cv2.getTextSize(label, self._font, scale, 1)

        # Position below the phase panel
        y0 = 50
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y0), (tw + 16, y0 + th + baseline + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(
            frame, label,
            (8, y0 + th + 4),
            self._font, scale,
            (0, 255, 0), 1, cv2.LINE_AA,
        )
        return frame

    # ── phase title card ───────────────────────────────────────────────

    def render_title_card(
        self,
        frame: np.ndarray,
        phase_name: str,
        phase_id: int,
        explanation: str,
        system_comment: str = "",
    ) -> np.ndarray:
        """Render a CAMMA-style phase title card over a darkened frame.

        Layout (top to bottom, all centred):
        1. Optional system comment (small, dim — from Nemotron)
        2. Phase name (large, bold — canonical)
        3. Separator line (phase colour)
        4. Educational explanation (body — canonical)
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Darken background
        overlay = np.zeros_like(out)
        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)

        # Phase colour accent line at top
        colour = PHASE_COLOURS_BGR[phase_id % len(PHASE_COLOURS_BGR)]
        cv2.rectangle(out, (0, 0), (w, 4), colour, -1)

        # Row 1: System comment (small, dim, optional)
        comment_bottom_y = int(h * 0.28)
        if system_comment:
            comment_scale = self.font_scale * 0.7
            (cw, ch), _ = cv2.getTextSize(system_comment, self._font, comment_scale, 1)
            cx = (w - cw) // 2
            cv2.putText(
                out, system_comment, (cx, comment_bottom_y),
                self._font, comment_scale,
                (150, 150, 150), 1, cv2.LINE_AA,
            )

        # Row 2: Phase name (large, centred)
        title_scale = self.font_scale * 2.2
        (tw, th), _ = cv2.getTextSize(phase_name, self._font, title_scale, 3)
        tx = (w - tw) // 2
        ty = comment_bottom_y + th + 20
        cv2.putText(
            out, phase_name, (tx, ty),
            self._font, title_scale,
            (255, 255, 255), 3, cv2.LINE_AA,
        )

        # Thin separator line under title
        sep_y = ty + 14
        sep_margin = int(w * 0.2)
        cv2.line(out, (sep_margin, sep_y), (w - sep_margin, sep_y), colour, 1, cv2.LINE_AA)

        # Row 3: Canonical explanation text (wrapped, centred)
        desc_scale = self.font_scale * 1.0
        max_chars = int(w / (desc_scale * 18))
        lines = _wrap_text(explanation, max_chars)
        (_, lh), _ = cv2.getTextSize("Tg", self._font, desc_scale, 1)
        line_height = int(lh * 2.0)
        start_y = sep_y + 30

        for line in lines:
            (lw, _), _ = cv2.getTextSize(line, self._font, desc_scale, 1)
            lx = (w - lw) // 2
            cv2.putText(
                out, line, (lx, start_y),
                self._font, desc_scale,
                (210, 210, 210), 1, cv2.LINE_AA,
            )
            start_y += line_height

        return out

    # ── composite render ──────────────────────────────────────────────

    def render_frame(
        self,
        frame: np.ndarray,
        scene: dict,
        explanation: str | None = None,
        grounded: bool = True,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render all overlays onto a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR video frame.
        scene : dict
            SceneState dict for this frame.
        explanation : str | None
            Nemotron explanation text (may be None).
        grounded : bool
            Whether the explanation passed the grounding check.
        mask : np.ndarray | None
            Segmentation mask (H, W uint8).  If None and the scene
            contains ``anatomy.mask_file``, the caller should load it
            before calling this method.
        """
        out = frame.copy()

        # Layer 1: segmentation mask (bottom layer, under everything)
        if mask is not None and self.show_masks:
            out = self.draw_mask(out, mask)

        # Layer 2: detection bounding boxes
        # Skipped by default — the generic COCO model produces misleading
        # boxes on surgical video (e.g. "sandwich" on a gallbladder).
        # Enable with show_detections=True when a surgical instrument
        # model is available.
        if self.show_detections:
            instruments = scene.get("instruments", [])
            if instruments:
                out = self.draw_detections(out, instruments)

        # Layer 3: phase label + confidence bar (top-left)
        out = self.draw_phase(out, scene.get("phase"))

        # Layer 4: instrument count badge
        out = self.draw_instrument_count(out, scene.get("instrument_count", 0))

        # Layer 5: frame info (top-right)
        out = self.draw_frame_info(
            out,
            frame_idx=scene.get("frame_idx", 0),
            timestamp_sec=scene.get("timestamp_sec", 0.0),
            video_id=scene.get("video_id", ""),
        )

        # Layer 6: explanation panel (bottom)
        out = self.draw_explanation(out, explanation, grounded)

        return out
