"""Professional surgical HUD overlay renderer.

Renders a modern, clean heads-up display onto surgical video frames
with instrument presence bars, phase timeline, and Nemotron commentary.
Designed to look like a real-time surgical AI monitoring system.

Layout
------
::

    ┌─ ACCENT LINE (2px, teal) ──────────────────────────────────┐
    │ ◈ OpenSurgAI   ▎ PREPARATION ██████░ 99%       12:34.56  │
    ├────────────────────────────────────────────────────────────┤
    │                                                            │
    │ ┌ INSTRUMENTS ─┐                                          │
    │ │ ● Grasper ██ 92%│                                       │
    │ │ ● Bipolar █░ 78%│                                       │
    │ │ ○ Hook    ░░  8%│          (surgical video)             │
    │ │ ○ Scissors░░  3%│                                       │
    │ │ ...            │                                        │
    │ └────────────────┘                                        │
    │                                                            │
    │  [Nemotron explanation text — semi-transparent strip]      │
    │ ┌ PHASE TIMELINE ──────────────────────────────────────┐  │
    │ │ PREP │  CALOT  │ CLIP │ DISSECT │ PKG │CLEAN│ RETRACT│  │
    │ └──────────────────────────────────────────────────────┘  │
    └────────────────────────────────────────────────────────────┘

Colour scheme
-------------
Dark panels (#0a1120) at ~80% opacity, surgical teal accent (#00CED1),
high-contrast white text, instrument-specific colour coding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Colour palettes (BGR for OpenCV)
# ═══════════════════════════════════════════════════════════════════════

# Teal accent – the signature colour
ACCENT = (209, 206, 0)        # #00CED1

# Panel / background
_BG = (26, 17, 10)            # #0A1120
_BAR_TRACK = (55, 48, 40)     # dark grey bar background

# Text
_WHITE = (240, 240, 240)
_DIM = (130, 130, 140)
_SHADOW = (0, 0, 0)

# Instrument colours (BGR) – vibrant on dark background
INSTRUMENT_COLOURS_BGR = [
    (80, 220, 100),    # 0 Grasper     – green
    (230, 190, 50),    # 1 Bipolar     – cyan-blue
    (50, 165, 245),    # 2 Hook        – orange
    (70, 70, 245),     # 3 Scissors    – red
    (220, 80, 220),    # 4 Clipper     – magenta
    (210, 210, 60),    # 5 Irrigator   – teal
    (50, 230, 230),    # 6 SpecimenBag – yellow
]

# Phase colours (BGR) – used in timeline and header
PHASE_COLOURS_BGR = [
    (170, 170, 170),   # 0 Preparation            – silver
    (50, 200, 255),    # 1 CalotTriangleDissection – amber
    (60, 60, 240),     # 2 ClippingCutting         – red
    (80, 220, 80),     # 3 GallbladderDissection   – green
    (255, 180, 50),    # 4 GallbladderPackaging    – sky blue
    (220, 80, 220),    # 5 CleaningCoagulation     – magenta
    (50, 230, 230),    # 6 GallbladderRetraction   – yellow
]

# Phase abbreviations for timeline
PHASE_ABBR = [
    "PREP", "CALOT", "CLIP", "DISSECT", "PKG", "CLEAN", "RETRACT",
]

CHOLEC80_INSTRUMENTS = [
    "Grasper", "Bipolar", "Hook", "Scissors",
    "Clipper", "Irrigator", "SpecimenBag",
]

# Segmentation palette (kept for backward compatibility)
SEG_PALETTE_BGR = np.array([
    [0,0,0],[0,0,128],[0,128,0],[0,128,128],[128,0,0],
    [128,0,128],[128,128,0],[128,128,128],[0,0,64],[0,0,192],
    [0,128,64],[0,128,192],[128,0,64],[128,0,192],[128,128,64],
    [128,128,192],[0,64,0],[0,64,128],[0,192,0],[0,192,128],
    [128,64,0],
], dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

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


def _alpha_rect(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Draw a filled rectangle with alpha blending (in-place)."""
    x1, y1 = pt1
    x2, y2 = pt2
    sub = frame[y1:y2, x1:x2]
    rect = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)


def _text_shadow(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    font: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Draw text with a subtle drop shadow for readability."""
    sx, sy = org[0] + 1, org[1] + 1
    cv2.putText(frame, text, (sx, sy), font, scale, _SHADOW, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════
# Main renderer
# ═══════════════════════════════════════════════════════════════════════

class OverlayRenderer:
    """Professional surgical HUD overlay renderer.

    Parameters
    ----------
    mask_alpha : float
        Opacity of segmentation mask overlay.
    font_scale : float
        Base font scale (adjusted per element).
    show_confidence : bool
        Show confidence percentages in overlays.
    show_masks : bool
        Render segmentation masks.
    show_detections : bool
        Render detection bounding boxes (legacy YOLO mode).
    show_class_labels : bool
        Show class labels on bounding boxes.
    tool_threshold : float
        Confidence threshold to consider a tool "detected" (bright vs dim).
    explanation_lines : int
        Max lines of explanation text.
    bbox_thickness : int
        Bounding box line thickness (legacy mode).
    """

    def __init__(
        self,
        mask_alpha: float = 0.35,
        font_scale: float = 0.6,
        show_confidence: bool = True,
        show_masks: bool = False,
        show_detections: bool = False,
        show_class_labels: bool = False,
        tool_threshold: float = 0.40,
        explanation_lines: int = 3,
        bbox_thickness: int = 2,
    ) -> None:
        self.mask_alpha = mask_alpha
        self.font_scale = font_scale
        self.show_confidence = show_confidence
        self.show_masks = show_masks
        self.show_detections = show_detections
        self.show_class_labels = show_class_labels
        self.tool_threshold = tool_threshold
        self.explanation_lines = explanation_lines
        self.bbox_thickness = bbox_thickness

        self._font = cv2.FONT_HERSHEY_DUPLEX
        self._font_sm = cv2.FONT_HERSHEY_SIMPLEX
        self._last_explanation: str | None = None

        # Phase timeline (set by recorder via set_phase_timeline)
        self._phase_segments: list[dict] | None = None
        self._total_frames: int = 0

    # ── phase timeline setup ───────────────────────────────────────

    def set_phase_timeline(
        self,
        segments: list[dict],
        total_frames: int,
    ) -> None:
        """Set the pre-computed phase timeline for the bottom bar.

        Parameters
        ----------
        segments : list[dict]
            Each dict: {"start": int, "end": int, "phase_id": int, "phase_name": str}
        total_frames : int
            Total frame count of the video.
        """
        self._phase_segments = segments
        self._total_frames = total_frames

    # ── segmentation mask overlay ──────────────────────────────────

    def draw_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Alpha-blend a class-index mask onto the frame."""
        h, w = frame.shape[:2]
        mh, mw = mask.shape[:2]
        if (mh, mw) != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        colour_mask = SEG_PALETTE_BGR[mask]
        fg = mask > 0
        if fg.any():
            frame[fg] = cv2.addWeighted(
                frame[fg], 1.0 - self.mask_alpha,
                colour_mask[fg], self.mask_alpha, 0,
            )
        return frame

    # ── legacy detection boxes (for YOLO mode) ─────────────────────

    def draw_detections(self, frame: np.ndarray, instruments: list[dict]) -> np.ndarray:
        """Draw instrument bounding boxes (legacy YOLO mode)."""
        for inst in instruments:
            bbox = inst.get("bbox", [])
            if len(bbox) != 4 or all(v == 0 for v in bbox):
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cls_id = inst.get("class_id", 0)
            colour = INSTRUMENT_COLOURS_BGR[cls_id % len(INSTRUMENT_COLOURS_BGR)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, self.bbox_thickness)
            if self.show_class_labels:
                label = inst.get("class_name", "Instrument")
                if self.show_confidence:
                    label += f" {inst.get('confidence', 0):.0%}"
                _text_shadow(frame, label, (x1 + 2, y1 - 6), self._font_sm, self.font_scale * 0.7, colour)
        return frame

    # ══════════════════════════════════════════════════════════════════
    # HUD COMPONENTS
    # ══════════════════════════════════════════════════════════════════

    def _draw_header(
        self,
        frame: np.ndarray,
        phase: dict | None,
        timestamp_sec: float,
        frame_idx: int,
        video_id: str,
    ) -> None:
        """Draw the top header bar: branding + phase + timestamp."""
        h, w = frame.shape[:2]
        header_h = max(34, int(h * 0.07))

        # Panel background
        _alpha_rect(frame, (0, 0), (w, header_h), _BG, 0.82)

        # Top accent line (2px teal)
        cv2.rectangle(frame, (0, 0), (w, 2), ACCENT, -1)

        # Scale fonts relative to frame height
        s = h / 480.0
        brand_scale = 0.50 * s
        phase_scale = 0.48 * s
        time_scale = 0.42 * s

        # ── Left: branding ─────────────────────────────────────────
        brand_y = int(header_h * 0.65)
        # Diamond accent
        _text_shadow(frame, "OpenSurgAI", (int(10 * s), brand_y),
                     self._font, brand_scale, ACCENT, max(1, int(s)))

        # ── Center: phase name + confidence bar ────────────────────
        if phase:
            phase_name = phase.get("phase_name", "Unknown")
            phase_id = phase.get("phase_id", 0)
            confidence = phase.get("confidence", 0.0)
            p_color = PHASE_COLOURS_BGR[phase_id % len(PHASE_COLOURS_BGR)]

            # Phase indicator dot
            dot_x = int(w * 0.30)
            dot_y = int(header_h * 0.50)
            cv2.circle(frame, (dot_x, dot_y), int(4 * s), p_color, -1, cv2.LINE_AA)

            # Phase name
            _text_shadow(frame, phase_name, (dot_x + int(10 * s), brand_y),
                         self._font_sm, phase_scale, _WHITE, max(1, int(s)))

            # Confidence bar
            bar_x = int(w * 0.58)
            bar_y = int(header_h * 0.35)
            bar_w = int(w * 0.14)
            bar_h = int(header_h * 0.28)

            # Track
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          _BAR_TRACK, -1)
            # Fill
            fill_w = int(bar_w * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          p_color, -1)
            # Percentage
            pct_text = f"{confidence:.0%}"
            _text_shadow(frame, pct_text, (bar_x + bar_w + int(6 * s), bar_y + bar_h),
                         self._font_sm, time_scale * 0.9, _DIM)

        # ── Right: timestamp ───────────────────────────────────────
        mins = int(timestamp_sec // 60)
        secs = timestamp_sec % 60
        ts_str = f"{mins:02d}:{secs:05.2f}"
        (tw, _), _ = cv2.getTextSize(ts_str, self._font_sm, time_scale, 1)
        _text_shadow(frame, ts_str, (w - tw - int(10 * s), brand_y),
                     self._font_sm, time_scale, _DIM)

    # ── instrument presence panel ──────────────────────────────────

    def _draw_tool_panel(
        self,
        frame: np.ndarray,
        instruments: list[dict],
    ) -> None:
        """Draw the left-side instrument presence panel."""
        h, w = frame.shape[:2]
        s = h / 480.0
        header_h = max(34, int(h * 0.07))

        panel_w = int(165 * s)
        row_h = int(22 * s)
        n_tools = len(CHOLEC80_INSTRUMENTS)
        panel_h = int(20 * s) + n_tools * row_h + int(6 * s)
        panel_x = 0
        panel_y = header_h + int(8 * s)

        # Panel background
        _alpha_rect(frame, (panel_x, panel_y),
                    (panel_x + panel_w, panel_y + panel_h), _BG, 0.78)

        # Left accent strip
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + int(2 * s), panel_y + panel_h), ACCENT, -1)

        # Title
        title_scale = 0.35 * s
        title_y = panel_y + int(14 * s)
        _text_shadow(frame, "INSTRUMENTS", (panel_x + int(10 * s), title_y),
                     self._font_sm, title_scale, ACCENT)

        # Build confidence map from instruments list
        tool_confs: dict[int, float] = {}
        for inst in instruments:
            cid = inst.get("class_id", -1)
            conf = inst.get("confidence", 0.0)
            tool_confs[cid] = conf

        # Draw each instrument row
        name_scale = 0.34 * s
        pct_scale = 0.30 * s
        bar_w = int(50 * s)
        bar_h = int(8 * s)

        for i, tool_name in enumerate(CHOLEC80_INSTRUMENTS):
            y = panel_y + int(20 * s) + i * row_h + int(row_h * 0.7)
            conf = tool_confs.get(i, 0.0)
            detected = conf >= self.tool_threshold
            color = INSTRUMENT_COLOURS_BGR[i]

            # Indicator dot
            dot_x = panel_x + int(12 * s)
            dot_y = y - int(4 * s)
            if detected:
                cv2.circle(frame, (dot_x, dot_y), int(3 * s), color, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (dot_x, dot_y), int(3 * s), _BAR_TRACK, 1, cv2.LINE_AA)

            # Tool name
            name_color = _WHITE if detected else _DIM
            # Truncate SpecimenBag for display
            display_name = tool_name[:8] if len(tool_name) > 8 else tool_name
            _text_shadow(frame, display_name, (panel_x + int(22 * s), y),
                         self._font_sm, name_scale, name_color)

            # Confidence bar
            bx = panel_x + int(90 * s)
            by = y - bar_h + int(1 * s)
            # Track
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), _BAR_TRACK, -1)
            # Fill
            fill_w = max(1, int(bar_w * conf))
            bar_color = color if detected else (80, 80, 90)
            cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), bar_color, -1)

            # Percentage (only for detected)
            if detected:
                pct = f"{conf:.0%}"
                _text_shadow(frame, pct, (bx + bar_w + int(4 * s), y),
                             self._font_sm, pct_scale, color)

    # ── phase timeline ─────────────────────────────────────────────

    def _draw_timeline(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> None:
        """Draw the bottom phase timeline bar."""
        if not self._phase_segments or self._total_frames <= 0:
            return

        h, w = frame.shape[:2]
        s = h / 480.0
        tl_h = int(26 * s)
        tl_y = h - tl_h
        margin = int(6 * s)

        # Panel background
        _alpha_rect(frame, (0, tl_y), (w, h), _BG, 0.82)

        # Bottom accent line
        cv2.rectangle(frame, (0, h - int(2 * s)), (w, h), ACCENT, -1)

        # Draw segments
        label_scale = 0.28 * s
        usable_w = w - 2 * margin

        for seg in self._phase_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            pid = seg["phase_id"]
            p_color = PHASE_COLOURS_BGR[pid % len(PHASE_COLOURS_BGR)]

            # Pixel positions
            x1 = margin + int(usable_w * seg_start / self._total_frames)
            x2 = margin + int(usable_w * seg_end / self._total_frames)
            seg_y = tl_y + int(4 * s)
            seg_h = int(10 * s)

            # Is this the active segment?
            active = seg_start <= frame_idx < seg_end

            # Segment bar
            if active:
                cv2.rectangle(frame, (x1, seg_y), (x2, seg_y + seg_h), p_color, -1)
            else:
                # Dimmed
                dim_color = tuple(max(30, c // 3) for c in p_color)
                cv2.rectangle(frame, (x1, seg_y), (x2, seg_y + seg_h), dim_color, -1)

            # Segment border (subtle)
            cv2.rectangle(frame, (x1, seg_y), (x2, seg_y + seg_h), (50, 50, 50), 1)

            # Label (if segment is wide enough)
            abbr = PHASE_ABBR[pid] if pid < len(PHASE_ABBR) else f"P{pid}"
            (tw, _), _ = cv2.getTextSize(abbr, self._font_sm, label_scale, 1)
            if (x2 - x1) > tw + 4:
                label_x = (x1 + x2 - tw) // 2
                label_y = seg_y + seg_h + int(10 * s)
                label_color = _WHITE if active else _DIM
                cv2.putText(frame, abbr, (label_x, label_y),
                            self._font_sm, label_scale, label_color, 1, cv2.LINE_AA)

        # Current position marker (bright vertical line)
        cursor_x = margin + int(usable_w * frame_idx / self._total_frames)
        seg_y = tl_y + int(2 * s)
        cv2.line(frame, (cursor_x, seg_y), (cursor_x, tl_y + int(16 * s)),
                 _WHITE, max(1, int(2 * s)), cv2.LINE_AA)
        # Small triangle at top
        tri_size = int(4 * s)
        pts = np.array([
            [cursor_x - tri_size, seg_y],
            [cursor_x + tri_size, seg_y],
            [cursor_x, seg_y + tri_size],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts], _WHITE, cv2.LINE_AA)

    # ── explanation text ───────────────────────────────────────────

    def draw_explanation(
        self,
        frame: np.ndarray,
        explanation: str | None,
        grounded: bool = True,
    ) -> np.ndarray:
        """Draw Nemotron explanation as a subtle text strip."""
        if explanation:
            self._last_explanation = explanation
        explanation = self._last_explanation
        if not explanation:
            return frame

        h, w = frame.shape[:2]
        s = h / 480.0
        tl_h = int(26 * s) if self._phase_segments else 0
        scale = 0.35 * s
        max_chars = int(w / (scale * 20))
        lines = _wrap_text(explanation, max_chars)[:self.explanation_lines]

        line_height = int(cv2.getTextSize("Tg", self._font_sm, scale, 1)[0][1] * 1.8)
        panel_h = line_height * len(lines) + int(10 * s)
        panel_y = h - tl_h - panel_h

        # Semi-transparent strip
        _alpha_rect(frame, (0, panel_y), (w, panel_y + panel_h), _BG, 0.70)

        # Left accent
        if not grounded:
            cv2.rectangle(frame, (0, panel_y), (int(3 * s), panel_y + panel_h),
                          (60, 60, 240), -1)

        text_color = (200, 210, 210) if grounded else (160, 140, 240)
        y = panel_y + line_height
        for line in lines:
            cv2.putText(frame, line, (int(10 * s), y),
                        self._font_sm, scale, text_color, 1, cv2.LINE_AA)
            y += line_height

        return frame

    # ── instrument count (small badge for compatibility) ───────────

    def draw_instrument_count(self, frame: np.ndarray, count: int) -> np.ndarray:
        """No-op in new HUD (tool panel shows this). Kept for API compat."""
        return frame

    def draw_phase(self, frame: np.ndarray, phase: dict | None) -> np.ndarray:
        """No-op in new HUD (header shows this). Kept for API compat."""
        return frame

    def draw_frame_info(self, frame: np.ndarray, frame_idx: int,
                        timestamp_sec: float, video_id: str = "") -> np.ndarray:
        """No-op in new HUD (header shows this). Kept for API compat."""
        return frame

    # ── title card ─────────────────────────────────────────────────

    def render_title_card(
        self,
        frame: np.ndarray,
        phase_name: str,
        phase_id: int,
        explanation: str,
        system_comment: str = "",
    ) -> np.ndarray:
        """Cinematic phase transition title card.

        Dark overlay with phase colour accent, large phase name,
        and educational description.
        """
        out = frame.copy()
        h, w = out.shape[:2]
        s = h / 480.0

        # Heavy darken
        dark = np.zeros_like(out)
        cv2.addWeighted(dark, 0.75, out, 0.25, 0, out)

        p_color = PHASE_COLOURS_BGR[phase_id % len(PHASE_COLOURS_BGR)]

        # Top and bottom accent lines
        cv2.rectangle(out, (0, 0), (w, int(3 * s)), p_color, -1)
        cv2.rectangle(out, (0, h - int(3 * s)), (w, h), p_color, -1)

        # Phase number badge
        badge_text = f"PHASE {phase_id + 1}/7"
        badge_scale = 0.38 * s
        (bw, bh), _ = cv2.getTextSize(badge_text, self._font_sm, badge_scale, 1)
        badge_x = (w - bw) // 2
        badge_y = int(h * 0.30)
        cv2.putText(out, badge_text, (badge_x, badge_y),
                    self._font_sm, badge_scale, p_color, max(1, int(s)), cv2.LINE_AA)

        # Phase name (large, centred)
        title_scale = 1.4 * s
        (tw, th), _ = cv2.getTextSize(phase_name, self._font, title_scale, max(2, int(2 * s)))
        tx = (w - tw) // 2
        ty = badge_y + th + int(20 * s)
        # Shadow
        cv2.putText(out, phase_name, (tx + 2, ty + 2),
                    self._font, title_scale, _SHADOW, max(3, int(3 * s)), cv2.LINE_AA)
        cv2.putText(out, phase_name, (tx, ty),
                    self._font, title_scale, _WHITE, max(2, int(2 * s)), cv2.LINE_AA)

        # Separator line
        sep_y = ty + int(16 * s)
        margin = int(w * 0.22)
        cv2.line(out, (margin, sep_y), (w - margin, sep_y), p_color, max(1, int(s)), cv2.LINE_AA)

        # Explanation text (wrapped, centred)
        desc_scale = 0.42 * s
        max_chars = int(w / (desc_scale * 18))
        lines = _wrap_text(explanation, max_chars)
        (_, lh), _ = cv2.getTextSize("Tg", self._font_sm, desc_scale, 1)
        line_h = int(lh * 2.2)
        start_y = sep_y + int(28 * s)

        for line in lines:
            (lw, _), _ = cv2.getTextSize(line, self._font_sm, desc_scale, 1)
            lx = (w - lw) // 2
            cv2.putText(out, line, (lx, start_y),
                        self._font_sm, desc_scale, (210, 210, 210), 1, cv2.LINE_AA)
            start_y += line_h

        # System comment (small, dim, below explanation)
        if system_comment:
            # Strip <think> tags
            if "<think>" in system_comment:
                import re
                system_comment = re.sub(r"<think>.*?</think>\s*", "", system_comment, flags=re.DOTALL)
            system_comment = system_comment.strip()
            if system_comment:
                sc_scale = 0.32 * s
                (scw, _), _ = cv2.getTextSize(system_comment, self._font_sm, sc_scale, 1)
                scx = (w - scw) // 2
                scy = start_y + int(10 * s)
                cv2.putText(out, system_comment, (scx, scy),
                            self._font_sm, sc_scale, (140, 140, 140), 1, cv2.LINE_AA)

        return out

    # ══════════════════════════════════════════════════════════════════
    # COMPOSITE RENDER
    # ══════════════════════════════════════════════════════════════════

    def render_frame(
        self,
        frame: np.ndarray,
        scene: dict,
        explanation: str | None = None,
        grounded: bool = True,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render the full surgical HUD onto a video frame.

        Composites all overlay elements in the correct order:
        1. Segmentation mask (if available)
        2. Detection boxes (legacy YOLO mode)
        3. Header bar (branding + phase + timestamp)
        4. Instrument panel (left side)
        5. Phase timeline (bottom)
        6. Explanation text (above timeline)
        """
        out = frame.copy()

        # Layer 1: segmentation mask
        if mask is not None and self.show_masks:
            out = self.draw_mask(out, mask)

        # Layer 2: detection boxes (legacy YOLO)
        if self.show_detections:
            instruments = scene.get("instruments", [])
            if instruments:
                out = self.draw_detections(out, instruments)

        # Layer 3: header bar
        phase = scene.get("phase")
        self._draw_header(
            out, phase,
            timestamp_sec=scene.get("timestamp_sec", 0.0),
            frame_idx=scene.get("frame_idx", 0),
            video_id=scene.get("video_id", ""),
        )

        # Layer 4: instrument panel
        self._draw_tool_panel(out, scene.get("instruments", []))

        # Layer 5: explanation text (above timeline)
        self.draw_explanation(out, explanation, grounded)

        # Layer 6: phase timeline
        self._draw_timeline(out, scene.get("frame_idx", 0))

        return out
