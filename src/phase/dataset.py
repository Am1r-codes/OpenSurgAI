"""Cholec80 phase dataset for training.

Simple frame-level dataset. Samples 1 frame per second (every 25 frames).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from src.phase.pipeline import CHOLEC80_PHASES, load_phase_annotations

log = logging.getLogger(__name__)


class Cholec80PhaseDataset(Dataset):
    """Frame-level phase classification dataset.

    Parameters
    ----------
    data_dir : Path
        Cholec80 root with ``videos/`` and ``phase_annotations/``.
    video_ids : list[str]
        Videos to include, e.g. ["video01", "video02"].
    frame_stride : int
        Sample every N-th frame (default 25 = 1 fps at 25fps video).
    img_size : int
        Output crop size.
    """

    def __init__(
        self,
        data_dir: Path,
        video_ids: list[str],
        frame_stride: int = 25,
        img_size: int = 224,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.samples: list[tuple[str, int, int]] = []  # (video_id, frame_idx, phase_id)

        for vid in video_ids:
            ann_path = self.data_dir / "phase_annotations" / f"{vid}-phase.txt"
            if not ann_path.exists():
                log.warning("Missing: %s", ann_path)
                continue

            frame_to_phase = load_phase_annotations(ann_path)
            frames = sorted(frame_to_phase.keys())[::frame_stride]

            for fidx in frames:
                self.samples.append((vid, fidx, frame_to_phase[fidx]))

        log.info("Dataset: %d samples from %d videos", len(self.samples), len(video_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video_id, frame_idx, phase_id = self.samples[idx]

        video_path = self.data_dir / "videos" / f"{video_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = cap.read()
        cap.release()

        if not ret:
            bgr = torch.zeros(3, self.img_size, self.img_size)
            return bgr, phase_id

        # Preprocess: resize, center crop, normalize
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = (self.img_size + 32) / min(h, w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        new_h, new_w = rgb.shape[:2]
        y0, x0 = (new_h - self.img_size) // 2, (new_w - self.img_size) // 2
        rgb = rgb[y0:y0 + self.img_size, x0:x0 + self.img_size]

        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = (t - mean) / std

        return t, phase_id

    def get_class_weights(self) -> torch.Tensor:
        """Inverse frequency weights for imbalanced classes."""
        counts = [0] * len(CHOLEC80_PHASES)
        for _, _, pid in self.samples:
            counts[pid] += 1
        total = sum(counts)
        weights = [total / (len(CHOLEC80_PHASES) * c) if c > 0 else 1.0 for c in counts]
        return torch.tensor(weights, dtype=torch.float32)
