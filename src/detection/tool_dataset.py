"""Cholec80 tool presence dataset for multi-label training.

Each sample is a video frame paired with a 7-dim binary vector indicating
which surgical instruments are present.  Annotations are provided at 1 fps
(every 25th frame) in the Cholec80 tool_annotations directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

CHOLEC80_INSTRUMENTS = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]
NUM_TOOLS = len(CHOLEC80_INSTRUMENTS)


def load_tool_annotations(ann_path: Path) -> dict[int, list[int]]:
    """Load ``videoXX-tool.txt`` into {frame_idx: [7 binary labels]}.

    File format is tab-separated::

        Frame  Grasper  Bipolar  Hook  Scissors  Clipper  Irrigator  SpecimenBag
        0      1        0        0     0         0        0          0
        25     1        0        0     0         0        0          0
    """
    mapping: dict[int, list[int]] = {}
    with open(ann_path) as fh:
        fh.readline()  # skip header
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            frame_idx = int(parts[0])
            labels = [int(x) for x in parts[1:8]]
            mapping[frame_idx] = labels
    return mapping


class Cholec80ToolDataset(Dataset):
    """Frame-level multi-label tool presence dataset.

    Parameters
    ----------
    data_dir : Path
        Cholec80 root with ``videos/`` and ``tool_annotations/``.
    video_ids : list[str]
        Videos to include, e.g. ["video01", "video02"].
    frame_stride : int
        Sample every N-th annotated frame (default 1 = use all annotated).
    img_size : int
        Output crop size.
    """

    def __init__(
        self,
        data_dir: Path,
        video_ids: list[str],
        frame_stride: int = 1,
        img_size: int = 224,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.samples: list[tuple[str, int, list[int]]] = []

        for vid in video_ids:
            ann_path = self.data_dir / "tool_annotations" / f"{vid}-tool.txt"
            if not ann_path.exists():
                log.warning("Missing: %s", ann_path)
                continue

            frame_to_tools = load_tool_annotations(ann_path)
            frames = sorted(frame_to_tools.keys())[::frame_stride]

            for fidx in frames:
                self.samples.append((vid, fidx, frame_to_tools[fidx]))

        log.info("Tool dataset: %d samples from %d videos", len(self.samples), len(video_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_id, frame_idx, labels = self.samples[idx]

        video_path = self.data_dir / "videos" / f"{video_id}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = cap.read()
        cap.release()

        target = torch.tensor(labels, dtype=torch.float32)

        if not ret:
            dummy = torch.zeros(3, self.img_size, self.img_size)
            return dummy, target

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

        return t, target

    def get_pos_weights(self) -> torch.Tensor:
        """Compute positive class weights for BCEWithLogitsLoss.

        Returns weights = num_negative / num_positive per tool.
        """
        pos_counts = [0.0] * NUM_TOOLS
        total = len(self.samples)
        for _, _, labels in self.samples:
            for i, v in enumerate(labels):
                pos_counts[i] += v
        weights = []
        for pc in pos_counts:
            if pc > 0:
                weights.append((total - pc) / pc)
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float32)
