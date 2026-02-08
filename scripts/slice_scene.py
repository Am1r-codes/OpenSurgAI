"""Slice a scene JSONL file by frame_idx range.

Usage:
    python scripts/slice_scene.py
"""

import json
from pathlib import Path

SRC = Path("experiments/scene/video49_scene.jsonl")
DST = Path("experiments/scene/video49_scene_slice.jsonl")
START_FRAME = 8500
MAX_FRAMES = 80

kept = 0
with open(SRC) as fin, open(DST, "w") as fout:
    for line in fin:
        obj = json.loads(line)
        if obj["frame_idx"] >= START_FRAME and kept < MAX_FRAMES:
            fout.write(line)
            kept += 1

print(f"Wrote {kept} frames to {DST}")
