#!/usr/bin/env python3
"""Export ResNet50 models to TensorRT-optimised TorchScript.

Converts the phase recognition and/or tool classifier models to
TensorRT-compiled versions for faster GPU inference via
torch_tensorrt.

Usage:
    # Export phase model
    python scripts/export_tensorrt.py --phase-weights weights/phase_resnet50.pt

    # Export tool model
    python scripts/export_tensorrt.py --tool-weights weights/tool_resnet50.pt

    # Export both
    python scripts/export_tensorrt.py \
        --phase-weights weights/phase_resnet50.pt \
        --tool-weights weights/tool_resnet50.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def export_model(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 224, 224),
    half: bool = True,
) -> Path:
    """Compile a model with TensorRT and save as TorchScript."""
    import torch_tensorrt

    model = model.eval().cuda()
    dtype = torch.float16 if half else torch.float32

    log.info("Compiling with TensorRT  (input=%s, dtype=%s) ...", input_shape, dtype)
    t0 = time.perf_counter()

    compiled = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[torch_tensorrt.Input(shape=input_shape, dtype=dtype)],
        min_block_size=1,
        use_explicit_typing=False,
        enabled_precisions={dtype},
    )

    elapsed = time.perf_counter() - t0
    log.info("TensorRT compilation done in %.1fs", elapsed)

    # Benchmark
    dummy = torch.randn(*input_shape, device="cuda", dtype=dtype)
    # Warmup
    for _ in range(10):
        compiled(dummy)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        compiled(dummy)
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - t0) / n_iters * 1000
    log.info("TensorRT inference: %.2f ms/frame  (%.0f FPS)", avg_ms, 1000 / avg_ms)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch_tensorrt.save(compiled, str(output_path), inputs=[dummy])
    log.info("Saved TensorRT model -> %s  (%.1f MB)",
             output_path, output_path.stat().st_size / 1e6)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export models to TensorRT")
    parser.add_argument("--phase-weights", type=Path, default=None,
                        help="Path to phase ResNet50 weights")
    parser.add_argument("--tool-weights", type=Path, default=None,
                        help="Path to tool ResNet50 weights")
    parser.add_argument("--output-dir", type=Path,
                        default=_PROJECT_ROOT / "weights" / "tensorrt",
                        help="Output directory for TRT models")
    parser.add_argument("--no-half", action="store_true",
                        help="Use FP32 instead of FP16")
    args = parser.parse_args()

    if args.phase_weights is None and args.tool_weights is None:
        log.error("Provide at least one of --phase-weights or --tool-weights")
        sys.exit(1)

    half = not args.no_half

    from src.phase.pipeline import build_phase_model, NUM_PHASES
    from src.detection.tool_dataset import NUM_TOOLS

    if args.phase_weights:
        log.info("=" * 60)
        log.info("Exporting phase model: %s", args.phase_weights)
        model = build_phase_model(num_classes=NUM_PHASES, weights_path=args.phase_weights)
        export_model(
            model,
            args.output_dir / "phase_resnet50_trt.ts",
            input_shape=(1, 3, 224, 224),
            half=half,
        )

    if args.tool_weights:
        log.info("=" * 60)
        log.info("Exporting tool model: %s", args.tool_weights)
        from scripts.train_tool import build_tool_model
        model = build_tool_model(num_tools=NUM_TOOLS, weights_path=args.tool_weights)
        export_model(
            model,
            args.output_dir / "tool_resnet50_trt.ts",
            input_shape=(1, 3, 224, 224),
            half=half,
        )

    log.info("=" * 60)
    log.info("All exports complete.")


if __name__ == "__main__":
    main()
