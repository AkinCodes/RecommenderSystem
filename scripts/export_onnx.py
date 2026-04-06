"""
Export DLRM model to ONNX for optimized inference.

Demonstrates model optimization skills:
- PyTorch -> ONNX conversion
- Input/output shape specification
- Dynamic batching support
- Model validation (compare PyTorch vs ONNX outputs)

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model-path trained_model_movielens.pth --output models/dlrm.onnx
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

# Ensure project root is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.dlrm import DLRMModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "trained_model_movielens.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "dlrm.onnx"


def load_config(config_path: Path) -> dict:
    """Load model configuration from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["model"]


def load_pytorch_model(model_path: Path, cfg: dict) -> DLRMModel:
    """Instantiate and load the trained DLRM model."""
    model = DLRMModel(
        num_features=cfg["num_features"],
        embedding_sizes=cfg["embedding_sizes"],
        mlp_layers=cfg["mlp_layers"],
    )
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loaded PyTorch model from %s", model_path)
    return model


def create_dummy_inputs(cfg: dict, batch_size: int = 1):
    """Create dummy inputs that match the model's expected shapes."""
    continuous = torch.randn(batch_size, cfg["num_features"], dtype=torch.float32)
    # Categorical values must be valid indices into each embedding table
    num_categorical = len(cfg["embedding_sizes"])
    categorical = torch.stack(
        [
            torch.randint(0, cfg["embedding_sizes"][i], (batch_size,))
            for i in range(num_categorical)
        ],
        dim=1,
    )
    return continuous, categorical


def export_to_onnx(
    model: DLRMModel,
    cfg: dict,
    output_path: Path,
    opset_version: int = 17,
) -> None:
    """Export the PyTorch DLRM to ONNX format with dynamic batch size."""
    continuous, categorical = create_dummy_inputs(cfg, batch_size=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting to ONNX (opset %d)...", opset_version)
    start = time.perf_counter()

    torch.onnx.export(
        model,
        (continuous, categorical),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["continuous_features", "categorical_features"],
        output_names=["score"],
        dynamic_axes={
            "continuous_features": {0: "batch_size"},
            "categorical_features": {0: "batch_size"},
            "score": {0: "batch_size"},
        },
    )

    elapsed = (time.perf_counter() - start) * 1000
    logger.info("ONNX export completed in %.1f ms -> %s", elapsed, output_path)


def validate_onnx(onnx_path: Path) -> None:
    """Run the ONNX checker to verify the exported model is well-formed."""
    logger.info("Validating ONNX model structure...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model passed structural validation.")


def compare_outputs(
    pytorch_model: DLRMModel,
    onnx_path: Path,
    cfg: dict,
    n_samples: int = 5,
    atol: float = 1e-5,
) -> bool:
    """Compare PyTorch and ONNX Runtime outputs to ensure numerical equivalence."""
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    all_passed = True
    max_diff_seen = 0.0

    logger.info("Comparing outputs across %d random samples...", n_samples)
    for i in range(n_samples):
        batch_size = np.random.choice([1, 4, 16, 32])
        continuous, categorical = create_dummy_inputs(cfg, batch_size=int(batch_size))

        # PyTorch inference
        with torch.no_grad():
            pt_output = pytorch_model(continuous, categorical).numpy()

        # ONNX Runtime inference
        ort_inputs = {
            "continuous_features": continuous.numpy(),
            "categorical_features": categorical.numpy(),
        }
        ort_output = session.run(["score"], ort_inputs)[0]

        diff = np.max(np.abs(pt_output - ort_output))
        max_diff_seen = max(max_diff_seen, diff)
        passed = diff < atol

        status = "PASS" if passed else "FAIL"
        logger.info(
            "  Sample %d (batch=%d): max_abs_diff=%.2e [%s]",
            i + 1,
            batch_size,
            diff,
            status,
        )

        if not passed:
            all_passed = False

    logger.info(
        "Validation %s (max absolute difference: %.2e, tolerance: %.2e)",
        "PASSED" if all_passed else "FAILED",
        max_diff_seen,
        atol,
    )
    return all_passed


def print_size_comparison(pytorch_path: Path, onnx_path: Path) -> None:
    """Print file size comparison between PyTorch and ONNX models."""
    pt_size = pytorch_path.stat().st_size
    onnx_size = onnx_path.stat().st_size

    def fmt(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.2f} MB"

    ratio = onnx_size / pt_size if pt_size > 0 else float("inf")

    print()
    print("=" * 60)
    print("MODEL SIZE COMPARISON")
    print("=" * 60)
    print(f"  PyTorch (.pth):  {fmt(pt_size):>12}  ({pt_size:,} bytes)")
    print(f"  ONNX    (.onnx): {fmt(onnx_size):>12}  ({onnx_size:,} bytes)")
    print(f"  Size ratio:      {ratio:.2f}x")
    if ratio < 1.0:
        print(f"  ONNX is {(1 - ratio) * 100:.1f}% smaller")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Export trained DLRM to ONNX format."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the trained PyTorch model (.pth).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the model config YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Max acceptable absolute difference between PyTorch and ONNX outputs.",
    )
    args = parser.parse_args()

    # ── Load config and model ─────────────────────────────────────────────
    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)
    if not args.model_path.exists():
        logger.error("Model file not found: %s", args.model_path)
        sys.exit(1)

    cfg = load_config(args.config)
    logger.info(
        "Model config: num_features=%d, embedding_sizes=%s, mlp_layers=%s",
        cfg["num_features"],
        cfg["embedding_sizes"],
        cfg["mlp_layers"],
    )

    model = load_pytorch_model(args.model_path, cfg)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %s", f"{total_params:,}")

    # ── Export ────────────────────────────────────────────────────────────
    export_to_onnx(model, cfg, args.output, opset_version=args.opset)

    # ── Validate structure ────────────────────────────────────────────────
    validate_onnx(args.output)

    # ── Numerical comparison ──────────────────────────────────────────────
    passed = compare_outputs(model, args.output, cfg, n_samples=5, atol=args.tolerance)

    # ── Size comparison ───────────────────────────────────────────────────
    print_size_comparison(args.model_path, args.output)

    if not passed:
        logger.error(
            "Output validation FAILED. ONNX model may not be numerically equivalent."
        )
        sys.exit(1)

    print()
    print("Export complete. ONNX model ready at:")
    print(f"  {args.output}")
    print()


if __name__ == "__main__":
    main()
