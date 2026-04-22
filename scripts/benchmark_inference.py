"""Benchmark PyTorch vs ONNX Runtime inference latency.

Usage:
    uv run python scripts/benchmark_inference.py
    uv run python scripts/benchmark_inference.py --onnx-path models/dlrm.onnx
"""

import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.dlrm import DLRMModel

# Defaults
NUM_FEATURES = 2
EMBEDDING_SIZES = [943, 1682]
MLP_LAYERS = [128, 64, 32]

BATCH_SIZES = [1, 8, 32, 64, 128, 256]
WARMUP_RUNS = 50
BENCHMARK_RUNS = 500


def benchmark_pytorch(model, batch_size, num_features, warmup=WARMUP_RUNS, runs=BENCHMARK_RUNS):
    """Benchmark PyTorch inference and return latencies in ms."""
    model.eval()
    cont = torch.randn(batch_size, num_features)
    cat = torch.randint(0, 100, (batch_size, 2))

    with torch.no_grad():
        for _ in range(warmup):
            model(cont, cat)

        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(cont, cat)
            latencies.append((time.perf_counter() - t0) * 1000)

    return np.array(latencies)


def benchmark_onnx(session, batch_size, num_features, warmup=WARMUP_RUNS, runs=BENCHMARK_RUNS):
    """Benchmark ONNX Runtime inference and return latencies in ms."""
    cont = np.random.randn(batch_size, num_features).astype(np.float32)
    cat = np.random.randint(0, 100, (batch_size, 2)).astype(np.int64)

    feeds = {
        "continuous_features": cont,
        "categorical_features": cat,
    }

    for _ in range(warmup):
        session.run(None, feeds)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, feeds)
        latencies.append((time.perf_counter() - t0) * 1000)

    return np.array(latencies)


def format_table(results):
    """Format results as a markdown table."""
    header = "| Batch Size | Runtime | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) | Throughput (samples/s) |"
    sep = "|------------|---------|-----------|-------------|----------|----------|----------------------|"
    rows = [header, sep]

    for r in results:
        throughput = r["batch_size"] / (r["mean_ms"] / 1000) if r["mean_ms"] > 0 else 0
        rows.append(
            f"| {r['batch_size']:>10} | {r['runtime']:<7} "
            f"| {r['mean_ms']:>9.3f} | {r['median_ms']:>11.3f} "
            f"| {r['p95_ms']:>8.3f} | {r['p99_ms']:>8.3f} "
            f"| {throughput:>20,.0f} |"
        )

    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "trained_model_movielens.pth"),
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "models", "dlrm.onnx"),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "BENCHMARK.md"),
    )
    args = parser.parse_args()

    # Load PyTorch model
    print("Loading PyTorch model...")
    pt_model = DLRMModel(
        num_features=NUM_FEATURES,
        embedding_sizes=EMBEDDING_SIZES,
        mlp_layers=MLP_LAYERS,
    )
    if os.path.exists(args.model_path):
        pt_model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))
        print(f"  Loaded weights from {args.model_path}")
    else:
        print("  Using random weights (no checkpoint found)")

    # Load ONNX model
    if not os.path.exists(args.onnx_path):
        print(f"ONNX model not found at {args.onnx_path}. Run export_onnx.py first.")
        sys.exit(1)

    print(f"Loading ONNX model from {args.onnx_path}...")
    ort_session = ort.InferenceSession(args.onnx_path)

    # Benchmark
    print(f"\nBenchmarking across batch sizes: {BATCH_SIZES}")
    print(f"  Warmup: {WARMUP_RUNS} runs, Benchmark: {BENCHMARK_RUNS} runs\n")

    all_results = []

    for bs in BATCH_SIZES:
        print(f"  Batch size {bs}...")

        # PyTorch
        pt_lat = benchmark_pytorch(pt_model, bs, NUM_FEATURES)
        pt_result = {
            "batch_size": bs,
            "runtime": "PyTorch",
            "mean_ms": float(np.mean(pt_lat)),
            "median_ms": float(np.median(pt_lat)),
            "p95_ms": float(np.percentile(pt_lat, 95)),
            "p99_ms": float(np.percentile(pt_lat, 99)),
        }
        all_results.append(pt_result)

        # ONNX
        ort_lat = benchmark_onnx(ort_session, bs, NUM_FEATURES)
        ort_result = {
            "batch_size": bs,
            "runtime": "ONNX",
            "mean_ms": float(np.mean(ort_lat)),
            "median_ms": float(np.median(ort_lat)),
            "p95_ms": float(np.percentile(ort_lat, 95)),
            "p99_ms": float(np.percentile(ort_lat, 99)),
        }
        all_results.append(ort_result)

        speedup = pt_result["mean_ms"] / ort_result["mean_ms"] if ort_result["mean_ms"] > 0 else 0
        print(f"    PyTorch: {pt_result['mean_ms']:.3f}ms  ONNX: {ort_result['mean_ms']:.3f}ms  Speedup: {speedup:.2f}x")

    # Print table
    table = format_table(all_results)
    print(f"\n{'=' * 80}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 80}\n")
    print(table)

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    for bs in BATCH_SIZES:
        pt = next(r for r in all_results if r["batch_size"] == bs and r["runtime"] == "PyTorch")
        ox = next(r for r in all_results if r["batch_size"] == bs and r["runtime"] == "ONNX")
        speedup = pt["mean_ms"] / ox["mean_ms"] if ox["mean_ms"] > 0 else 0
        print(f"  Batch {bs:>4}: ONNX is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than PyTorch")

    # Save BENCHMARK.md
    onnx_size_kb = os.path.getsize(args.onnx_path) / 1024
    pt_params = sum(p.numel() for p in pt_model.parameters())

    md_lines = [
        "# Inference Benchmark: PyTorch vs ONNX Runtime",
        "",
        "## Model",
        "",
        f"- **Architecture:** DLRM (num_features={NUM_FEATURES}, embeddings={EMBEDDING_SIZES}, mlp={MLP_LAYERS})",
        f"- **Parameters:** {pt_params:,}",
        f"- **ONNX file size:** {onnx_size_kb:.1f} KB",
        "",
        "## Setup",
        "",
        f"- **Warmup runs:** {WARMUP_RUNS}",
        f"- **Benchmark runs:** {BENCHMARK_RUNS}",
        f"- **Batch sizes:** {BATCH_SIZES}",
        "- **Device:** CPU",
        "",
        "## Results",
        "",
        table,
        "",
        "## Speedup Summary",
        "",
        "| Batch Size | Speedup (ONNX vs PyTorch) |",
        "|------------|--------------------------|",
    ]

    for bs in BATCH_SIZES:
        pt = next(r for r in all_results if r["batch_size"] == bs and r["runtime"] == "PyTorch")
        ox = next(r for r in all_results if r["batch_size"] == bs and r["runtime"] == "ONNX")
        speedup = pt["mean_ms"] / ox["mean_ms"] if ox["mean_ms"] > 0 else 0
        md_lines.append(f"| {bs} | {speedup:.2f}x |")

    md_content = "\n".join(md_lines) + "\n"

    with open(args.output_path, "w") as f:
        f.write(md_content)
    print(f"\nBenchmark results saved to {args.output_path}")


if __name__ == "__main__":
    main()
