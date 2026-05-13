"""Build script for deployment: download data, train model, generate serving artifacts.

This runs during Docker build or Render build step to produce:
  - trained_model_movielens.pth  (DLRM weights)
  - serving_context.pkl          (user/item mappings for /recommend endpoints)
  - reports/model_comparison.json (benchmark results)

Uses retrain_and_compare.py which does NOT require wandb.
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(cmd, description):
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=PROJECT_ROOT,
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  FAILED: {description}")
        sys.exit(result.returncode)
    print(f"  OK: {description}")


def verify_artifacts():
    required = [
        ("trained_model_movielens.pth", "DLRM model weights"),
        ("serving_context.pkl", "Serving context (user/item mappings)"),
    ]
    print(f"\n{'=' * 60}")
    print("  Verifying artifacts")
    print(f"{'=' * 60}")
    all_ok = True
    for filename, label in required:
        path = os.path.join(PROJECT_ROOT, filename)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  OK  {filename} ({size_mb:.1f} MB) — {label}")
        else:
            print(f"  MISSING  {filename} — {label}")
            all_ok = False

    if not all_ok:
        print("\n  Build failed: missing artifacts.")
        sys.exit(1)

    print("\n  All artifacts present. Ready to serve.")


def main():
    print("=" * 60)
    print("  RecommenderSystem — Build for Deployment")
    print("=" * 60)

    # Step 1: Download MovieLens 100K if not present
    run(
        ["scripts/download_data.py"],
        "Download MovieLens 100K dataset",
    )

    # Step 2: Train DLRM + classical baselines, generate all artifacts
    run(
        ["scripts/retrain_and_compare.py"],
        "Train DLRM + baselines, generate model + serving context",
    )

    # Step 3: Verify everything was created
    verify_artifacts()


if __name__ == "__main__":
    main()
