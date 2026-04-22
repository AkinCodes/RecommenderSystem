"""CLI entry-point for running A/B experiments.

Usage:
    uv run python scripts/run_experiment.py                          # default config
    uv run python scripts/run_experiment.py --config configs/experiments/dlrm_vs_baseline.yaml
    uv run python scripts/run_experiment.py --interleaved            # interleaved mode
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.experiment_framework import (
    ExperimentConfig,
    ExperimentRunner,
    to_report,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "experiments", "results")


def load_config(path: str) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    exp = data.get("experiment", data)
    model_a = exp.get("model_a", {})
    model_b = exp.get("model_b", {})

    return ExperimentConfig(
        name=exp.get("name", "unnamed"),
        description=exp.get("description", ""),
        model_a_name=model_a.get("name", "baseline"),
        model_a_embedding_sizes=model_a.get("embedding_sizes", [943, 1682]),
        model_a_mlp_layers=model_a.get("mlp_layers", [64, 32, 16]),
        model_a_lr=model_a.get("learning_rate", 0.001),
        model_a_epochs=model_a.get("epochs", 10),
        model_b_name=model_b.get("name", "dlrm"),
        model_b_embedding_sizes=model_b.get("embedding_sizes", [943, 1682]),
        model_b_mlp_layers=model_b.get("mlp_layers", [128, 64, 32]),
        model_b_lr=model_b.get("learning_rate", 0.001),
        model_b_epochs=model_b.get("epochs", 20),
        num_features=exp.get("num_features", 2),
        batch_size=exp.get("batch_size", 256),
        primary_metric=exp.get("primary_metric", "NDCG@10"),
        significance_level=exp.get("significance_level", 0.05),
        top_k=exp.get("top_k", 10),
        random_seed=exp.get("random_seed", 42),
    )


def main():
    parser = argparse.ArgumentParser(description="Run an A/B experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--interleaved",
        action="store_true",
        help="Run interleaved experiment instead of standard A/B",
    )
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = ExperimentConfig(
            name="default_dlrm_vs_baseline",
            description="Default comparison of small baseline vs larger DLRM",
        )

    runner = ExperimentRunner(config)

    if args.interleaved:
        result = runner.run_interleaved()
    else:
        result = runner.run()

    # Print report
    report = to_report(result)
    print(report)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_name = config.name.replace(" ", "_").replace("/", "_")

    json_path = os.path.join(RESULTS_DIR, f"{safe_name}.json")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nJSON results saved to {json_path}")

    md_path = os.path.join(RESULTS_DIR, f"{safe_name}.md")
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Markdown report saved to {md_path}")


if __name__ == "__main__":
    main()
