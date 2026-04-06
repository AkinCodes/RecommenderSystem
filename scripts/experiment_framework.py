"""A/B Experiment Framework for recommender model comparison.

Provides ExperimentConfig, ExperimentResult dataclasses and an
ExperimentRunner that trains two models, collects per-user metrics,
and performs statistical analysis (t-test, confidence interval,
Cohen's d, power analysis).
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from scipy import stats

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""

    name: str
    description: str = ""

    # Model A (control) settings
    model_a_name: str = "baseline"
    model_a_embedding_sizes: list[int] = field(default_factory=lambda: [943, 1682])
    model_a_mlp_layers: list[int] = field(default_factory=lambda: [64, 32, 16])
    model_a_lr: float = 0.001
    model_a_epochs: int = 10

    # Model B (treatment) settings
    model_b_name: str = "dlrm"
    model_b_embedding_sizes: list[int] = field(default_factory=lambda: [943, 1682])
    model_b_mlp_layers: list[int] = field(default_factory=lambda: [128, 64, 32])
    model_b_lr: float = 0.001
    model_b_epochs: int = 20

    # Shared settings
    num_features: int = 2
    batch_size: int = 256
    device: str = "cpu"
    primary_metric: str = "NDCG@10"
    significance_level: float = 0.05
    top_k: int = 10
    random_seed: int = 42


@dataclass
class ExperimentResult:
    """Results from a completed A/B experiment."""

    config: ExperimentConfig
    model_a_metrics: dict[str, float] = field(default_factory=dict)
    model_b_metrics: dict[str, float] = field(default_factory=dict)
    model_a_per_user: dict[str, list[float]] = field(default_factory=dict)
    model_b_per_user: dict[str, list[float]] = field(default_factory=dict)
    model_a_train_time: float = 0.0
    model_b_train_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Statistical results (populated after analysis)
    t_statistic: float = 0.0
    p_value: float = 1.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    cohens_d: float = 0.0
    statistical_power: float = 0.0
    is_significant: bool = False
    winner: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialise result to a dictionary."""
        return {
            "config_name": self.config.name,
            "model_a_name": self.config.model_a_name,
            "model_b_name": self.config.model_b_name,
            "model_a_metrics": self.model_a_metrics,
            "model_b_metrics": self.model_b_metrics,
            "model_a_train_time": self.model_a_train_time,
            "model_b_train_time": self.model_b_train_time,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "confidence_interval": list(self.confidence_interval),
            "cohens_d": self.cohens_d,
            "statistical_power": self.statistical_power,
            "is_significant": self.is_significant,
            "winner": self.winner,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def compute_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two samples."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
        / (na + nb - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((np.mean(b) - np.mean(a)) / pooled_std)


def compute_power(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Approximate statistical power for a two-sample t-test.

    Uses the non-central t-distribution approximation.
    """
    if n < 2 or effect_size == 0:
        return 0.0
    df = 2 * n - 2
    noncentrality = effect_size * np.sqrt(n / 2)
    critical_t = stats.t.ppf(1 - alpha / 2, df)
    # Power = P(reject H0 | H1 true)
    power = 1.0 - stats.nct.cdf(critical_t, df, noncentrality) + stats.nct.cdf(
        -critical_t, df, noncentrality
    )
    return float(np.clip(power, 0.0, 1.0))


def compute_confidence_interval(
    a: np.ndarray, b: np.ndarray, alpha: float = 0.05
) -> tuple[float, float]:
    """Compute confidence interval for the mean difference (B - A)."""
    diff = np.mean(b) - np.mean(a)
    na, nb = len(a), len(b)
    se = np.sqrt(np.var(a, ddof=1) / na + np.var(b, ddof=1) / nb)
    df = na + nb - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    return (float(diff - t_crit * se), float(diff + t_crit * se))


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------
class ExperimentRunner:
    """Runs an A/B experiment comparing two DLRM configurations."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.result = ExperimentResult(config=config)

    # -- internal helpers ---------------------------------------------------
    def _build_model(self, embedding_sizes, mlp_layers):
        from models.dlrm import DLRMModel

        return DLRMModel(
            num_features=self.config.num_features,
            embedding_sizes=embedding_sizes,
            mlp_layers=mlp_layers,
        )

    def _load_data(self):
        """Load and split MovieLens data (reuses logic from train_movielens)."""
        from scripts.train_movielens import load_data, prepare_splits

        raw = load_data()
        splits = prepare_splits(raw)
        return raw, splits

    def _train_model(self, model, train_loader, lr, epochs):
        import torch
        import torch.nn as nn

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        t0 = time.time()
        for _epoch in range(epochs):
            model.train()
            for cont_b, cat_b, target_b in train_loader:
                optimizer.zero_grad()
                preds = model(cont_b, cat_b).squeeze()
                loss = criterion(preds, target_b)
                loss.backward()
                optimizer.step()
        return time.time() - t0

    def _evaluate_per_user(self, model, test_cont, test_cat, test_targets, test_raw, user2idx, item2idx, k):
        """Evaluate and return per-user metric arrays (for statistical tests)."""
        import torch

        model.eval()
        user_test: dict[int, list[tuple[int, float]]] = {}
        for i in range(len(test_raw)):
            uid = test_raw[i, 0]
            uidx = user2idx[uid]
            iidx = int(test_cat[i, 1])
            rating = test_targets[i]
            user_test.setdefault(uidx, []).append((iidx, rating))

        user_cont: dict[int, np.ndarray] = {}
        for i in range(len(test_raw)):
            uidx = int(test_cat[i, 0])
            if uidx not in user_cont:
                user_cont[uidx] = test_cont[i]

        ndcgs, precisions, recalls, hit_rates = [], [], [], []

        with torch.no_grad():
            for uidx, items_ratings in user_test.items():
                if len(items_ratings) < 2:
                    continue
                relevant = {iidx for iidx, r in items_ratings if r >= 0.8}
                if not relevant:
                    continue

                item_indices = [ir[0] for ir in items_ratings]
                n = len(item_indices)
                cont_tensor = torch.tensor(np.tile(user_cont[uidx], (n, 1)), dtype=torch.float32)
                cat_tensor = torch.zeros(n, 2, dtype=torch.long)
                cat_tensor[:, 0] = uidx
                cat_tensor[:, 1] = torch.tensor(item_indices, dtype=torch.long)

                scores = model(cont_tensor, cat_tensor).squeeze().numpy()
                if scores.ndim == 0:
                    scores = np.array([scores.item()])

                ranked_idx = np.argsort(-scores)
                top_k_items = [item_indices[j] for j in ranked_idx[:k]]

                # NDCG
                from scripts.train_movielens import ndcg_at_k
                ranked_rels = [1.0 if item_indices[j] in relevant else 0.0 for j in ranked_idx]
                ndcgs.append(ndcg_at_k(ranked_rels, k))

                # Precision@K
                hits_in_k = sum(1 for it in top_k_items if it in relevant)
                precisions.append(hits_in_k / k)
                recalls.append(hits_in_k / len(relevant) if relevant else 0.0)
                hit_rates.append(1.0 if hits_in_k > 0 else 0.0)

        return {
            "NDCG@10": ndcgs,
            "Precision@10": precisions,
            "Recall@10": recalls,
            "HitRate@10": hit_rates,
        }

    def _run_stats(self):
        """Run statistical tests on the primary metric."""
        metric = self.config.primary_metric
        a = np.array(self.result.model_a_per_user.get(metric, []))
        b = np.array(self.result.model_b_per_user.get(metric, []))

        if len(a) < 2 or len(b) < 2:
            return

        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        self.result.t_statistic = float(t_stat)
        self.result.p_value = float(p_val)
        self.result.confidence_interval = compute_confidence_interval(
            a, b, self.config.significance_level
        )
        self.result.cohens_d = compute_cohens_d(a, b)
        self.result.statistical_power = compute_power(
            abs(self.result.cohens_d), min(len(a), len(b)), self.config.significance_level
        )
        self.result.is_significant = p_val < self.config.significance_level
        if self.result.is_significant:
            self.result.winner = (
                self.config.model_b_name if np.mean(b) > np.mean(a) else self.config.model_a_name
            )

    # -- public API ---------------------------------------------------------
    def run(self) -> ExperimentResult:
        """Run a standard A/B experiment: train both models, evaluate, compare."""
        import torch
        from torch.utils.data import DataLoader
        from scripts.train_movielens import MovieLensDataset

        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        print(f"[Experiment] {self.config.name}")
        print(f"  {self.config.model_a_name} vs {self.config.model_b_name}")

        # Load data
        raw, (train_cont, train_cat, train_targets,
               test_cont, test_cat, test_targets,
               user2idx, item2idx, test_raw) = self._load_data()

        train_ds = MovieLensDataset(train_cont, train_cat, train_targets)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        # Train Model A
        print(f"\n  Training {self.config.model_a_name}...")
        model_a = self._build_model(self.config.model_a_embedding_sizes, self.config.model_a_mlp_layers)
        self.result.model_a_train_time = self._train_model(
            model_a, train_loader, self.config.model_a_lr, self.config.model_a_epochs
        )
        print(f"    Done in {self.result.model_a_train_time:.1f}s")

        # Train Model B
        print(f"  Training {self.config.model_b_name}...")
        model_b = self._build_model(self.config.model_b_embedding_sizes, self.config.model_b_mlp_layers)
        self.result.model_b_train_time = self._train_model(
            model_b, train_loader, self.config.model_b_lr, self.config.model_b_epochs
        )
        print(f"    Done in {self.result.model_b_train_time:.1f}s")

        # Evaluate
        print("  Evaluating...")
        per_user_a = self._evaluate_per_user(
            model_a, test_cont, test_cat, test_targets, test_raw, user2idx, item2idx, self.config.top_k
        )
        per_user_b = self._evaluate_per_user(
            model_b, test_cont, test_cat, test_targets, test_raw, user2idx, item2idx, self.config.top_k
        )

        self.result.model_a_per_user = per_user_a
        self.result.model_b_per_user = per_user_b
        self.result.model_a_metrics = {k: float(np.mean(v)) for k, v in per_user_a.items()}
        self.result.model_b_metrics = {k: float(np.mean(v)) for k, v in per_user_b.items()}

        # Stats
        self._run_stats()

        print("  Done.\n")
        return self.result

    def run_interleaved(self) -> ExperimentResult:
        """Run an interleaved experiment.

        Both models score the same users; we randomly pick which model's
        ranking is shown per user and collect metrics for each group.
        This controls for temporal / population shift.
        """
        import torch
        from torch.utils.data import DataLoader
        from scripts.train_movielens import MovieLensDataset

        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        print(f"[Interleaved Experiment] {self.config.name}")

        raw, (train_cont, train_cat, train_targets,
               test_cont, test_cat, test_targets,
               user2idx, item2idx, test_raw) = self._load_data()

        train_ds = MovieLensDataset(train_cont, train_cat, train_targets)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        # Train both
        model_a = self._build_model(self.config.model_a_embedding_sizes, self.config.model_a_mlp_layers)
        self.result.model_a_train_time = self._train_model(
            model_a, train_loader, self.config.model_a_lr, self.config.model_a_epochs
        )
        model_b = self._build_model(self.config.model_b_embedding_sizes, self.config.model_b_mlp_layers)
        self.result.model_b_train_time = self._train_model(
            model_b, train_loader, self.config.model_b_lr, self.config.model_b_epochs
        )

        # Evaluate both on the same users, randomly assign
        per_user_a = self._evaluate_per_user(
            model_a, test_cont, test_cat, test_targets, test_raw, user2idx, item2idx, self.config.top_k
        )
        per_user_b = self._evaluate_per_user(
            model_b, test_cont, test_cat, test_targets, test_raw, user2idx, item2idx, self.config.top_k
        )

        # Interleave: randomly assign each user index to A or B
        n_users = min(len(per_user_a.get("NDCG@10", [])), len(per_user_b.get("NDCG@10", [])))
        assignments = np.random.choice(["A", "B"], size=n_users)

        interleaved_a: dict[str, list[float]] = {k: [] for k in per_user_a}
        interleaved_b: dict[str, list[float]] = {k: [] for k in per_user_b}

        for i in range(n_users):
            for metric_name in per_user_a:
                if assignments[i] == "A":
                    interleaved_a[metric_name].append(per_user_a[metric_name][i])
                else:
                    interleaved_b[metric_name].append(per_user_b[metric_name][i])

        self.result.model_a_per_user = interleaved_a
        self.result.model_b_per_user = interleaved_b
        self.result.model_a_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in interleaved_a.items()}
        self.result.model_b_metrics = {k: float(np.mean(v)) if v else 0.0 for k, v in interleaved_b.items()}

        self._run_stats()
        print("  Done.\n")
        return self.result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def to_report(result: ExperimentResult) -> str:
    """Generate a markdown report from an ExperimentResult."""
    cfg = result.config
    lines = [
        f"# Experiment Report: {cfg.name}",
        "",
        f"**Date:** {result.timestamp}",
        f"**Primary metric:** {cfg.primary_metric}",
        f"**Significance level:** {cfg.significance_level}",
        "",
        "## Models",
        "",
        f"| Property | {cfg.model_a_name} (Control) | {cfg.model_b_name} (Treatment) |",
        "|----------|-----|-----|",
        f"| MLP layers | {cfg.model_a_mlp_layers} | {cfg.model_b_mlp_layers} |",
        f"| Learning rate | {cfg.model_a_lr} | {cfg.model_b_lr} |",
        f"| Epochs | {cfg.model_a_epochs} | {cfg.model_b_epochs} |",
        f"| Train time | {result.model_a_train_time:.1f}s | {result.model_b_train_time:.1f}s |",
        "",
        "## Metrics",
        "",
        f"| Metric | {cfg.model_a_name} | {cfg.model_b_name} | Delta |",
        "|--------|-----|-----|-------|",
    ]

    all_metrics = set(list(result.model_a_metrics.keys()) + list(result.model_b_metrics.keys()))
    for m in sorted(all_metrics):
        a_val = result.model_a_metrics.get(m, 0.0)
        b_val = result.model_b_metrics.get(m, 0.0)
        delta = b_val - a_val
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {m} | {a_val:.4f} | {b_val:.4f} | {sign}{delta:.4f} |")

    lines += [
        "",
        "## Statistical Analysis",
        "",
        f"| Statistic | Value |",
        "|-----------|-------|",
        f"| t-statistic | {result.t_statistic:.4f} |",
        f"| p-value | {result.p_value:.6f} |",
        f"| 95% CI (B-A) | [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}] |",
        f"| Cohen's d | {result.cohens_d:.4f} |",
        f"| Statistical power | {result.statistical_power:.4f} |",
        f"| Significant | {'Yes' if result.is_significant else 'No'} |",
        f"| Winner | {result.winner or 'No clear winner'} |",
        "",
    ]

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    if result.is_significant:
        d = abs(result.cohens_d)
        effect = "small" if d < 0.5 else ("medium" if d < 0.8 else "large")
        lines.append(
            f"The difference is **statistically significant** (p={result.p_value:.6f}). "
            f"The effect size is **{effect}** (Cohen's d={result.cohens_d:.4f}). "
            f"**{result.winner}** is recommended for deployment."
        )
    else:
        lines.append(
            f"The difference is **not statistically significant** (p={result.p_value:.6f}). "
            f"There is insufficient evidence to prefer one model over the other."
        )

    if result.statistical_power < 0.8:
        lines.append(
            f"\n> **Warning:** Statistical power is low ({result.statistical_power:.2f}). "
            f"Consider increasing the sample size for more reliable conclusions."
        )

    return "\n".join(lines)
