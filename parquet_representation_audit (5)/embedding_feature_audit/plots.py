from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_purity_lift(summary: pd.DataFrame, out_dir: str | Path, prefix: str = "observed") -> None:
    out = Path(out_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["k"], summary["micro_purity"], marker="o", label="Micro purity")
    ax.plot(summary["k"], summary["macro_purity"], marker="o", label="Macro purity")
    ax.plot(summary["k"], summary["chance_purity"], linestyle="--", label="Chance purity")
    ax.set_xscale("log")
    ax.set_xlabel("Neighborhood size k")
    ax.set_ylabel("Purity")
    ax.set_title("Neighborhood purity across scale")
    ax.legend()
    _save(fig, out / f"{prefix}_purity_curve.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["k"], summary["micro_lift"], marker="o", label="Micro lift")
    ax.plot(summary["k"], summary["macro_lift"], marker="o", label="Macro lift")
    ax.axhline(1.0, linestyle="--", label="No enrichment")
    ax.set_xscale("log")
    ax.set_xlabel("Neighborhood size k")
    ax.set_ylabel("Lift")
    ax.set_title("Neighborhood lift across scale")
    ax.legend()
    _save(fig, out / f"{prefix}_lift_curve.png")


def plot_per_label_lift(per_label: pd.DataFrame, out_dir: str | Path, top_n: int = 20) -> None:
    out = Path(out_dir)
    largest_k = per_label["k"].max()
    d = per_label[per_label["k"] == largest_k].nlargest(top_n, "count").sort_values("lift")
    fig, ax = plt.subplots(figsize=(9, max(5, len(d) * 0.3)))
    ax.barh(d["label"].astype(str), d["lift"])
    ax.axvline(1.0, linestyle="--")
    ax.set_xlabel("Lift")
    ax.set_title(f"Per-label lift at k={largest_k} (largest {top_n} labels)")
    _save(fig, out / "per_label_lift.png")


def plot_probe_results(df: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    metrics = [c for c in ["balanced_accuracy", "macro_f1", "roc_auc", "r2"] if c in df.columns]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(df["model"], df[metric])
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Probe performance: {metric.replace('_', ' ')}")
        ax.tick_params(axis="x", rotation=20)
        _save(fig, out / f"probe_{metric}.png")


def plot_null_comparison(observed_with_null: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(observed_with_null["k"], observed_with_null["micro_lift"], marker="o", label="Observed")
    ax.plot(observed_with_null["k"], observed_with_null["null_micro_lift_mean"], marker="o", label="Permutation mean")
    ax.set_xscale("log")
    ax.set_xlabel("Neighborhood size k")
    ax.set_ylabel("Micro lift")
    ax.set_title("Observed lift vs permutation null")
    ax.legend()
    _save(fig, out / "lift_vs_null.png")


def plot_distortion(df: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["k"], df["micro_lift_delta"], marker="o", label="Micro lift delta")
    ax.plot(df["k"], df["macro_lift_delta"], marker="o", label="Macro lift delta")
    ax.axhline(0.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Neighborhood size k")
    ax.set_ylabel("Predicted - true")
    ax.set_title("Prediction-induced lift distortion")
    ax.legend()
    _save(fig, out / "prediction_lift_distortion.png")


def plot_missingness(summary: pd.DataFrame, geometry: pd.DataFrame, out_dir: str | Path) -> None:
    out = Path(out_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["missing_fraction"], summary["macro_f1"], marker="o", label="Macro F1")
    ax.plot(summary["missing_fraction"], summary["balanced_accuracy"], marker="o", label="Balanced accuracy")
    ax.set_xlabel("Missing fraction")
    ax.set_ylabel("Prediction performance")
    ax.set_title("Reconstruction performance under missingness")
    ax.legend()
    _save(fig, out / "missingness_prediction_performance.png")

    g = geometry.groupby("missing_fraction", as_index=False)[["micro_lift_delta", "macro_lift_delta"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(g["missing_fraction"], g["micro_lift_delta"], marker="o", label="Micro lift delta")
    ax.plot(g["missing_fraction"], g["macro_lift_delta"], marker="o", label="Macro lift delta")
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Missing fraction")
    ax.set_ylabel("Mean lift distortion")
    ax.set_title("Geometric distortion as labels are reconstructed")
    ax.legend()
    _save(fig, out / "missingness_lift_distortion.png")
