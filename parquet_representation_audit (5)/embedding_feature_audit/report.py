from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import save_json
from .missingness import run_missingness_experiment
from .neighborhood import categorical_purity_lift, continuous_neighborhood_stats
from .nulls import attach_null_zscores, permutation_null
from .plots import plot_distortion, plot_missingness, plot_null_comparison, plot_per_label_lift, plot_probe_results, plot_purity_lift
from .prediction import distribution_distortion, geometric_distortion, prediction_accuracy
from .probes import categorical_probes, continuous_probes
from .source_stats import categorical_source_stats, continuous_source_stats


def run_categorical_audit(
    z: np.ndarray,
    y: np.ndarray,
    out_dir: str | Path,
    ks: list[int],
    y_pred: np.ndarray | None = None,
    n_permutations: int = 30,
    missingness_fractions: list[float] | None = None,
    random_state: int = 42,
) -> None:
    out = Path(out_dir)
    tables, plots = out / "tables", out / "plots"
    tables.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)

    source, prevalence = categorical_source_stats(y)
    save_json(source, out / "source_summary.json")
    prevalence.to_csv(tables / "label_prevalence.csv", index=False)

    summary, per_label = categorical_purity_lift(z, y, ks)
    summary.to_csv(tables / "purity_lift_by_k.csv", index=False)
    per_label.to_csv(tables / "purity_lift_by_label.csv", index=False)
    plot_purity_lift(summary, plots)
    plot_per_label_lift(per_label, plots)

    probes = categorical_probes(z, y, random_state=random_state)
    probes.to_csv(tables / "probe_results.csv", index=False)
    plot_probe_results(probes, plots)

    nulls = permutation_null(z, y, ks, n_permutations=n_permutations, random_state=random_state)
    nulls.to_csv(tables / "permutation_null.csv", index=False)
    observed_null = attach_null_zscores(summary, nulls)
    observed_null.to_csv(tables / "purity_lift_with_null_zscores.csv", index=False)
    plot_null_comparison(observed_null, plots)

    if y_pred is not None:
        save_json(prediction_accuracy(y, y_pred), out / "prediction_accuracy.json")
        save_json(distribution_distortion(y, y_pred), out / "prediction_distribution_distortion.json")
        geom = geometric_distortion(z, y, y_pred, ks)
        geom.to_csv(tables / "prediction_geometric_distortion.csv", index=False)
        pred_summary, pred_per_label = categorical_purity_lift(z, y_pred, ks)
        pred_summary.to_csv(tables / "predicted_purity_lift_by_k.csv", index=False)
        pred_per_label.to_csv(tables / "predicted_purity_lift_by_label.csv", index=False)
        plot_purity_lift(pred_summary, plots, prefix="predicted")
        plot_distortion(geom, plots)

    if missingness_fractions:
        for mode in ["random", "rare-first"]:
            ms, mg = run_missingness_experiment(z, y, missingness_fractions, ks, mode=mode, random_state=random_state)
            ms.to_csv(tables / f"missingness_{mode}_summary.csv", index=False)
            mg.to_csv(tables / f"missingness_{mode}_geometry.csv", index=False)
            mode_plot_dir = plots / f"missingness_{mode}"
            plot_missingness(ms, mg, mode_plot_dir)


def run_continuous_audit(z: np.ndarray, y: np.ndarray, out_dir: str | Path, ks: list[int], random_state: int = 42) -> None:
    out = Path(out_dir)
    tables, plots = out / "tables", out / "plots"
    tables.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True)
    save_json(continuous_source_stats(y), out / "source_summary.json")
    neigh = continuous_neighborhood_stats(z, y, ks)
    neigh.to_csv(tables / "continuous_neighborhood_stats.csv", index=False)
    probes = continuous_probes(z, y, random_state=random_state)
    probes.to_csv(tables / "probe_results.csv", index=False)
    plot_probe_results(probes, plots)
