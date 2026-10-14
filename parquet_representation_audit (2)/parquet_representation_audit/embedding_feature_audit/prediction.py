from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import balanced_accuracy_score, f1_score

from .source_stats import categorical_source_stats
from .neighborhood import categorical_purity_lift


def prediction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = (~pd.isna(y_true)) & (~pd.isna(y_pred))
    yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    return {
        "n_compared": int(mask.sum()),
        "accuracy": float(np.mean(yt == yp)) if len(yt) else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan,
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)) if len(yt) else np.nan,
    }


def distribution_distortion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_labels, true_counts = np.unique(y_true, return_counts=True)
    pred_labels, pred_counts = np.unique(y_pred, return_counts=True)
    labels = np.union1d(true_labels, pred_labels)
    tp = np.array([true_counts[np.where(true_labels == x)[0][0]] if x in true_labels else 0 for x in labels], dtype=float)
    pp = np.array([pred_counts[np.where(pred_labels == x)[0][0]] if x in pred_labels else 0 for x in labels], dtype=float)
    tp /= tp.sum(); pp /= pp.sum()
    js = float(jensenshannon(tp, pp, base=2.0) ** 2)
    true_summary, _ = categorical_source_stats(y_true)
    pred_summary, _ = categorical_source_stats(y_pred)
    return {
        "jensen_shannon_divergence": js,
        "largest_label_share_delta": pred_summary["largest_label_share"] - true_summary["largest_label_share"],
        "normalized_entropy_delta": pred_summary["normalized_entropy"] - true_summary["normalized_entropy"],
        "n_labels_delta": pred_summary["n_labels"] - true_summary["n_labels"],
    }


def geometric_distortion(z: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, ks: list[int]) -> pd.DataFrame:
    a, _ = categorical_purity_lift(z, y_true, ks)
    b, _ = categorical_purity_lift(z, y_pred, ks)
    merged = a.merge(b, on="k", suffixes=("_true", "_pred"))
    for metric in ["micro_purity", "macro_purity", "micro_lift", "macro_lift"]:
        merged[f"{metric}_delta"] = merged[f"{metric}_pred"] - merged[f"{metric}_true"]
        merged[f"{metric}_ratio"] = merged[f"{metric}_pred"] / merged[f"{metric}_true"].replace(0, np.nan)
    return merged
