from __future__ import annotations

import numpy as np
import pandas as pd


def categorical_source_stats(y: np.ndarray) -> tuple[dict, pd.DataFrame]:
    s = pd.Series(y, name="label")
    counts = s.value_counts(dropna=False)
    n = len(s)
    probs = counts / max(n, 1)
    entropy = float(-(probs * np.log(probs.clip(lower=1e-15))).sum())
    max_entropy = float(np.log(len(probs))) if len(probs) > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    summary = {
        "n_items": int(n),
        "n_labels": int(len(counts)),
        "largest_label_share": float(probs.max()) if len(probs) else 0.0,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "missing_share": float(s.isna().mean()),
    }
    table = pd.DataFrame({"count": counts, "share": probs}).reset_index(names="label")
    return summary, table


def continuous_source_stats(y: np.ndarray) -> dict:
    s = pd.Series(y, dtype="float64")
    return {
        "n_items": int(len(s)),
        "missing_share": float(s.isna().mean()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "q05": float(s.quantile(0.05)),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(0.75)),
        "q95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }
