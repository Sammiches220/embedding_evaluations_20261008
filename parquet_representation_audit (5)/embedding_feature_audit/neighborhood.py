from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def neighbor_indices(z: np.ndarray, max_k: int, metric: str = "euclidean") -> np.ndarray:
    n = len(z)
    if n < 2:
        raise ValueError("Need at least two embedding rows")
    k = min(max_k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
    nn.fit(z)
    # X=None already excludes each query point. Do not discard the closest neighbor.
    return nn.kneighbors(n_neighbors=k, return_distance=False)


def categorical_purity_lift(
    z: np.ndarray,
    y: np.ndarray,
    ks: list[int],
    metric: str = "euclidean",
    neighbors: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = np.asarray(y)
    valid = ~pd.isna(y)
    z = z[valid]
    y = y[valid]
    if len(y) < 2:
        raise ValueError("Not enough labeled items")
    ks = sorted({int(k) for k in ks if int(k) > 0 and int(k) < len(y)})
    if not ks:
        raise ValueError("No valid k values")
    idx = neighbor_indices(z, max(ks), metric) if neighbors is None else neighbors
    values, counts = np.unique(y, return_counts=True)
    prevalence = {v: c / len(y) for v, c in zip(values, counts)}
    rows = []
    per_label_rows = []
    for k in ks:
        neigh = idx[:, :k]
        same = y[neigh] == y[:, None]
        item_purity = same.mean(axis=1)
        item_baseline = np.array([prevalence[v] for v in y], dtype=float)
        item_lift = np.divide(item_purity, item_baseline, out=np.zeros_like(item_purity), where=item_baseline > 0)
        rows.append({
            "k": k,
            "micro_purity": float(item_purity.mean()),
            "macro_purity": float(np.mean([item_purity[y == v].mean() for v in values])),
            "micro_lift": float(item_lift.mean()),
            "macro_lift": float(np.mean([item_lift[y == v].mean() for v in values])),
            "chance_purity": float(np.mean(item_baseline)),
        })
        for v in values:
            mask = y == v
            purity = float(item_purity[mask].mean())
            base = float(prevalence[v])
            per_label_rows.append({
                "k": k,
                "label": v,
                "count": int(mask.sum()),
                "prevalence": base,
                "purity": purity,
                "lift": purity / base if base > 0 else np.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(per_label_rows)


def continuous_neighborhood_stats(
    z: np.ndarray,
    y: np.ndarray,
    ks: list[int],
    metric: str = "euclidean",
) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    z = z[valid]
    y = y[valid]
    ks = sorted({int(k) for k in ks if 0 < int(k) < len(y)})
    idx = neighbor_indices(z, max(ks), metric)
    global_var = float(np.var(y))
    rows = []
    for k in ks:
        nvals = y[idx[:, :k]]
        abs_diff = np.abs(nvals - y[:, None]).mean(axis=1)
        local_var = np.var(nvals, axis=1)
        rows.append({
            "k": k,
            "mean_neighbor_abs_difference": float(abs_diff.mean()),
            "mean_local_variance": float(local_var.mean()),
            "variance_ratio_to_global": float(local_var.mean() / global_var) if global_var > 0 else np.nan,
        })
    return pd.DataFrame(rows)
