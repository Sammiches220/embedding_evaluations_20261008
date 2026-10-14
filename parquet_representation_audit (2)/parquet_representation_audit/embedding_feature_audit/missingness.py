from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from .prediction import prediction_accuracy, distribution_distortion, geometric_distortion


def _mask_indices(y: np.ndarray, frac: float, rng: np.random.Generator, mode: str) -> np.ndarray:
    n = len(y)
    m = max(1, int(round(frac * n)))
    if mode == "random":
        return rng.choice(n, size=m, replace=False)
    if mode == "rare-first":
        vals, counts = np.unique(y, return_counts=True)
        freq = dict(zip(vals, counts))
        order = np.argsort([freq[v] for v in y])
        pool = order[: max(m, min(n, m * 3))]
        return rng.choice(pool, size=m, replace=False)
    raise ValueError("mode must be 'random' or 'rare-first'")


def run_missingness_experiment(
    z: np.ndarray,
    y: np.ndarray,
    fractions: list[float],
    ks: list[int],
    mode: str = "random",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    summary_rows = []
    geometry_rows = []
    for frac in fractions:
        hidden = _mask_indices(y, frac, rng, mode)
        keep = np.ones(len(y), dtype=bool)
        keep[hidden] = False
        k = min(15, max(1, keep.sum() - 1))
        model = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
        model.fit(z[keep], y[keep])
        predicted_hidden = model.predict(z[hidden])
        reconstructed = np.asarray(y).copy()
        reconstructed[hidden] = predicted_hidden
        acc = prediction_accuracy(y[hidden], predicted_hidden)
        dist = distribution_distortion(y, reconstructed)
        summary_rows.append({"missing_fraction": frac, "mode": mode, **acc, **dist})
        geom = geometric_distortion(z, y, reconstructed, ks)
        geom["missing_fraction"] = frac
        geom["mode"] = mode
        geometry_rows.append(geom)
    return pd.DataFrame(summary_rows), pd.concat(geometry_rows, ignore_index=True)
