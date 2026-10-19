from __future__ import annotations

import numpy as np
import pandas as pd

from .neighborhood import categorical_purity_lift, neighbor_indices


def permutation_null(
    z: np.ndarray,
    y: np.ndarray,
    ks: list[int],
    n_permutations: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    valid = ~pd.isna(y)
    z, y = z[valid], y[valid]
    ks = [k for k in ks if 0 < k < len(y)]
    idx = neighbor_indices(z, max(ks))
    rows = []
    for p in range(n_permutations):
        yp = rng.permutation(y)
        summary, _ = categorical_purity_lift(z, yp, ks, neighbors=idx)
        summary["permutation"] = p
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def attach_null_zscores(observed: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in ["micro_purity", "macro_purity", "micro_lift", "macro_lift"] if c in observed.columns]
    out = observed.copy()
    for metric in metrics:
        stats = nulls.groupby("k")[metric].agg(["mean", "std"]).reset_index()
        stats = stats.rename(columns={"mean": f"null_{metric}_mean", "std": f"null_{metric}_std"})
        out = out.merge(stats, on="k", how="left")
        denom = out[f"null_{metric}_std"].replace(0, np.nan)
        out[f"{metric}_z"] = (out[metric] - out[f"null_{metric}_mean"]) / denom
    return out
