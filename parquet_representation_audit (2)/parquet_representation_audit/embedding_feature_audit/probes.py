from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


def categorical_probes(z: np.ndarray, y: np.ndarray, random_state: int = 42, test_size: float = 0.25) -> pd.DataFrame:
    y = np.asarray(y)
    valid = ~pd.isna(y)
    z, y = z[valid], y[valid]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train, test = next(splitter.split(z, y))
    models = {
        "linear_logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "knn": KNeighborsClassifier(n_neighbors=min(15, max(1, len(train) - 1)), weights="distance", n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1),
    }
    rows = []
    classes = np.unique(y)
    for name, model in models.items():
        m = clone(model).fit(z[train], y[train])
        pred = m.predict(z[test])
        row = {
            "model": name,
            "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
            "macro_f1": float(f1_score(y[test], pred, average="macro", zero_division=0)),
        }
        if hasattr(m, "predict_proba") and len(classes) > 1:
            try:
                proba = m.predict_proba(z[test])
                if len(classes) == 2:
                    row["roc_auc"] = float(roc_auc_score(y[test], proba[:, 1]))
                else:
                    y_bin = label_binarize(y[test], classes=classes)
                    row["roc_auc"] = float(roc_auc_score(y_bin, proba, average="macro", multi_class="ovr"))
            except Exception:
                row["roc_auc"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def continuous_probes(z: np.ndarray, y: np.ndarray, random_state: int = 42, test_size: float = 0.25) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y)
    z, y = z[valid], y[valid]
    splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train, test = next(splitter.split(z))
    models = {
        "linear_ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "knn": KNeighborsRegressor(n_neighbors=min(15, max(1, len(train) - 1)), weights="distance", n_jobs=-1),
        "random_forest": RandomForestRegressor(n_estimators=250, random_state=random_state, n_jobs=-1),
    }
    rows = []
    for name, model in models.items():
        m = clone(model).fit(z[train], y[train])
        pred = m.predict(z[test])
        rows.append({
            "model": name,
            "r2": float(r2_score(y[test], pred)),
            "mae": float(mean_absolute_error(y[test], pred)),
        })
    return pd.DataFrame(rows)
