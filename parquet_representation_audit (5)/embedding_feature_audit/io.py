from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        if len(archive.files) != 1:
            raise ValueError(f"Expected one array in {path}, found {archive.files}")
        return archive[archive.files[0]]
    if path.suffix in {".csv", ".tsv"}:
        sep = "\t" if path.suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep).to_numpy()
    raise ValueError(f"Unsupported array format: {path}")


def load_labels(path: str | Path, column: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if path.suffix in {".csv", ".tsv", ".parquet"}:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
        if column is None:
            if len(df.columns) != 1:
                raise ValueError("Specify --label-column when label table has multiple columns")
            column = str(df.columns[0])
        return df[column].to_numpy()
    raise ValueError(f"Unsupported label format: {path}")


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)
