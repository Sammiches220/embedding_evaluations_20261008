"""Create three small, deterministic Parquet-to-embedding audit examples.

Run from any directory with:
    python examples/generate_examples.py

Each embedding matrix and its ID vector use the exact Parquet row order.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
N = 360
D = 16


def save_case(name: str, frame: pd.DataFrame, embedding: np.ndarray, id_column: str,
              id_type: str, features: dict) -> None:
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / "data.parquet", index=False)
    matrix = np.asarray(embedding, dtype=np.float32)
    row_ids = frame[id_column].to_numpy(dtype=str) if id_type == "string" else frame[id_column].to_numpy(dtype=np.int64)
    # The second representation has lower signal-to-noise, but preserves exact row order.
    rng = np.random.default_rng(91000 + sum(map(ord, name)))
    second = (0.72 * matrix + rng.normal(0, 0.65, matrix.shape)).astype(np.float32)
    for suffix, values in (("a", matrix), ("b", second)):
        np.save(folder / f"embedding_{suffix}.npy", values, allow_pickle=False)
        np.save(folder / f"embedding_{suffix}_ids.npy", row_ids, allow_pickle=False)
        assert values.shape == (len(frame), D)
        assert np.array_equal(np.load(folder / f"embedding_{suffix}_ids.npy", allow_pickle=False), row_ids)

    columns = {id_column: {"role": "id", "id_type": id_type}, **features}
    schema = {
        "schema_version": 1,
        "status": "ready",
        "dataset": {"path": "data.parquet", "identity_columns": [id_column], "unmapped_columns": "error"},
        "columns": columns,
        "embeddings": [{
            "name": "signal_embedding",
            "matrix_path": "embedding_a.npy",
            "ids_path": "embedding_a_ids.npy",
            "expected_dimension": D,
            "main_normalization": "l2",
            "missing_ids": "error",
        }, {
            "name": "noisier_embedding",
            "matrix_path": "embedding_b.npy",
            "ids_path": "embedding_b_ids.npy",
            "expected_dimension": D,
            "main_normalization": "l2",
            "missing_ids": "error",
        }],
        "prediction_pipelines": {},
    }
    (folder / "dataset_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    check_alignment(folder, frame, id_column, id_type)


def project(values: np.ndarray, rng: np.random.Generator, noise: float = 0.10) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    weights = rng.normal(0, 1 / np.sqrt(values.shape[1]), size=(values.shape[1], D)).astype(np.float32)
    return values @ weights + rng.normal(0, noise, size=(len(values), D)).astype(np.float32)


def check_alignment(folder: Path, frame: pd.DataFrame, id_column: str, id_type: str) -> None:
    parquet_ids = pd.read_parquet(folder / "data.parquet", columns=[id_column])[id_column].to_numpy()
    for suffix in ("a", "b"):
        matrix = np.load(folder / f"embedding_{suffix}.npy", mmap_mode="r", allow_pickle=False)
        embedding_ids = np.load(folder / f"embedding_{suffix}_ids.npy", mmap_mode="r", allow_pickle=False)
        if not np.array_equal(parquet_ids, embedding_ids) or matrix.shape[0] != len(parquet_ids):
            raise ValueError(f"{folder.name}: embedding {suffix} row order does not match Parquet item IDs")
    print(f"{folder.name}: {len(parquet_ids)} rows, two {matrix.shape[1]}-dimensional embeddings, exact {id_type} ID order verified")


def make_numeric_categorical() -> None:
    rng = np.random.default_rng(20261017)
    ids = np.array([f"customer-{i:04d}" for i in range(N)])
    age = rng.integers(18, 76, N)
    income = np.clip(rng.normal(62000 + (age - 40) * 450, 18000, N), 18000, 160000).round(2)
    tenure = rng.integers(1, 1600, N)
    plan = rng.choice(np.array(["basic", "plus", "premium"]), N, p=[0.48, 0.34, 0.18])
    plan_effect = pd.Series(plan).map({"basic": -0.5, "plus": 0.25, "premium": 0.8}).to_numpy()
    spend = np.clip(18 + plan_effect * 20 + income / 7000 + rng.normal(0, 7, N), 5, 150).round(2)
    churn_probability = 1 / (1 + np.exp(-(-1.1 + 0.0010 * (age - 42) - 0.0007 * (tenure - 500) - 0.22 * plan_effect)))
    churn = (rng.random(N) < churn_probability).astype(bool)
    frame = pd.DataFrame({
        "customer_id": ids,
        "age_years": age.astype(np.int64),
        "annual_income_usd": income,
        "tenure_days": tenure.astype(np.int64),
        "plan_type": plan,
        "monthly_spend_usd": spend,
        "churned": churn,
    })
    onehot = pd.get_dummies(frame["plan_type"], prefix="plan", dtype=np.float32)
    vectors = np.column_stack([
        (age - age.mean()) / age.std(),
        (income - income.mean()) / income.std(),
        (tenure - tenure.mean()) / tenure.std(),
        (spend - spend.mean()) / spend.std(),
        onehot.to_numpy(),
    ])
    save_case("numeric_categorical", frame, project(vectors, rng, .18), "customer_id", "string", {
        "age_years": {"role": "feature", "kind": "count", "units": "years", "description": "Integer customer age."},
        "annual_income_usd": {"role": "feature", "kind": "numeric", "units": "USD/year", "description": "Continuous income; retain its raw value."},
        "tenure_days": {"role": "feature", "kind": "count", "units": "days"},
        "plan_type": {"role": "feature", "kind": "categorical", "id_type": "string"},
        "monthly_spend_usd": {"role": "feature", "kind": "numeric", "units": "USD/month"},
        "churned": {"role": "feature", "kind": "boolean"},
    })


def make_disjoint_clusters() -> None:
    rng = np.random.default_rng(20261018)
    labels = np.arange(N, dtype=np.int64) % 4
    rng.shuffle(labels)
    centers = np.array([[-2.2, -1.5], [-2.0, 1.7], [2.0, -1.4], [2.2, 1.6]])
    latent = centers[labels] + rng.normal(0, 0.48, size=(N, 2))
    subcluster = (labels * 2 + (latent[:, 1] > centers[labels, 1])).astype(np.int64)
    frame = pd.DataFrame({
        "item_id": np.arange(100000, 100000 + N, dtype=np.int64),
        "measure_x": latent[:, 0].round(5),
        "measure_y": latent[:, 1].round(5),
        "event_count": rng.poisson(np.exp(1.3 + 0.2 * latent[:, 0]), N).astype(np.int64),
        "cluster_4": labels,
        "cluster_8": subcluster,
    })
    source = np.column_stack([latent, frame["event_count"].to_numpy() / 4,
                              np.eye(4, dtype=np.float32)[labels],
                              np.eye(8, dtype=np.float32)[subcluster]])
    save_case("disjoint_clusters", frame, project(source, rng, .14), "item_id", "integer", {
        "measure_x": {"role": "feature", "kind": "numeric", "units": "standardized coordinate"},
        "measure_y": {"role": "feature", "kind": "numeric", "units": "standardized coordinate"},
        "event_count": {"role": "feature", "kind": "count"},
        "cluster_4": {"role": "feature", "kind": "cluster_label", "id_type": "integer", "description": "Exactly one of four cluster IDs per item."},
        "cluster_8": {"role": "feature", "kind": "cluster_label", "id_type": "integer", "description": "A second disjoint clustering, also one ID per item."},
    })


def make_multilabel() -> None:
    rng = np.random.default_rng(20261019)
    topic = np.arange(N) % 5
    rng.shuffle(topic)
    centers = np.array([[-2, 0, 0], [-1, 2, 0], [1, 1.4, -1], [2, -1, 1], [0, -2, 2]], dtype=float)
    latent = centers[topic] + rng.normal(0, 0.62, size=(N, 3))
    vocab = ["outdoor", "indoor", "blue", "warm", "compact", "premium", "quiet", "sport", "travel", "daily"]
    topic_tags = {
        0: ["outdoor", "sport", "travel"], 1: ["indoor", "quiet", "daily"],
        2: ["blue", "compact", "daily"], 3: ["warm", "premium", "quiet"],
        4: ["travel", "compact", "premium"],
    }
    tags = []
    for t in topic:
        chosen = set(topic_tags[int(t)])
        for tag in vocab:
            if tag not in chosen and rng.random() < 0.045:
                chosen.add(tag)
        tags.append(sorted(chosen))
    frame = pd.DataFrame({
        "product_id": [f"sku-{i:05d}" for i in range(N)],
        "feature_1": latent[:, 0].round(5),
        "feature_2": latent[:, 1].round(5),
        "feature_3": latent[:, 2].round(5),
        "price_usd": np.clip(45 + topic * 12 + rng.normal(0, 10, N), 10, 200).round(2),
        "tags": tags,
    })
    onehot = np.eye(5, dtype=np.float32)[topic]
    vectors = np.column_stack([latent, onehot, frame["price_usd"].to_numpy() / 50])
    save_case("multilabel_tags", frame, project(vectors, rng, .22), "product_id", "string", {
        "feature_1": {"role": "feature", "kind": "numeric"},
        "feature_2": {"role": "feature", "kind": "numeric"},
        "feature_3": {"role": "feature", "kind": "numeric"},
        "price_usd": {"role": "feature", "kind": "numeric", "units": "USD"},
        "tags": {"role": "feature", "kind": "multilabel", "encoding": "list", "description": "A co-occurring set of tags per product.", "empty_is_unknown": False},
    })


if __name__ == "__main__":
    make_numeric_categorical()
    make_disjoint_clusters()
    make_multilabel()
