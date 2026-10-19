# Prepare this dataset for an embedding audit

Create exactly two files for the current `parquet-audit` schema-driven format: `dataset_schema.json` and `audit_config.json`. Use the inspected Parquet file below. Do not invent semantics from column names; flag uncertain columns as `ignore` and list questions that need a human answer. Do not create one-hot or expanded copies of feature columns.

## Table

- Path: `data.parquet`
- Rows: 360
- Physical samples: First at most 12 rows; descriptive, not population estimates

- `item_id` (int64): examples 100000, 100001, 100002; sampled list lengths {}; sampled list element types {}
- `measure_x` (double): examples -1.66148, 1.81881, 1.98014; sampled list lengths {}; sampled list element types {}
- `measure_y` (double): examples 2.44115, -0.73634, -1.3251; sampled list lengths {}; sampled list element types {}
- `event_count` (int64): examples 3, 3, 5; sampled list lengths {}; sampled list element types {}
- `cluster_4` (int64): examples 1, 2, 2; sampled list lengths {}; sampled list element types {}
- `cluster_8` (int64): examples 3, 5, 5; sampled list lengths {}; sampled list element types {}

## Data contract

- One row is one item. Keep related feature columns together in this Parquet table. Every declared feature must have one value per item, with null for unknown/missing.
- Use `numeric` for real-valued engineered measurements, `count` for nonnegative integer counts, `categorical` for one scalar category, and `cluster_label` for one mutually exclusive cluster ID per item. Each separate clustering is its own feature column; do not turn a clustering into one column per cluster.
- Use `multilabel` only when a row has a set/list of tags or labels. Preserve the set semantics and declare its encoding (list, delimited string, or JSON string) explicitly.
- Use `ordinal` only when the categories have a meaningful declared order. Do not convert nominal categories to numbers. Such conversion loses meaning unless a justified order and mapping are explicitly supplied.
- Stable IDs must remain exact strings, integers, or booleans; never pass identifiers through floating point. Embeddings are real numeric vectors of one fixed declared dimension.
- Preserve raw feature information. Do not bin numeric values or discard categories unless the user explicitly asks for that transformation.
- Put every source column in `columns`, assigning an explicit role (`id`, `feature`, `embedding`, `group`, `split`, `stratum`, `prediction`, `probabilities`, or `ignore`). For a first run, use a stable ID, one embedding vector column or external aligned matrix, and only the features the user wants evaluated.
- Keep unknown semantics explicit: `ignore` excludes a column from the audit; it does not prove the full dataset was represented. Do not infer a binning, nominal order or missing sentinel to make a column fit.
- The probe names are `baseline`, `linear`, `knn`, `random_forest`, and `xgboost`. XGBoost requires the optional installation `python -m pip install -e ".[xgboost]"`. Normalization inherits `probes.models` and `probes.trees` unless `normalization.models` / `normalization.trees` explicitly override them.
- Distinguish preserving source values from auditing every distinction. Report vocabulary/probe label caps, sampling, precision conversions, and any set conversion that drops order or multiplicity. Do not claim a summary or a successful probe proves complete information preservation.
- Use current schema version 1 only. Do not add compatibility behavior for older formats.

## Required schema shape

Use this structure, adding every physical column under `columns`. Map the real ID, embedding, and wanted features to their explicit roles; mark genuinely unused columns `ignore`.

```json
{
  "schema_version": 1,
  "status": "ready",
  "dataset": {"path": "data.parquet", "identity_columns": ["<stable ID column>"], "unmapped_columns": "error"},
  "columns": {
    "<stable ID column>": {"role": "id", "id_type": "string"},
    "<embedding column>": {"role": "embedding", "expected_dimension": 768},
    "<feature column>": {"role": "feature", "kind": "numeric", "description": "...", "geometry": true, "probe": true}
  },
  "embeddings": [{"name": "embedding", "column": "<embedding column>", "main_normalization": "l2", "expected_dimension": 768}],
  "prediction_pipelines": {}
}
```

The following is the complete current configuration. Copy it to `audit_config.json` and only change values when a data constraint requires it. Keep `geometry.ks` in the listed order for comparisons.

```json
{
  "mode": "embedding",
  "seed": 42,
  "verbose": true,
  "source": {
    "scope": "full",
    "batch_rows": 1024,
    "max_feature_columns": 100,
    "max_vocabulary": 10000,
    "quantile_sample_size": 10000
  },
  "sample": {
    "strategy": "all",
    "max_rows": 400
  },
  "geometry": {
    "enabled": true,
    "metric": "cosine",
    "ks": [
      5,
      10,
      20,
      50
    ],
    "reference": "full_source",
    "invalid_embeddings": "exclude",
    "distance_cap": 2000,
    "distance_pairs": 50000
  },
  "limits": {
    "max_evaluation_rows": 400,
    "max_exact_pairs": 1000000,
    "max_embedding_bytes": 100000000,
    "max_external_ids": 10000,
    "max_dense_feature_bytes": 100000000,
    "max_work_units": 1000000000,
    "max_source_batch_bytes": 100000000,
    "max_source_batch_cells": 1000000,
    "external_batch_rows": 256
  },
  "nulls": {
    "permutations": 8,
    "modes": [
      "permutation"
    ],
    "correction": "bh",
    "testing_family": "all_features_scales_modes_per_metric"
  },
  "probes": {
    "enabled": true,
    "models": [
      "baseline",
      "linear",
      "knn",
      "xgboost"
    ],
    "max_labels": 20,
    "trees": 20,
    "regularization_grid": [
      1.0
    ]
  },
  "normalization": {
    "enabled": true,
    "features": true,
    "variants": [
      "raw_standardized",
      "unit",
      "unit_standardized",
      "norm_only"
    ],
    "max_rows": 400,
    "regularization_grid": [
      1.0
    ]
  },
  "topology": {
    "enabled": true,
    "features": true,
    "spaces": [
      "unit_chordal",
      "raw_euclidean"
    ],
    "sample_size": 96,
    "repetitions": 2,
    "quantiles": [
      0,
      0.01,
      0.05,
      0.1,
      0.25,
      0.5,
      1
    ],
    "radii": {},
    "permutations": 5,
    "min_component_known": 1,
    "min_retained_share": 0.5,
    "null_mode": "permutation",
    "h1": true,
    "h1_sample_size": 32,
    "max_simplices": 10000,
    "max_distance_bytes": 16000000
  },
  "missingness": {
    "enabled": false,
    "features": true,
    "fractions": [
      0.1,
      0.25
    ],
    "modes": [
      "random"
    ]
  },
  "report": {
    "title": "Example embedding feature audit",
    "overview": {
      "min_known": 10,
      "min_test": 10,
      "min_known_neighbor_share": 0.5,
      "neutral_delta": 0.02,
      "probe_neutral_delta": 0.02,
      "trusted_ks": [
        5,
        10,
        20,
        50
      ],
      "metric_thresholds": {}
    },
    "row_explorer": true,
    "max_rows": 400,
    "max_bytes": 10000000,
    "pca": true
  }
}
```

If embeddings are stored separately from the Parquet table, use an external asset entry instead of declaring an embedding column. For example, `embedding.npy` must be an `N × D` real numeric matrix and `embedding_ids.npy` must contain its `N` item IDs. In this example, line `i` of both arrays corresponds to Parquet row `i`; the ID array lets the audit verify and align that relationship.

```json
{"name": "model_a", "matrix_path": "embedding.npy", "ids_path": "embedding_ids.npy", "expected_dimension": 768, "main_normalization": "l2", "missing_ids": "error"}
```

Return the two complete JSON files, a short list of unresolved semantic questions, and a brief explanation of any conversion that is not lossless. Copy the embedded `audit_config.json` template and keep the ordered `geometry.ks` values unchanged unless the user requests different neighborhood scales. Do not enable optional expensive analyses by default. Sample list lengths are clues only; verify a declared embedding dimension across the selected data during the audit.

## User-supplied meanings and requested settings

- One row is one item. `item_id` is an exact integer ID; never route it through float.
- `measure_x` and `measure_y` are signed numeric coordinates. `event_count` is a nonnegative integer count.
- `cluster_4` assigns exactly one of four mutually exclusive cluster IDs per row; `cluster_8` assigns exactly one of eight IDs. These are TWO separate clusterings of the SAME rows. Their IDs are nominal and have no numeric order. Preserve each as one cluster_label column.
- Cluster ID 0 is an observed cluster, not unknown. No source column should be removed. No numeric binning or one-hot conversion is requested.
- Both external embeddings have 360 rows and 16 float32 coordinates. Use names `signal_embedding` and `noisier_embedding`; assets `embedding_a.npy`, `embedding_b.npy`, `embedding_a_ids.npy`, `embedding_b_ids.npy`.
- For BOTH embeddings, matrix row i and ID array entry i correspond to Parquet row i. Require missing_ids=error; the arrays have exactly the source item IDs in that order.
- Use L2 normalization for the main cosine geometry. Keep the original vectors for the enabled raw/direction/magnitude comparisons.

Write the two JSON files beside data.parquet. Use the complete supplied configuration; these small walkthroughs intentionally enable normalization and topology. Do not alter the Parquet file or embedding arrays.
