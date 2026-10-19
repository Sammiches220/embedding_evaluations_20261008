# Worked response: disjoint_clusters

This is an authored reference response to [prep_prompt.md](prep_prompt.md). The JSON below is identical to the ready schema and shared configuration used for the included audit. No external LLM was invoked to produce this walkthrough.

## dataset_schema.json

```json
{
  "schema_version": 1,
  "status": "ready",
  "dataset": {
    "path": "data.parquet",
    "identity_columns": [
      "item_id"
    ],
    "unmapped_columns": "error"
  },
  "columns": {
    "item_id": {
      "role": "id",
      "id_type": "integer"
    },
    "measure_x": {
      "role": "feature",
      "kind": "numeric",
      "units": "standardized coordinate"
    },
    "measure_y": {
      "role": "feature",
      "kind": "numeric",
      "units": "standardized coordinate"
    },
    "event_count": {
      "role": "feature",
      "kind": "count"
    },
    "cluster_4": {
      "role": "feature",
      "kind": "cluster_label",
      "id_type": "integer",
      "description": "Exactly one of four cluster IDs per item."
    },
    "cluster_8": {
      "role": "feature",
      "kind": "cluster_label",
      "id_type": "integer",
      "description": "A second disjoint clustering, also one ID per item."
    }
  },
  "embeddings": [
    {
      "name": "signal_embedding",
      "matrix_path": "embedding_a.npy",
      "ids_path": "embedding_a_ids.npy",
      "expected_dimension": 16,
      "main_normalization": "l2",
      "missing_ids": "error"
    },
    {
      "name": "noisier_embedding",
      "matrix_path": "embedding_b.npy",
      "ids_path": "embedding_b_ids.npy",
      "expected_dimension": 16,
      "main_normalization": "l2",
      "missing_ids": "error"
    }
  ],
  "prediction_pipelines": {}
}
```

## audit_config.json

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

## Mapping decisions

The source table and vector files are unchanged. Each source scalar keeps its original interpretation; each clustering stays one disjoint label column. IDs remain exact typed IDs. Both embedding entries point to separate matrices and separate matching ID arrays.

No unresolved meanings remain for this synthetic example because the user supplied the column meanings above. For real data, unresolved meanings require an explicit decision rather than invented semantics.

Main L2/cosine geometry omits length by design. The normalization experiment also evaluates raw coordinates and magnitude; topology examines both raw Euclidean and unit-chordal geometry. Those are different questions, not interchangeable views of all information.

## Execute the prepared files

To use the two JSON blocks verbatim, save both beside `data.parquet` and run from this example folder:

```bash
parquet-audit run --schema dataset_schema.json --config audit_config.json --out audit_output_from_prompt
```

The project also stores the same configuration once as `../audit_config.json`; the existing [report](audit_output/report.html) and audit files were generated using that shared file and this example's [schema](dataset_schema.json). No reshaped label tables or intermediate prepared dataset are required.
