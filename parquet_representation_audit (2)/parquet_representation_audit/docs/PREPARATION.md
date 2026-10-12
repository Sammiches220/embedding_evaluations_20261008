# Dataset preparation contract

Use `prompts/prepare_dataset.md` to ask another chat to create the files. The filled `examples/retail/` bundle is executable and illustrates the actual JSON contract. Schemas use JSON Schema Draft 2020-12 and reject unrecognized fields. `schema_version` is 1 in both documents.

## Semantics

`dataset_schema.json` has `status: "validated"` or `"draft"`, `dataset` (`name`, `path`, `unmapped_columns`), `columns`, and `embeddings`. Optional `unresolved` lists issues, `prediction_pipelines`, `prediction_assets`, `neighbors`, and `weights: "unweighted"` declare auxiliary contracts. Draft/unresolved bundles cannot run. You can validate draft structure without touching data:

```bash
parquet-representation-audit validate --schema dataset_schema.json --config audit_config.json --structure-only --allow-draft
```

Roles are `id`, `embedding`, `feature`, `prediction`, `probabilities`, `group`, `split`, `stratum`, `display`, `ignore`. Declare an ignore reason for unsupported/weight columns. An observed feature may additionally declare `auxiliary_roles: ["stratum"]` and remain a target. Explicit analysis references such as `nulls.stratum_column` choose auxiliary uses.

Feature kinds are `cluster_label`, `categorical`, `boolean`, `ordinal`, `count`, `numeric`, `multilabel`, `datetime`, `text`. Declare categorical `id_type: "integer"|"string"|"boolean"` (strings default; boolean kind implies boolean). Integer IDs are unordered labels and remain exact strings internally. Typed metadata is under `metadata_path`; its `id_type` must agree. Computational IDs never merge merely because readable names coincide. Unmapped observed labels remain raw IDs.

Use `missing_values` and `empty_is_unknown` explicitly; null is unknown. False/zero remain known by default. Missing sentinels are tested before nonnegative-count/range validation. Counts must be nonnegative integral values; use `numeric` for negative net quantities. Numeric fields retain units. `invalid_values: "unknown"` is an explicit permissive policy; `"error"` is the default. Permitted feature conversions are `none`, `integer`, `numeric`, `string`, `boolean`. The boolean conversion requires a `boolean_values` object mapping source text to JSON booleans. Identity fields accept lossless primitive values and do not support lossy conversion.

For tag sets use `encoding: "list"|"delimited"|"json"`. A literal `delimiter` is required for delimited strings; whitespace is not automatically special. Optional `strip`, `case`, `empty_token`, `min_count`, `max_labels`, or a fixed `vocabulary` specify preprocessing/budgets. A null row and an empty list/string differ when `empty_is_unknown: false`. An empty token inside a nonempty delimited string errors unless `empty_token: "drop"`. Null/numeric/nested tag tokens are rejected. Text fields are never automatically split into tags.

Ordinal fields require a unique typed `order`. The primary analysis remains categorical. To add order-aware numeric evidence, explicitly derive `ordinal_value` with a complete `mapping` keyed by normalized level ID and a `reasoning` statement. No lexical numeric ordering is inferred. Datetime fields require `timezone`; naive values are localized explicitly and ambiguous/nonexistent times error. Date operations are `year`, `month`, `dayofweek`, `unix_seconds`. Text may explicitly derive `text_length`. Derived names must not collide; rules become manifest provenance. Base text/time fields keep summaries and unavailable geometry/probes. Set base `geometry` and `probe` false if supplied explicitly.

`source_statistics: false` suppresses streamed source detail while retaining coverage. Geometry requires its sample reference statistics even when full-file detail is disabled. `geometry` and `probe` switches independently choose evaluable targets. Explicit derived targets default to both.

## Identity and assets

Declare one `role: "id"` or `dataset.identity_columns` for a composite key, with optional `canonicalization: "lossless_string"`. IDs must be nonmissing and unique across the entire Parquet. Strings retain leading zeros; large integers never pass through float. Full-file uniqueness is checked on disk during run. For source-only data without IDs, explicitly select `dataset.row_identity: "fingerprint_position"`; the entire source byte stream is hashed. Such identity is not stable across different exports.

Embedded vectors need `role: "embedding"`, `expected_dimension`, and a model entry with `column`, `name`, `main_normalization: "none"|"l2"`. External models need `matrix_path`, `ids_path`, `expected_dimension`, `name`, and `main_normalization`; both paths must be actual numeric/string NPY assets. No pickled object arrays. Asset model dimensions may differ. `missing_ids: "exclude"` requires the common invalid-row policy to permit exclusion. Read `CONTRACTS.md` for supplied graphs and predictions.

## Analysis configuration

Copy `examples/retail/audit_config.json` and change values deliberately. Required sections are `sample`, `source`, `limits`, `geometry`, `nulls`, `probes`, `normalization`, `topology`, `missingness`, `report`, plus `schema_version`, `mode`, `seed`, `output_policy`, `verbose`. Each accepted field is specified by `audit_config.schema.json`. Use the example's concrete caps rather than creating empty sections.

- `mode: "embedding"|"source_only"`. Source-only requires all five embedding analysis `enabled` flags false.
- `sample.strategy: "uniform_rows"|"all"`, `max_rows`. All must fit its explicit cap. The chosen file-row positions are saved. Sampling is uniform over file rows, not balanced over groups.
- `source.scope: "full"|"evaluation"`, batch/vocabulary/feature/quantile budgets. Population and evaluation references are separate.
- `geometry.metric: "cosine"|"euclidean"`, positive unique `ks`, `invalid_embeddings: "error"|"exclude"`, `reference: "evaluation"|"full_source"`, `distance_cap`, `distance_pairs`. Every k must fit the final cohort; no silent truncation. Full-source reference applies to label/tag prevalence and requires streamed full counts. Numeric global variance is the evaluation cohort's fixed variance.
- `nulls.modes: ["permutation"]` or include `"stratified"` with `stratum_column`; `permutations`, `correction: "none"|"bh"`, `testing_family`. Implemented BH family is `all_features_scales_modes_per_metric`.
- `probes.enabled`, `models`, `max_labels`, `trees`, `regularization_grid`, optional `group_column`, `split_column`. Splits default to 25% test; explicit split columns map configured source values to `train` and `test`. Groups protect both outer and tuning folds. A grid value is Ridge alpha or logistic C, recorded by model kind.
- `normalization.enabled`, optional feature selection, variants, `max_rows`, training-only grid; all four pipelines share feature-specific test rows.
- `topology.enabled`, bounded exact samples, repeats, geometries, pooled quantiles or explicit radii, shuffle settings, H1 and distance/simplex limits. The exact IDs and right censoring are exported.
- `missingness.enabled`, feature list, fractions, modes. Default recovery is a documented kNN illustration; Python callbacks replace it.
- `report` title, row/byte caps, PCA toggle, support thresholds, `trusted_ks` and descriptive color margins. Those margins are not confidence bounds.
- Optional `comparison.bootstraps` enables conditional paired anchor/group resampling (0 disables intervals).

All paths in the semantic schema resolve relative to that file. The analysis config has no asset path fields. CLI `--out` resolves relative to the shell working directory. Use fresh output folders; the only supported policy is `"fresh"`.
