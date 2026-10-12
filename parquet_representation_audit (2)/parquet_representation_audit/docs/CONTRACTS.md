# Alignment and auxiliary asset contracts

## External embeddings

A model entry contains `name`, `matrix_path`, `ids_path`, `expected_dimension`, `main_normalization`, optionally `missing_ids`. Matrices are real numeric N×D NPY files; IDs are nonobject one-dimensional NPY arrays of the same length and declared scalar type. IDs must be unique after lossless canonicalization. Rows can be shuffled independently for A/B; ID lookup restores the source sample order. Extra asset IDs are allowed. Missing sampled IDs error by default, or explicitly become exclusions.

Source IDs are validated across the full file, using SQLite uniqueness on temporary disk. Each external ID index is loaded/capped; the large vector matrix is memory mapped. Gathering copies only selected rows in `external_batch_rows` batches. This is not a streaming full-vector audit. All source IDs, matrix headers and retained vectors are checked; unsampled matrix contents are not claimed to be validated. Composite external keys must be precomposed into a stable scalar string outside the audit.

Nonfinite/missing embeddings or zero vectors required by direction-based analyses follow `geometry.invalid_embeddings`. A/B use the same intersection of valid rows, the same feature-known masks and candidate pool. Exclusion reasons identify each model. Ragged, textual, complex and nonnumeric vectors error. Vector null components become nonfinite rather than being silently zero-filled. Large IDs never pass through pandas float inference.

## Supplied neighbors

Optional `neighbors` entries contain `embedding`, `indices_path`, `ids_path`, `distances_path`, `index_kind: "positions"|"ids"`, `metric`, `candidate_scope: "evaluation_cohort"`, `self_excluded: true`, `ordered: true`, `preprocessing`, `provenance`. Anchor IDs must exactly match the final evaluation row order. Position neighbors are in [0,N), with no duplicates or self edges. ID neighbors are resolved inside the declared pool. Distances must have the same shape, be finite/nonnegative, ordered, and agree with the declared representation and metric. Search recall/fidelity is still the supplier's responsibility.

A supplied graph requires `sample.strategy: "all"` on its explicitly exported bounded Parquet cohort. The audit never silently filters/reindexes that graph. Fix invalid rows and export aligned graphs explicitly. A/B graph contracts use identical pools, masks, metric and preprocessing. Large population ANN pools are not supported by this first release; prepare a bounded cohort whose exported neighbors obey these contracts.

## Predictions and probabilities

Inline columns declare `role: "prediction"` or `"probabilities"`, `target`, `pipeline`. Pipelines under `prediction_pipelines` declare `uses_embedding: true|false|"unknown"`, optional description/provenance. One prediction column per target is supported per run. Predictions inherit the target's type/parser/missing semantics, with optional explicit prediction sentinels.

Probability columns have numeric list cells and an explicit typed `label_order`. Categorical vectors must sum to one and stay in [0,1]. Multilabel entries are independent positive-label probabilities and need not sum to one. Unknown probability rows are excluded; zero probabilities are valid. Output includes matched support, log loss for categories, mean one-versus-rest Brier score, per-label ROC AUC where both classes exist, and AP where positives exist. Explicit label order must cover known categorical truth; no implicit vocabulary ordering is invented.

External prediction tables are declared under `prediction_assets` as `{ "path": "predictions.parquet", "id_column": "sample_id", "columns": { "predicted_status": { "role": "prediction", "target": "status", "pipeline": "external" } } }`. These Parquet tables need unique stable IDs of the source ID type. Only sampled IDs are retained. Extra IDs are allowed, unavailable sampled predictions stay unknown, and external column names may not collide with source columns. Probabilities can use the same route. NPY probability arrays need an explicit preparatory conversion to this Parquet contract; they are not silently assumed aligned.

Matched observed truth measures accuracy; predictions without truth describe output distribution only. If predictions use embeddings, the metric measures that entire pipeline. Probability evidence is also usable in source-only mode. Geometric comparisons are unavailable there. Separate model fits, annotation bias and incomplete labels affect interpretation.

## Artificial missingness

Enable `missingness` and choose features, fractions and `random` or `rare-first` hiding. Hiding is performed only on observed truth. The supplied callback signature is:

```python
def recover(z, y, train_rows, hidden_rows, kind):
    # Fit on y[train_rows]; return predictions for z[hidden_rows].
    ...
```

Call `run_audit(schema_path, config_path, fresh_out, predictor=recover)`. This callback is trusted user-supplied Python, not executable text from configuration. The default is a replaceable embedding-based kNN illustration. Do not use its coherence as independent validation. Groups are preserved by the hiding experiment's configured group handling. Accuracy on naturally unannotated rows is never claimed.

## Provenance and scope

`manifest.json` records schema/configuration, versions, seed, scopes, sampling, scales, resource estimates and identity validation. `asset_alignment.csv` identifies model-specific invalid rows, `evaluation_sample.csv` records the originally sampled file positions and inclusion in the common cohort, and `row_alignment.csv` maps final row positions to exact item identities and source positions. Feature-known masks change feature eligibility, not the full embedding cohort. Normalization splits and topology sampled IDs are separately exported.
