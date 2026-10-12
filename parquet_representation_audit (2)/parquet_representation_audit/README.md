# Parquet representation audit

Evaluate how observed information is organized in an explicitly declared embedding representation. This project generalizes the attached embedding-feature-audit 0.6.0 code and retains its **feature-first, offline human-interface report**: white cards, a five-column overview, every configured k, subset controls, direction/magnitude comparisons, every filtration radius, and detailed evidence.

It supports embedded numeric vectors, ID-aligned memory-mapped external A/B NPY matrices, and source-only analysis. Source counts and moments stream through projected Parquet batches; expensive geometry and fitting use a bounded sample selected **before** vector allocation. Results describe the declared scope, rather than asserting that an embedding is universally “good.”

## Install in a virtual environment

Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For tests: `python -m pip install '.[test]'`. Run commands from the extracted project root. No system Python changes are required.

## Run a supplied example

The ZIP already includes small synthetic Parquet/NPY assets and generated offline reports. No download or data generation is required.

```bash
parquet-representation-audit inspect --parquet examples/retail/orders.parquet --out retail_inspection
parquet-representation-audit validate --schema examples/retail/dataset_schema.json --config examples/retail/audit_config.json
parquet-representation-audit run --schema examples/retail/dataset_schema.json --config examples/retail/audit_config.json --out my_retail_audit
```

Open `my_retail_audit/report.html` directly in a browser. All charts, scripts, CSS and measurement data are local. Output folders must be fresh; rerun into a new folder rather than overwriting an existing audit.

Other prepared bundles:

| Bundle | Demonstrates |
|---|---|
| `examples/retail/` | Large exact integer IDs, embedded list vectors, dictionary categories, integer cluster labels, counts, booleans, ordinal order, list tags, date/text derived targets, supplied predictions/probabilities, explicit group-safe train/test split |
| `examples/telemetry/` | Leading-zero string IDs, different column names, pipe-delimited tags, shuffled external A/B IDs, different model dimensions, shared valid cohort, group splits |
| `examples/source_only/` | Same schema semantics with all embedding analyses explicitly disabled; source summaries and matched external-prediction evidence without invented geometry |

Generated reports are in `demo_generic_audit/`, `demo_telemetry_ab/`, and `demo_source_only/`. These are **synthetic demonstrations**, not measurements of your real dataset. Regenerate example inputs with `python examples/generate_examples.py` if desired.

## Prepare your own Parquet

Paste **`prompts/prepare_dataset.md`** into another chat with your actual schema, column meanings, optional assets and the three JSON Schemas under `parquet_audit/schemas/`. It asks that chat to produce a runnable bundle matching this implementation, including preparation notes and validation steps. `examples/retail/` is the filled prepared-bundle example. `docs/PREPARATION.md` explains the fields.

Keep two separate files:

- `dataset_schema.json`: roles, feature kinds, missing policies, units, label metadata, identity and asset alignment.
- `audit_config.json`: sampling, scopes, resource caps, geometry, nulls, splits/probes, normalization, topology and report controls.

Every source column needs a declared role or an explicit `unmapped_columns: "ignore"` policy. Inspection reports physical types and bounded observations; it never establishes semantic roles. A numerical dtype does not establish that a column is a count, ordered level or continuous target. IDs are not default targets. Cluster ID 0, count 0, and false are valid unless explicitly excluded. Missing tag annotation differs from an annotated empty set.

Relative JSON asset paths resolve against **the containing dataset schema**, independent of the working directory. CLI output paths resolve against the caller's working directory. Config overrides through Python should use the two real JSON files; there are no hidden path defaults.

## Plain Python API

```python
from parquet_audit import inspect_parquet, validate_bundle, run_audit

inspection = inspect_parquet("my_data.parquet", sample_rows=100)
validation = validate_bundle("dataset_schema.json", "audit_config.json")
result = run_audit("dataset_schema.json", "audit_config.json", "new_audit")
print(result["report"])
```

The numerical functions remain independently callable with arrays and dictionaries: `embedding_feature_audit.geometry.purity_lift`, `continuous_geometry`, `distance_relationship`, `modeling.run_probes`, `normalization.l2_normalize`, and `topology.component_filtration` / `rips_h1`. A recovery callback may be supplied to `run_audit(..., predictor=callback)`; see `docs/CONTRACTS.md`.

## What is measured

- Known coverage, label counts/prevalence/entropy, numeric moments/quantiles, tag co-occurrence and numeric feature–feature Spearman relationships on the evaluation sample.
- Across-k positive categorical/tag purity and lift with micro/macro weighting, local categorical/binary-tag entropy and known-neighbor support.
- Numeric edge absolute difference in the target's units, local/global variance and graph-specific Moran's I.
- Sampled distance–difference Spearman; bounded Euclidean distance correlation or accurately named cosine centered-distance association.
- Whole-known-row permutations on a fixed graph; optional strata, empirical tails, and explicitly configured BH correction families.
- Held-out training-only dummy baselines and linear logistic/Ridge probes; optional kNN/forest. A/B and normalization variants share feature splits.
- Raw standardized, unit direction, unit standardized, and norm-only inputs, with fit timing, selected regularization and convergence diagnostics. Training-only tuning preserves groups.
- Sampled exact distance filtrations, H0 mergers, component feature organization/null gaps, descriptive repeated-sample variation, and bounded H1 with triangle filling and cutoff censoring.
- Matched supplied predictions and probability metrics; artificial missingness with a replaceable recovery callback. Naturally missing truth never establishes prediction accuracy.
- Paired A/B anchor differences and optional conditional bootstrap intervals.

See `docs/METRICS.md`, `docs/CONTRACTS.md`, and `docs/SCALING.md` for formulas, alignment contracts and practical limits.

## Report and exports

The main table connects source information, all neighborhood scales, held-out probes, external pipelines and short interpretations. Feature selection applies to normalization and filtration tables. Brushing changes anchor/source summaries and rescoring of saved held-out predictions; it does not refit probes, narrow the candidate pool, or rerun global experiments. Models are compared on paired eligible anchors.

The output contains `manifest.json`, `resolved_configuration.json`, `row_alignment.csv`, `evaluation_sample.csv`, `asset_alignment.csv`, `excluded_embedding_rows.csv`, `feature_measurements.json`, complete per-feature CSVs, held-out predictions and splits, normalization diagnostics, topology trial/structure/merger/barcode tables, exact sampled IDs and scientific plots. `source_population/` stores the explicitly requested source scope, separately from evaluation metrics; its summaries name the scope and exact/approximate algorithms. Feature-level `population_source_summary.json` repeats that scope for convenience.

Exporting from the report downloads exact selected IDs, displayed values, settings and scope. The report safely escapes names, metadata and embedded JSON. Routine explanations use deterministic code, without network requests or model calls. Optional row exploration has row/byte caps; aggregate tables remain available when it is disabled.

## Compatibility and limits

The old `embedding-dataframe-audit` / `embedding-feature-audit` entry points and `embedding_feature_audit.audit_dataframe` are retained for existing workflows. They use the old in-memory configuration and discovery rules. The **new generic entry point** requires the two explicit versioned contracts and disables all source-domain column discovery. Do not feed the new schema to the legacy CLI.

Typed metadata adds signed integers, strings (including leading zeros) and booleans. Duplicate display names retain distinct IDs. Generic metadata/schema missing-policy disagreement requires an explicit reasoned override. Legacy metadata retains its historical canonicalization for compatibility. Cosine distance association is now correctly named; its JSON key is `centered_distance_association`, while Euclidean retains `distance_correlation`. Constant-truth R² and unestimable null statistics are undefined rather than misleadingly finite.

External assets/graphs currently require one stable scalar ID column; embedded Parquet supports explicit composite keys. One supplied prediction pipeline per target is supported per run. Probability assets are Parquet columns or ID-aligned Parquet tables, with explicit label order. Weights are unsupported: declare `weights: "unweighted"` and ignore a weight column explicitly, or validation rejects a weighting request. Structs need explicit `struct_path` extraction; map/other uninterpreted values must be ignored or converted outside the audit. H1 is limited to at most 128 sampled points and a configured simplex cap. No production ANN backend or encoder retraining is included.

This is suitable for streamed source summaries on very large Parquet files plus bounded expensive experiments. It has **not** been benchmarked on 8 million rows. Full exact graphs, full-population tag matrices and full Rips persistence at that scale are not supported defaults. External vector gathering uses a capped population ID index; budget that memory separately from vectors. See `VALIDATION.md` for checks actually run and the unavailable browser-rendering verification.
