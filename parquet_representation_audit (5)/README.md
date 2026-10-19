# Parquet Representation Audit

Evaluate whether embedding neighborhoods organize the features in a Parquet dataset, then read the results in a standalone HTML report.

## Start here

1. Install with `python -m pip install -e .`.
2. Run `parquet-audit inspect --parquet data.parquet --out inspection`.
3. Use `inspection/prepare_prompt.md` to map the inspected columns into `dataset_schema.json` and `audit_config.json`. Examples and the feature type guide are in [`docs/`](docs/).
4. Run `parquet-audit run --schema dataset_schema.json --config audit_config.json --out audit_output`.
5. Open `audit_output/report.html`.

For complete Parquet-to-audit walkthroughs, start with [`examples/README.md`](examples/README.md). Each example includes a standard Parquet table, two row-aligned embeddings, a ready schema and config, and a generated report with its audit outputs.

The run checks the data contract itself and prints actionable errors. There is no separate preparation or validation stage to maintain.

## The data shape

One Parquet row represents one item. Keep each feature as a column aligned to those rows. A scalar numeric engineered feature stays one numeric column. A disjoint clustering stays one `cluster_label` column with exactly one cluster ID per item. A true multi-label feature stays one `multilabel` column containing a set/list of tags. Do not expand labels into one-hot columns just for this tool.

For XGBoost, install `python -m pip install -e ".[xgboost]"` and include `"xgboost"` in `probes.models`.

The brief documentation map is [`docs/index.md`](docs/index.md). The full data contract is in [`docs/data-model.md`](docs/data-model.md); analysis and report interpretation are in [`docs/analysis.md`](docs/analysis.md). These explain what transformations preserve meaning and which ones change it.

## Scale handling

`geometry.ks` is the ordered list of neighborhood sizes shared by the report tables and comparisons. Keep the same list and order across runs you intend to compare. Source summaries stream over the Parquet file; vector and graph analyses use the configured bounded evaluation sample. The core computations and human-readable report are preserved.

## Commands

- `parquet-audit inspect ...` writes a compact physical profile and a tailored mapping prompt. It does not guess feature semantics.
- `parquet-audit run ...` validates the schema and types, streams source statistics, samples rows if configured, computes analyses, and writes the report.

The JSON contract is version 1. This project does not translate or accept legacy configuration formats.
