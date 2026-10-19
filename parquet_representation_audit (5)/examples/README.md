# From a pandas Parquet table to an audit

These three small examples start with ordinary pandas data tables and keep each embedding outside the Parquet file. Each case includes:

- `data.parquet`: one row per item, with feature columns in normal pandas/Parquet form.
- `embedding_a.npy` and `embedding_b.npy`: two numeric `N × D` matrices for distinct embedding representations.
- `embedding_a_ids.npy` and `embedding_b_ids.npy`: row IDs for each matrix; each is in exact Parquet row order.
- `dataset_schema.json`: the semantic mapping the audit needs.
- `audit_output/report.html`: the report produced by actually running the project.

`dataset_schema.template.json` shows the generic schema shape; the case-specific schemas are complete runnable examples.

For every case, row `i` in both embedding matrices is the vector for Parquet row `i`; both ID arrays have the exact same order as the Parquet ID column. The report lets you select which embedding drives “Features at a glance” and compare one against the other. The generator checks this after writing and the project independently aligns external arrays by the declared IDs.

## Choose an example

| Example | Ordinary table contents | How the project reads it |
| --- | --- | --- |
| `numeric_categorical` | Numeric measurements, integer counts, a string category, and a boolean outcome | Keep numeric columns numeric, category as `categorical`, outcome as `boolean` |
| `disjoint_clusters` | Numeric features plus two integer cluster-assignment columns | Each clustering stays a single `cluster_label` value per row; each assignment is a separate feature column |
| `multilabel_tags` | Numeric product features and a list of tags that can co-occur | Keep each row's list in one `multilabel` column; do not make one column per tag |

The first two tables resemble typical classical ML feature tables. A classical model may apply scaling and one-hot encoding inside its training pipeline. The audit reads source columns directly and declares their meaning; no model-ready one-hot matrix is required.

| Typical pandas column | Declare as | Keep as | Lossless? |
| --- | --- | --- | --- |
| Real-valued engineered feature | `numeric` | One numeric column | Yes, when raw values and units are retained |
| Nonnegative integer count | `count` | One integer column | Yes |
| Nominal category / boolean | `categorical` / `boolean` | One scalar value per row | Yes; do not map labels to arbitrary integers |
| One clustering with one assignment per item | `cluster_label` | One cluster ID column | Yes; each independent clustering gets its own column |
| Tags that can co-occur | `multilabel` | One list/set per row | Yes, when the set and missingness semantics are preserved |
| Several engineered measurement columns | Separate `numeric` feature columns | Retain every scalar column | Values are preserved; the audit does not test their full joint distribution |

The two separate embedding arrays are not derived from or substituted for the source features. They are the representation being evaluated. In each case, both matrices have one row per Parquet row and a parallel ID file. The project verifies those IDs and evaluates both embeddings on the same aligned rows. This distinction matters: a feature table by itself is not an embedding, and arbitrary engineered feature columns should not be silently treated as one unless they are intentionally the vector representation.

## Run the examples

From the project root:

```bash
python -m pip install -e ".[xgboost]"
python examples/generate_examples.py
parquet-audit run --schema examples/numeric_categorical/dataset_schema.json --config examples/audit_config.json --out examples/numeric_categorical/audit_output_rerun
parquet-audit run --schema examples/disjoint_clusters/dataset_schema.json --config examples/audit_config.json --out examples/disjoint_clusters/audit_output_rerun
parquet-audit run --schema examples/multilabel_tags/dataset_schema.json --config examples/audit_config.json --out examples/multilabel_tags/audit_output_rerun
```

The included reports were generated with these commands, writing to `audit_output`. Each output contains the HTML report, machine-readable result tables, and audit metadata. The shared configuration enables probes, normalization analysis, and topology analysis so the report sections are populated. The output path must be new or empty. These are synthetic workflow demonstrations, not claims about a trained production encoder.

## What changes from the usual ML table?

Usually, only two things:

1. Add a stable ID column if the table does not already have one.
2. Add a schema that tells the audit which columns are IDs, features, and what each feature means. For separate vectors, save an ordered `.npy` matrix and a parallel ID `.npy` array, then point to both in `embeddings`.

Do not expand a scalar category, disjoint cluster label, or tag set just to make it look like an ML design matrix. Keep values in their native form and declare the correct `kind`. The audit checks types and alignment when it runs.

## Use a preparation prompt

Two [worked prompt examples](prompt_walkthroughs.md) show the same template used by `parquet-audit inspect`, filled with real example schemas and domain facts. Each includes the complete prompt, a worked response with the two JSON files, and the report produced from those files.

The shared example configuration selects `linear`, `knn` and `xgboost` plus their dummy baseline for both the overview and normalization experiments. Read [information preservation and limits](../docs/information-preservation.md) before treating strong scores as proof that the whole dataset is represented.
