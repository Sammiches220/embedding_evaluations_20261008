"""Create two worked preparations using the real inspect-generated prompt template.

These are authored reference responses, not claims that an external LLM was run.
Run after generate_examples.py. The prepared JSON files are the actual audited inputs.
"""
import json
from pathlib import Path
from parquet_audit.cli import _prepare_prompt
from parquet_audit.ingest import inspect_parquet

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'audit_config.json').read_text())
FACTS={
 'numeric_categorical':'''- One row is one customer. `customer_id` is an exact string ID.
- `age_years` and `tenure_days` are nonnegative integer counts. `annual_income_usd` and `monthly_spend_usd` are real measurements, with the units shown by their names. Do not bin or standardize the source values.
- `plan_type` is a nominal string category with no ordinal order. `churned` is an observed boolean, and false is not missing.
- Every source column is wanted in the audit. No sentinel values represent missingness.
- Both embeddings have 360 rows and 16 float32 coordinates. The names are `signal_embedding` and `noisier_embedding`. Their files are `embedding_a.npy`, `embedding_b.npy`, `embedding_a_ids.npy`, and `embedding_b_ids.npy`.
- For BOTH embeddings, matrix row i and ID array entry i correspond exactly to Parquet row i. Require missing_ids=error; keep the ID files to verify alignment.
- Use L2 normalization for the main cosine geometry, while retaining raw vectors for the enabled direction/magnitude experiments. This deliberately excludes norm from main cosine distance; do not claim main cosine geometry retains vector magnitude.''',
 'disjoint_clusters':'''- One row is one item. `item_id` is an exact integer ID; never route it through float.
- `measure_x` and `measure_y` are signed numeric coordinates. `event_count` is a nonnegative integer count.
- `cluster_4` assigns exactly one of four mutually exclusive cluster IDs per row; `cluster_8` assigns exactly one of eight IDs. These are TWO separate clusterings of the SAME rows. Their IDs are nominal and have no numeric order. Preserve each as one cluster_label column.
- Cluster ID 0 is an observed cluster, not unknown. No source column should be removed. No numeric binning or one-hot conversion is requested.
- Both external embeddings have 360 rows and 16 float32 coordinates. Use names `signal_embedding` and `noisier_embedding`; assets `embedding_a.npy`, `embedding_b.npy`, `embedding_a_ids.npy`, `embedding_b_ids.npy`.
- For BOTH embeddings, matrix row i and ID array entry i correspond to Parquet row i. Require missing_ids=error; the arrays have exactly the source item IDs in that order.
- Use L2 normalization for the main cosine geometry. Keep the original vectors for the enabled raw/direction/magnitude comparisons.'''
}
for name,facts in FACTS.items():
 folder=ROOT/name
 profile=inspect_parquet(folder/'data.parquet',sample_rows=12)
 profile['path']='data.parquet'  # Prompt/output files are used beside this source table.
 prompt=_prepare_prompt(profile,CONFIG)
 prompt+='\n## User-supplied meanings and requested settings\n\n'+facts+'\n\nWrite the two JSON files beside data.parquet. Use the complete supplied configuration; these small walkthroughs intentionally enable normalization and topology. Do not alter the Parquet file or embedding arrays.\n'
 (folder/'prep_prompt.md').write_text(prompt)
 schema=(folder/'dataset_schema.json').read_text()
 response=f'''# Worked response: {name}

This is an authored reference response to [prep_prompt.md](prep_prompt.md). The JSON below is identical to the ready schema and shared configuration used for the included audit. No external LLM was invoked to produce this walkthrough.

## dataset_schema.json

```json
{schema.rstrip()}
```

## audit_config.json

```json
{json.dumps(CONFIG,indent=2)}
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
'''
 (folder/'prep_response.md').write_text(response)
 print(name,'prompt and worked response created')
