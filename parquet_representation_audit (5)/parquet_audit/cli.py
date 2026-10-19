"""Distinct inspection, contract validation and execution commands."""
import argparse
from pathlib import Path
import sys
import json
from .ingest import inspect_parquet
from .runner import run_audit
from embedding_feature_audit.dataframe_report import write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description='Schema-driven Parquet representation audit')
    subs=parser.add_subparsers(dest='command',required=True)
    inspect=subs.add_parser('inspect',help='Footer/physical schema and bounded sample; no semantics inferred')
    inspect.add_argument('--parquet',required=True);inspect.add_argument('--out',required=True);inspect.add_argument('--sample-rows',type=int,default=100)
    run=subs.add_parser('run',help='Stream source statistics, gather a bounded sample, audit, render offline HTML')
    run.add_argument('--schema',required=True);run.add_argument('--config',required=True)
    run.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='inspect':
            if not 1<=args.sample_rows<=10000: raise ValueError('sample-rows must be between 1 and 10000')
            out=Path(args.out).resolve()
            if out.exists() and (not out.is_dir() or any(out.iterdir())): raise ValueError('Inspection output folder must be fresh')
            result=inspect_parquet(args.parquet,sample_rows=args.sample_rows);out.mkdir(parents=True,exist_ok=True)
            write_json(result,out/'inspection.json');(out/'inspection.md').write_text('# Parquet inspection\n\n'+f'Footer rows: {result["footer_rows"]:,}. {result["sample_scope"]}.\n\n'+ '\n'.join(f'- `{n}`: {r["physical_type"]}; semantic role unassigned.' for n,r in result['columns'].items())+'\n')
            (out/'prepare_prompt.md').write_text(_prepare_prompt(result))
            print(out/'inspection.json')
        else:
            result=run_audit(args.schema,args.config,args.out);print(result['report'])
        return 0
    except (ValueError,OSError,KeyError) as exc:
        print(f'Audit error: {exc}',file=sys.stderr);return 2


def _prepare_prompt(profile, config=None):
    """One focused prompt for mapping an inspected table into the current contract."""
    columns='\n'.join(f"- `{name}` ({item['physical_type']}): examples {', '.join(item['representative_previews']) or '(none)'}; sampled list lengths {item.get('sample_list_lengths',{})}; sampled list element types {item.get('sample_list_element_types',{})}"
                      for name,item in profile['columns'].items())
    config_template=json.dumps(config if config is not None else json.loads((Path(__file__).parent/'templates'/'audit_config.json').read_text()),indent=2)
    return f'''# Prepare this dataset for an embedding audit

Create exactly two files for the current `parquet-audit` schema-driven format: `dataset_schema.json` and `audit_config.json`. Use the inspected Parquet file below. Do not invent semantics from column names; flag uncertain columns as `ignore` and list questions that need a human answer. Do not create one-hot or expanded copies of feature columns.

## Table

- Path: `{profile['path']}`
- Rows: {profile['footer_rows']:,}
- Physical samples: {profile['sample_scope']}

{columns}

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
{{
  "schema_version": 1,
  "status": "ready",
  "dataset": {{"path": "{profile['path']}", "identity_columns": ["<stable ID column>"], "unmapped_columns": "error"}},
  "columns": {{
    "<stable ID column>": {{"role": "id", "id_type": "string"}},
    "<embedding column>": {{"role": "embedding", "expected_dimension": 768}},
    "<feature column>": {{"role": "feature", "kind": "numeric", "description": "...", "geometry": true, "probe": true}}
  }},
  "embeddings": [{{"name": "embedding", "column": "<embedding column>", "main_normalization": "l2", "expected_dimension": 768}}],
  "prediction_pipelines": {{}}
}}
```

The following is the complete current configuration. Copy it to `audit_config.json` and only change values when a data constraint requires it. Keep `geometry.ks` in the listed order for comparisons.

```json
{config_template}
```

If embeddings are stored separately from the Parquet table, use an external asset entry instead of declaring an embedding column. For example, `embedding.npy` must be an `N × D` real numeric matrix and `embedding_ids.npy` must contain its `N` item IDs. In this example, line `i` of both arrays corresponds to Parquet row `i`; the ID array lets the audit verify and align that relationship.

```json
{{"name": "model_a", "matrix_path": "embedding.npy", "ids_path": "embedding_ids.npy", "expected_dimension": 768, "main_normalization": "l2", "missing_ids": "error"}}
```

Return the two complete JSON files, a short list of unresolved semantic questions, and a brief explanation of any conversion that is not lossless. Copy the embedded `audit_config.json` template and keep the ordered `geometry.ks` values unchanged unless the user requests different neighborhood scales. Do not enable optional expensive analyses by default. Sample list lengths are clues only; verify a declared embedding dimension across the selected data during the audit.
'''


if __name__=='__main__': raise SystemExit(main())
