"""Distinct inspection, contract validation and execution commands."""
import argparse
import json
from pathlib import Path
import sys
from .config import validate_bundle
from .ingest import inspect_parquet
from .runner import run_audit
from embedding_feature_audit.dataframe_report import write_json


def main(argv=None):
    parser=argparse.ArgumentParser(description='Schema-driven Parquet representation audit')
    subs=parser.add_subparsers(dest='command',required=True)
    inspect=subs.add_parser('inspect',help='Footer/physical schema and bounded sample; no semantics inferred')
    inspect.add_argument('--parquet',required=True);inspect.add_argument('--out',required=True);inspect.add_argument('--sample-rows',type=int,default=100)
    validate=subs.add_parser('validate',help='Validate configuration, physical types and asset headers before fitting')
    run=subs.add_parser('run',help='Stream source statistics, gather a bounded sample, audit, render offline HTML')
    for cmd in [validate,run]:
        cmd.add_argument('--schema',required=True);cmd.add_argument('--config',required=True)
    validate.add_argument('--structure-only',action='store_true');validate.add_argument('--allow-draft',action='store_true')
    run.add_argument('--out',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='inspect':
            if not 1<=args.sample_rows<=10000: raise ValueError('sample-rows must be between 1 and 10000')
            out=Path(args.out).resolve()
            if out.exists() and (not out.is_dir() or any(out.iterdir())): raise ValueError('Inspection output folder must be fresh')
            result=inspect_parquet(args.parquet,sample_rows=args.sample_rows);out.mkdir(parents=True,exist_ok=True)
            write_json(result,out/'inspection.json');(out/'inspection.md').write_text('# Parquet inspection\n\n'+f'Footer rows: {result["footer_rows"]:,}. {result["sample_scope"]}.\n\n'+ '\n'.join(f'- `{n}`: {r["physical_type"]}; semantic role unassigned.' for n,r in result['columns'].items())+'\n')
            print(out/'inspection.json')
        elif args.command=='validate':
            result=validate_bundle(args.schema,args.config,check_data=not args.structure_only,allow_draft=args.allow_draft)
            print(json.dumps({k:v for k,v in result.items() if k not in ['schema','config']},indent=2))
        else:
            result=run_audit(args.schema,args.config,args.out);print(result['report'])
        return 0
    except (ValueError,OSError,KeyError) as exc:
        print(f'Audit error: {exc}',file=sys.stderr);return 2


if __name__=='__main__': raise SystemExit(main())
