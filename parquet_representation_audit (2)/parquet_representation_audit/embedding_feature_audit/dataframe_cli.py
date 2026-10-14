"""Parquet CLI for the DataFrame-first project."""
import argparse
import json
from pathlib import Path
import numpy as np
from .dataframe import load_dataframe
from .dataframe_report import audit_dataframe
from .cluster_metadata import resolve_config_metadata_paths


def main():
    parser=argparse.ArgumentParser(description='Audit Danbooru DataFrame features against aligned embeddings')
    parser.add_argument('--dataframe',required=True,type=Path,help='Parquet with features and optionally an embedding column')
    parser.add_argument('--embeddings',type=Path,help='Row-aligned N×D .npy (numeric only)')
    parser.add_argument('--embedding-ids',type=Path,help='IDs for embedding rows; must exactly match dataframe row order')
    parser.add_argument('--neighbors',type=Path,help='N×K .npy of row positions, excluding self')
    parser.add_argument('--neighbor-ids',type=Path,help='Required with --neighbors, proves row ordering')
    parser.add_argument('--predictions',type=Path,help='Prediction Parquet, joined by unique post ID')
    parser.add_argument('--compare-embeddings',type=Path,help='Second N×D .npy; IDs align it to the source cohort')
    parser.add_argument('--compare-embedding-ids',type=Path,help='Required with --compare-embeddings; unique IDs in that matrix order')
    parser.add_argument('--compare-neighbors',type=Path,help='Optional second graph in dataframe row order')
    parser.add_argument('--compare-neighbor-ids',type=Path,help='Required with --compare-neighbors; must match dataframe row order')
    parser.add_argument('--drop-invalid-embeddings',action='store_true',help='Exclude nonfinite/zero cosine rows in either model; write excluded IDs')
    parser.add_argument('--no-explorer',action='store_true',help='Keep aggregate scorecards, omit interactive row snapshot')
    parser.add_argument('--normalize-embeddings',choices=['none','l2'],help='Main embedding preprocessing; unit normalization discards magnitude')
    parser.add_argument('--normalization-analysis',action='store_true',help='Compare same-split raw, unit, standardized-unit and norm-only linear probes')
    parser.add_argument('--topology',action='store_true',help='Enable sampled H0/feature filtrations and bounded H1; configure limits in JSON')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--cluster-metadata-dir',type=Path,help='Recursively discover cluster JSON by name; paths relative to working directory')
    parser.add_argument('--out',type=Path,default=Path('dataframe_audit'))
    args=parser.parse_args()
    config=resolve_config_metadata_paths(json.loads(args.config.read_text()),args.config.resolve().parent) if args.config else {}
    if args.cluster_metadata_dir is not None:
        config['cluster_metadata_dir']=str(args.cluster_metadata_dir.resolve())
    if args.drop_invalid_embeddings: config['drop_invalid_embeddings']=True
    if args.no_explorer: config['explorer']=False
    if args.normalize_embeddings: config['embedding_normalization']=args.normalize_embeddings
    if args.normalization_analysis: config['normalization_analysis']={**config.get('normalization_analysis',{}),'enabled':True}
    if args.topology: config['topology']={**config.get('topology',{}),'enabled':True}
    df=load_dataframe(args.dataframe)
    ids=df[config.get('id_column','id')].astype(str).to_numpy(dtype=str)
    if args.embeddings and not args.embedding_ids:
        parser.error('--embedding-ids is required with --embeddings to verify alignment')
    if args.neighbors and not args.neighbor_ids:
        parser.error('--neighbor-ids is required with --neighbors to verify alignment')
    if args.compare_embeddings and not args.compare_embedding_ids:
        parser.error('--compare-embedding-ids is required with --compare-embeddings')
    if (args.compare_embedding_ids or args.compare_neighbors or args.compare_neighbor_ids) and not args.compare_embeddings:
        parser.error('Comparison IDs and graphs require --compare-embeddings')
    if args.compare_neighbors and not args.compare_neighbor_ids:
        parser.error('--compare-neighbor-ids is required with --compare-neighbors')
    for path in [args.embedding_ids,args.neighbor_ids,args.compare_neighbor_ids]:
        if path is not None and not np.array_equal(np.load(path,allow_pickle=False).astype(str),ids):
            parser.error(f'IDs in {path} do not match dataframe row order')
    z=np.load(args.embeddings,mmap_mode='r',allow_pickle=False) if args.embeddings else None
    graph=np.load(args.neighbors,mmap_mode='r',allow_pickle=False) if args.neighbors else None
    pred=load_dataframe(args.predictions) if args.predictions else None
    zb=np.load(args.compare_embeddings,mmap_mode='r',allow_pickle=False) if args.compare_embeddings else None
    b_ids=np.load(args.compare_embedding_ids,allow_pickle=False) if args.compare_embedding_ids else None
    b_graph=np.load(args.compare_neighbors,mmap_mode='r',allow_pickle=False) if args.compare_neighbors else None
    result=audit_dataframe(df,args.out,config=config,embeddings=z,neighbors=graph,predictions=pred,
                           comparison_embeddings=zb,comparison_ids=b_ids,comparison_neighbors=b_graph)
    print(f'Report: {result["report"]}')


if __name__=='__main__':
    main()
