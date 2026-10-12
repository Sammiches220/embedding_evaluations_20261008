"""Audits from explicit contracts; ingestion, encoding, fitting and report adapters stay separate."""
import json
import platform
from pathlib import Path
import sys
from importlib.metadata import version
import numpy as np
import pandas as pd
from .config import validate_bundle,roles
from .semantics import KINDS,computation_spec,is_null
from .ingest import (source_statistics,sample_positions,gather_parquet,external_vectors,
                     embedded_vectors,supplied_graph,identity_columns,sample_source_summaries)
from .predictions import attach_predictions,probability_metrics
from .reporting import source_report,refresh_report
from embedding_feature_audit.dataframe_report import audit_dataframe,write_json,save_table
from embedding_feature_audit.normalization import safe_norms


def run_audit(schema_path: str | Path,config_path: str | Path,out: str | Path,*,predictor=None) -> dict:
    """Validate, stream source statistics, gather an explicit bounded cohort, audit and render.

    `predictor(z,y,train_rows,hidden_rows,kind)` replaces the documented artificial
    missingness recovery callback. Paths in JSON are relative to their containing
    schema; output CLI paths are relative to the caller's current directory.
    """
    checked=validate_bundle(schema_path,config_path);schema=checked['schema'];cfg=checked['config']
    output=Path(out).resolve();input_path=Path(schema['dataset']['path']).resolve()
    if output==input_path or output in input_path.parents: raise ValueError('Output cannot replace a source file or its parent directory')
    if output.exists() and (not output.is_dir() or any(output.iterdir())): raise ValueError('Output folder must be fresh/empty; no source files or prior runs overwritten')
    progress=(lambda text:print(text,flush=True)) if cfg['verbose'] else None
    if progress: progress('Validating all source identities and streaming declared feature summaries…')
    summaries,source_tables,source_info=source_statistics(schema,cfg,progress=progress)
    positions=sample_positions(checked['n_rows'],cfg)
    if progress: progress(f'Gathering {len(positions):,} explicitly sampled evaluation rows before vector allocation…')
    frame,derived=gather_parquet(schema,cfg,positions,source_fingerprint=source_info['fingerprint'])
    specs={n:s for n,s in schema['columns'].items() if s['role']=='feature'};specs.update(derived)
    idcol='_audit_item_identity';ids=frame[idcol].to_numpy(dtype=str);idcols=identity_columns(schema)
    original_sample_ids=ids.copy()
    frame,pred_specs=attach_predictions(frame,schema,cfg)
    arrays=[];alignment=[];reasons=[[] for _ in range(len(frame))]
    direction_required=cfg['geometry']['metric']=='cosine' or cfg['normalization']['enabled'] and bool(set(cfg['normalization'].get('variants',['unit']))&{'unit','unit_standardized'}) or cfg['topology']['enabled'] and 'unit_chordal' in cfg['topology'].get('spaces',['unit_chordal'])
    for model in schema['embeddings'] if cfg['mode']=='embedding' else []:
        if 'column' in model: z=embedded_vectors(frame,model,schema['columns'][model['column']]);absent=np.zeros(len(frame),bool)
        else: z,absent=external_vectors(model,ids,schema['columns'][idcols[0]],cfg)
        finite=np.isfinite(z).all(1);direction=direction_required or model['main_normalization']=='l2'
        zero=finite & (np.max(np.abs(z),axis=1)==0) if direction else np.zeros(len(z),bool)
        for i in range(len(frame)):
            reason='missing_external_id' if absent[i] else 'nonfinite_vector' if not finite[i] else 'zero_direction' if zero[i] else None
            if reason: reasons[i].append(model['name']+':'+reason)
            alignment.append({'item_id':ids[i],'model':model['name'],'source_position':int(positions[i]),'aligned':not absent[i],'valid':reason is None,'reason':reason})
        arrays.append(z)
    good=np.array([not r for r in reasons]);excluded=pd.DataFrame([{'item_id':ids[i],'source_position':int(positions[i]),'reason':';'.join(reasons[i])} for i in range(len(frame)) if not good[i]],columns=['item_id','source_position','reason'])
    if len(excluded):
        if cfg['geometry']['invalid_embeddings']=='error': raise ValueError(f'{len(excluded)} common-cohort rows invalid: {excluded.iloc[0].reason}; explicitly select invalid_embeddings=exclude')
        if schema.get('neighbors'): raise ValueError('Cannot filter invalid embeddings with a supplied graph; export/rebuild explicitly aligned assets first')
        frame=frame.iloc[np.flatnonzero(good)].reset_index(drop=True);ids=ids[good];arrays=[z[good] for z in arrays]
    if cfg['mode']=='embedding' and (len(frame)<2 or any(k>=len(frame) for k in cfg['geometry']['ks'])):
        raise ValueError('Common valid cohort is too small for requested k; fix data or explicitly choose smaller scales')
    # Explicit group leakage is a contract error, not a silently skipped probe.
    group=cfg['probes'].get('group_column') or (roles(schema,'group')+[None])[0]
    split=cfg['probes'].get('split_column') or (roles(schema,'split')+[None])[0]
    if group and split:
        train=set(x for x,s in zip(frame[group],frame[split]) if s=='train' and not is_null(x));test=set(x for x,s in zip(frame[group],frame[split]) if s=='test' and not is_null(x))
        if train&test: raise ValueError('Explicit train/test split leaks related-item groups')
    graph_assets={asset['embedding']:asset for asset in schema.get('neighbors',[])};graphs=[];graph_distances=[]
    for model,z in zip(schema['embeddings'],arrays):
        asset=graph_assets.get(model['name'])
        if asset:
            graph,dist=supplied_graph(asset,ids,schema['columns'][idcols[0]],cfg)
            from embedding_feature_audit.exploration import edge_distances
            from embedding_feature_audit.normalization import l2_normalize
            expected=edge_distances(l2_normalize(z) if model['main_normalization']=='l2' else z,graph,cfg['geometry']['metric'])
            if not np.allclose(dist,expected,rtol=1e-4,atol=1e-6): raise ValueError('Supplied graph distances disagree with declared metric/vectors')
            graphs.append(graph);graph_distances.append(dist)
        else: graphs.append(None);graph_distances.append(None)
    output.mkdir(parents=True,exist_ok=True)
    computable={n:computation_spec(s) for n,s in specs.items() if KINDS[s['kind']] is not None}
    for name,spec in computable.items():
        if specs[name]['kind']=='multilabel' and 'vocabulary' not in spec:
            # Vocabulary frozen on declared source scope; geometry/probe cap is explicitly recorded.
            table=source_tables.get(name,pd.DataFrame())
            if cfg['source']['scope']=='full' and len(table): spec['vocabulary']=table[table['count']>=spec['min_count']].label.head(spec['max_labels']).tolist()
    missingness=cfg['missingness'];probe_names=[n for n,s in specs.items() if n in computable and s.get('probe',True)]
    geom_names=[n for n,s in specs.items() if n in computable and s.get('geometry',True)]
    def selection(requested,eligible):
        return eligible if requested is True else [n for n in requested if n in eligible] if isinstance(requested,list) else []
    norm={**cfg['normalization'],'features':selection(cfg['normalization'].get('features',True),probe_names)}
    topo={**cfg['topology'],'features':selection(cfg['topology'].get('features',True),geom_names)}
    predictions={}
    for name,spec in pred_specs.items():
        if spec['role']=='prediction': predictions[spec['target']]={'column':name,'uses_embedding':schema['prediction_pipelines'][spec['pipeline']]['uses_embedding'],'missing_values':[]}
    baseline={}
    if cfg['geometry']['reference']=='full_source':
        if cfg['source']['scope']!='full': raise ValueError('full_source prevalence requires source.scope=full')
        baseline={n:dict(zip(t.label,t.prevalence)) for n,t in source_tables.items() if n in computable and len(t)}
    internal=dict(generic_schema=True,id_column=idcol,ks=cfg['geometry']['ks'],metric=cfg['geometry']['metric'],
        permutations=cfg['nulls']['permutations'],null_modes=cfg['nulls']['modes'],null_correction=cfg['nulls']['correction'],
        seed=cfg['seed'],verbose=cfg['verbose'],probe_features=probe_names if cfg['probes']['enabled'] else False,
        source_only_targets=[n for n in computable if n not in geom_names],probe_models=cfg['probes']['models'],
        max_probe_labels=cfg['probes']['max_labels'],trees=cfg['probes']['trees'],probe_regularization_grid=cfg['probes']['regularization_grid'],
        missingness_features=selection(missingness['features'],geom_names) if missingness['enabled'] else False,
        missingness_fractions=missingness['fractions'],missingness_modes=missingness['modes'],predictions=predictions,
        group_column=group,split_column=split,strata_column=cfg['nulls'].get('stratum_column'),population_prevalence=baseline,
        pca_overview=cfg['report']['pca'],explorer=cfg['report']['row_explorer'],explorer_max_rows=cfg['report']['max_rows'],explorer_max_bytes=cfg['report']['max_bytes'],
        comparison_bootstraps=cfg.get('comparison',{}).get('bootstraps',0),drop_invalid_embeddings=False,
        embedding_name=schema['embeddings'][0]['name'] if arrays else 'Unavailable',comparison_name=schema['embeddings'][1]['name'] if len(arrays)>1 else None,
        overview=cfg['report']['overview'],embedding_normalization=schema['embeddings'][0]['main_normalization'] if arrays else 'none',
        normalization_analysis=norm,topology=topo,report_title=cfg['report']['title'],distance_cap=cfg['geometry']['distance_cap'],distance_pairs=cfg['geometry']['distance_pairs'],supplied_graph_contract_validated=True)
    if cfg['source']['scope']=='evaluation':
        summaries,source_tables=sample_source_summaries(frame,specs)
    if cfg['mode']=='source_only':
        result=source_report(output,schema,cfg,summaries,source_tables,frame)
    else:
        result=audit_dataframe(frame,output,features=computable,config=internal,embeddings=arrays[0],
            comparison_embeddings=arrays[1] if len(arrays)>1 else None,neighbors=graphs[0],comparison_neighbors=graphs[1] if len(graphs)>1 else None,predictor=predictor)
        # Independent source/geometry/probe switches; a target may be probed without geometric metrics.
        if cfg['probes']['enabled']:
            from embedding_feature_audit.dataframe import encode_feature
            from embedding_feature_audit.modeling import run_probes
            from embedding_feature_audit.normalization import l2_normalize
            groups=None if not group else np.array([f'row:{i}' if is_null(v) else v for i,v in enumerate(frame[group])])
            for name in internal['source_only_targets']:
                if name not in probe_names: continue
                y,valid,names=encode_feature(frame[name],computable[name]);kind=computable[name]['kind']
                if kind=='multilabel': names=names[:cfg['probes']['max_labels']];y=y[:,:len(names)]
                for i,z in enumerate(arrays):
                    matrix=l2_normalize(z) if schema['embeddings'][i]['main_normalization']=='l2' else z
                    try:
                        scores,per_label,predicted,train,test=run_probes(matrix,y,valid,names,kind,cfg['seed'],groups,None if not split else frame[split].to_numpy(),cfg['probes']['trees'],cfg['probes']['models'],regularization_grid=cfg['probes']['regularization_grid'])
                        field='probes' if i==0 else 'comparison_probes';result['features'][name][field]=scores
                        result['features'][name]['probe_status']=None
                        for tab,file in [(scores,field+'.csv'),(per_label,field+'_per_label.csv'),(predicted,field+'_predictions.csv')]: save_table(tab,result['features'][name]['folder']/'tables'/file)
                    except ValueError as exc: result['features'][name]['probe_status']=str(exc)
        for name,spec in pred_specs.items():
            if spec['role']=='probabilities':
                target=spec['target'];scores=probability_metrics(frame[target],frame[name],schema['columns'][target],spec)
                scores['predictor_inputs']=schema['prediction_pipelines'][spec['pipeline']]['uses_embedding']
                prior=result['features'][target].get('prediction_metrics',{});prior.update(scores);prior.setdefault('n_compared',scores['probability_n_matched'])
                result['features'][target]['prediction_metrics']=prior;write_json(prior,result['features'][target]['folder']/'prediction_metrics.json')
    # Supplied predictions remain meaningful without an embedding. Evaluate only
    # known truth; direct source-only targets never receive fabricated geometry.
    from embedding_feature_audit.dataframe import encode_feature
    from embedding_feature_audit.statistics import prediction_metrics,feature_source
    for column,pred_spec in pred_specs.items():
        target=pred_spec['target'];target_result=result['features'].get(target)
        if target_result is None: continue
        if pred_spec['role']=='prediction' and 'prediction_metrics' not in target_result:
            spec=computable[target];kind=spec['kind']
            truth,known,names=encode_feature(frame[target],spec)
            predicted,pknown,_=encode_feature(frame[column],spec,names if kind=='multilabel' else None)
            matched=known&pknown;score,per_label=prediction_metrics(truth[matched],predicted[matched],kind,names)
            score.update(predictor_inputs=schema['prediction_pipelines'][pred_spec['pipeline']]['uses_embedding'],evaluation_scope='Matched known truth and supplied predictions in the explicit evaluation sample')
            target_result['prediction_metrics']=score;target_result['prediction_series']=frame[column]
            write_json(score,target_result['folder']/'prediction_metrics.json');save_table(per_label,target_result['folder']/'tables/prediction_per_label.csv')
            dist,counts,_=feature_source(frame[column],predicted,pknown,names,kind,spec)
            write_json(dist,target_result['folder']/'predicted_source_summary.json');save_table(counts,target_result['folder']/'tables/predicted_source_prevalence.csv')
        elif pred_spec['role']=='probabilities' and 'probability_n_matched' not in target_result.get('prediction_metrics',{}):
            score=probability_metrics(frame[target],frame[column],schema['columns'][target],pred_spec)
            score['predictor_inputs']=schema['prediction_pipelines'][pred_spec['pipeline']]['uses_embedding']
            prior=target_result.get('prediction_metrics',{});prior.update(score);prior.setdefault('n_compared',score['probability_n_matched'])
            target_result['prediction_metrics']=prior;write_json(prior,target_result['folder']/'prediction_metrics.json')
    population=output/'source_population';population.mkdir(exist_ok=True)
    write_json(summaries,population/'summaries.json')
    for number,(name,table) in enumerate(source_tables.items(),1): save_table(table,population/f'{number:03d}.csv')
    if source_tables: write_json({n:f'{i:03d}.csv' for i,n in enumerate(source_tables,1)},population/'table_index.json')
    refresh_report(output,result,schema,cfg,summaries,source_tables,specs)
    pd.DataFrame(alignment,columns=['item_id','model','source_position','aligned','valid','reason']).to_csv(output/'asset_alignment.csv',index=False)
    excluded.to_csv(output/'excluded_embedding_rows.csv',index=False)
    evaluated=pd.DataFrame({'row_position':np.arange(len(frame)),'item_id':frame[idcol],'source_position':frame['_audit_source_position']})
    evaluated.to_csv(output/'row_alignment.csv',index=False)
    pd.DataFrame({'source_position':positions,'item_id':original_sample_ids,'included_common_cohort':good}).to_csv(output/'evaluation_sample.csv',index=False)
    manifest=json.loads((output/'manifest.json').read_text());manifest.update(contract_version=1,
        dataset_schema={k:v for k,v in schema.items() if not k.startswith('_')},audit_config={k:v for k,v in cfg.items() if not k.startswith('_')},
        source_scope=cfg['source']['scope'],source_population_rows=checked['n_rows'],evaluation_sample={'strategy':cfg['sample']['strategy'],'requested_rows':len(positions),'common_valid_rows':len(frame),'seed':cfg['seed'],'estimand':'Uniform file-row sample before validity filtering; common valid cohort thereafter'},
        alignment={'identity':source_info['identity_validation'],'excluded_rows':len(excluded),'records':'asset_alignment.csv','evaluation_positions':'evaluation_sample.csv'},
        weights='unweighted',versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','pyarrow','jsonschema']},python=platform.python_version(),
        supplied_graph_contracts=schema.get('neighbors',[]),resource_estimates={k:v for k,v in checked.items() if k.startswith('estimated_')},
        full_vector_validation='Only retained evaluation vectors validated; external matrix headers and all source IDs checked',
        correction={'method':cfg['nulls']['correction'],'family':cfg['nulls']['testing_family']},
        limitations=['Expensive analyses are bounded sampled in-memory experiments. Source counts/moments stream; quantiles may be reservoir estimates.',
            'Uniform row sampling does not balance groups; supplied groups prevent train/test leakage and optionally bootstrap groups.',
            'External ID lookup is population-sized and explicitly capped. Source identity uniqueness uses temporary disk SQLite.',
            'One external prediction pipeline per target per run. No automatic ANN backend or encoder retraining.',
            'Repeated topology samples give descriptive variability; conditional graph bootstraps do not account for retraining or search errors.'])
    write_json(manifest,output/'manifest.json')
    write_json({'dataset_schema':schema,'audit_config':cfg},output/'resolved_configuration.json')
    # Refresh the expandable manifest after adding provenance.
    result['manifest']=manifest
    doc=(output/'report.html').read_text();start='<details><summary>Run settings and limitations</summary><pre>'
    import html
    beginning=doc.find(start)
    if beginning>=0:
        end=doc.find('</pre>',beginning);doc=doc[:beginning+len(start)]+html.escape(json.dumps(manifest,indent=2))+doc[end:]
    (output/'report.html').write_text(doc)
    if progress: progress(f'Complete: {output / "report.html"}')
    return result
