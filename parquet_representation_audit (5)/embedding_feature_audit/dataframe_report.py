"""DataFrame-first audit orchestration; see individual functions for the math."""
from pathlib import Path
import json
import warnings
import uuid
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import false_discovery_control
from .dataframe import (default_features,add_derived_features,embedding_matrix,
                        align_predictions,encode_feature)
from .geometry import make_neighbors,validate_neighbors,distance_relationship
from .statistics import feature_source,score_geometry,permutation_analysis,prediction_metrics,geometry_delta
from .modeling import run_probes, validate_probe_models
from .experiments import missingness_experiments
from .visuals import feature_plots,write_report
from .cluster_metadata import configure_cluster_metadata,annotate_labels,metadata_label_table,cluster_id,metadata_key
from .explanations import explain_feature
from .exploration import build_explorer_data, copy_explorer_assets
from .comparison import align_embedding_rows, compare_graphs
from .feature_overview import feature_overview, overview_settings
from .normalization import l2_normalize, safe_norms, normalization_options, normalization_probes
from .topology import topology_options, audit_topology
from .enhancement_report import enhancement_plots


def write_json(value,path):
    """Standards-compliant JSON: missing/undefined numbers become null."""
    def clean(v):
        if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)): return [clean(x) for x in v]
        if isinstance(v,Path): return str(v)
        if isinstance(v,np.generic): v=v.item()
        if isinstance(v,float) and not np.isfinite(v): return None
        return v
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False),encoding='utf-8')


def save_table(table,path):
    if len(table.columns):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            table.to_csv(path,index=False)


def selected(name,option):
    return option is True or isinstance(option,list) and name in option


def audit_dataframe(sample_df,out_dir='dataframe_audit',*,features=None,config=None,
                    embeddings=None,predictions=None,neighbors=None,predictor=None,
                    comparison_embeddings=None,comparison_ids=None,comparison_neighbors=None):
    """Audit aligned embeddings and plain columns. Return a dict of pandas tables.

    features: optional {column: {kind: categorical|continuous|multilabel, ...}}.
    predictions: separate DataFrame aligned by config['id_column'], or prediction
                 columns in sample_df named in config['predictions'].
    neighbors: optional N×K row-position graph, excluding self, matching this df.
    predictor: optional (z, y, train_rows, hidden_rows, kind) -> hidden predictions.
    comparison_embeddings: optional second N×D matrix (D may differ). Supply
                 comparison_ids to align by ID; otherwise row order must match.
    comparison_neighbors: optional graph in the exact final common row order.
    config['drop_invalid_embeddings']: explicitly drop nonfinite/zero cosine
                 vectors from either model. Supplied graphs cannot be filtered.
    """
    cfg=dict(id_column='id',embedding_column='embedding',metric='cosine',ks=[5,10,25,50,100],
        permutations=20,null_modes=['permutation'],seed=42,probe_features=True,probe_models=['baseline','linear'],
        max_probe_labels=10,trees=100,missingness_features=['rating','artist_avg_fav_count_cluster'],
        missingness_fractions=[.1,.25,.5,.75],missingness_modes=['random','rare-first'],
        predictions={},population_prevalence={},pca_overview=True,verbose=True,
        cluster_metadata_dir=None,cluster_metadata={},explorer=True,explorer_max_rows=10000,
        explorer_max_bytes=80000000,comparison_bootstraps=200,drop_invalid_embeddings=False,
        embedding_name='Embedding A',comparison_name='Embedding B',cohorts={},overview={},
        embedding_normalization='none',normalization_analysis={},topology={})
    cfg.update(config or {})
    norm_options=normalization_options(cfg);topo_options=topology_options(cfg)
    if cfg['embedding_normalization'] not in ['none','l2']:
        raise ValueError('embedding_normalization must be none or l2')
    if cfg['embedding_normalization']=='l2' and cfg['metric']=='euclidean' and (neighbors is not None or comparison_neighbors is not None) and not cfg.get('supplied_graph_contract_validated'):
        raise ValueError('L2 normalization changes Euclidean neighbors; rebuild graphs from normalized arrays before supplying them')
    if topo_options['enabled'] and topo_options['null_mode']=='stratified' and not cfg.get('strata_column'):
        raise ValueError('Stratified topology nulls require strata_column')
    validate_probe_models(cfg['probe_models'])
    if comparison_embeddings is None and (comparison_ids is not None or comparison_neighbors is not None):
        raise ValueError('Comparison IDs/graph require comparison_embeddings')
    id_col=cfg['id_column']
    if id_col not in sample_df or sample_df[id_col].isna().any() or sample_df[id_col].duplicated().any():
        raise ValueError('DataFrame must have a unique, nonmissing ID column')
    if not isinstance(cfg['permutations'],int) or cfg['permutations']<0:
        raise ValueError('permutations must be a nonnegative integer')
    if cfg['metric'] not in ['cosine','euclidean']:
        raise ValueError('Supported metrics are cosine and euclidean')
    df=sample_df.copy(deep=False) if cfg.get('generic_schema') else add_derived_features(sample_df)
    original_ids=df[id_col].astype(str).to_numpy(dtype=str)
    if len(set(original_ids)) != len(original_ids):
        raise ValueError('Post IDs collide after string normalization')
    z=embedding_matrix(df,cfg['embedding_column'],embeddings,allow_invalid=True)
    zb=align_embedding_rows(comparison_embeddings,comparison_ids,original_ids) if comparison_embeddings is not None else None
    good=np.isfinite(z).all(axis=1)
    direction_required=cfg['metric']=='cosine' or cfg['embedding_normalization']=='l2' or norm_options['enabled'] and bool(set(norm_options['variants']) & {'unit','unit_standardized'}) or topo_options['enabled'] and 'unit_chordal' in topo_options['spaces']
    if direction_required: good &= safe_norms(z)>0
    if zb is not None:
        good &= np.isfinite(zb).all(axis=1)
        if direction_required: good &= safe_norms(zb)>0
    excluded_ids=original_ids[~good]
    if not good.all():
        if not cfg['drop_invalid_embeddings']:
            raise ValueError(f'{len(excluded_ids):,} rows have nonfinite embeddings or zero vectors required by directional analyses in an evaluated model. '
                             'Set drop_invalid_embeddings=true or use --drop-invalid-embeddings to evaluate only valid rows.')
        if neighbors is not None or comparison_neighbors is not None:
            raise ValueError('Cannot drop rows while using supplied graphs; filter embeddings and rebuild/reindex both graphs first')
        if good.sum()<2:
            raise ValueError('Need at least two common valid embedding rows')
        df=df.iloc[np.flatnonzero(good)].reset_index(drop=True)
        z=z[good]
        if zb is not None: zb=zb[good]
        if cfg['verbose']: print(f'Excluded {len(excluded_ids):,} invalid rows; evaluating {len(df):,} common valid rows.',flush=True)
    raw_z=z;raw_zb=zb
    if cfg['embedding_normalization']=='l2':
        z=l2_normalize(z)
        if zb is not None: zb=l2_normalize(zb)
    for option in ['explorer_max_rows','explorer_max_bytes','comparison_bootstraps']:
        if not isinstance(cfg[option],int) or cfg[option] < (0 if option=='comparison_bootstraps' else 1):
            raise ValueError(f'{option} must be an integer in the supported range')
    specs=features if features is not None else cfg.get('features')
    if specs is None:
        specs=default_features(df)
        for c in ['megapixels','aspect_ratio']+[c for c in df if c.endswith('_year')]:
            if c in df:
                specs[c]={'kind':'continuous'}
    specs={name:({'kind':spec} if isinstance(spec,str) else dict(spec)) for name,spec in specs.items()}
    missing_cols=set(specs)-set(df)
    if missing_cols:
        raise ValueError(f'Configured feature columns are missing: {sorted(missing_cols)}')
    specs,metadata=configure_cluster_metadata(specs,cfg)
    ks=sorted(set(int(k) for k in cfg['ks'] if 0<int(k)<len(df)))
    if not ks:
        raise ValueError('At least one k must be between 1 and N-1')
    overview_settings(ks,cfg)
    out=Path(out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'Output directory is not empty: {out}. Choose a fresh folder to avoid mixing runs.')
    out.mkdir(parents=True,exist_ok=True)
    ids=df[id_col].astype(str).to_numpy(dtype=str)
    if len(set(ids)) != len(ids):
        raise ValueError('Post IDs collide after string normalization')
    np.save(out/'row_ids.npy',ids,allow_pickle=False)
    norm_table=pd.DataFrame({'row_position':np.arange(len(df)),id_col:ids,'embedding_a_norm':safe_norms(raw_z)})
    if raw_zb is not None: norm_table['embedding_b_norm']=safe_norms(raw_zb)
    save_table(norm_table,out/'embedding_norms.csv')
    pd.DataFrame({id_col:excluded_ids,'reason':'invalid embedding in at least one evaluated model'}).to_csv(out/'excluded_embedding_rows.csv',index=False)
    pd.DataFrame({'row_position':np.arange(len(df)),id_col:df[id_col].to_numpy()}).to_csv(out/'row_alignment.csv',index=False)
    if neighbors is None:
        if cfg['verbose']: print(f'Building one {cfg["metric"]} neighbor graph for {len(df):,} rows…',flush=True)
        graph,distances=make_neighbors(z,max(ks),cfg['metric'])
        graph_source='exact brute-force search on this sample'
    else:
        graph=validate_neighbors(neighbors,len(df),max(ks)); distances=None
        graph_source='supplied row-position graph; search fidelity is caller responsibility'
    np.save(out/'neighbors.npy',graph,allow_pickle=False)
    comparison=None
    if zb is not None:
        if cfg['verbose']: print(f'Building comparison graph for {cfg["comparison_name"]} on the same {len(df):,} rows…',flush=True)
        if comparison_neighbors is None:
            graph_b,distances_b=make_neighbors(zb,max(ks),cfg['metric'])
            comparison_source='exact brute-force search on the same common cohort'
        else:
            from .exploration import edge_distances
            graph_b=validate_neighbors(comparison_neighbors,len(df),max(ks))
            distances_b=edge_distances(zb,graph_b,cfg['metric'])
            comparison_source='supplied row-position graph; search fidelity is caller responsibility'
        np.save(out/'comparison_neighbors.npy',graph_b,allow_pickle=False)
        comparison=dict(graph=graph_b,distances=distances_b)
    groups=None
    if cfg.get('group_column'):
        raw=df[cfg['group_column']].to_numpy()
        # Missing groups are separate items, not one giant missing-value group.
        groups=np.array([f'row:{i}' if pd.isna(v) else 'group:'+str(v) for i,v in enumerate(raw)])
    split=df[cfg['split_column']].astype(str).to_numpy() if cfg.get('split_column') else None
    strata=df[cfg['strata_column']].astype('string').fillna('__missing__').to_numpy(dtype=str) if cfg.get('strata_column') else None
    priority=pd.to_numeric(df[cfg['missingness_priority_column']],errors='coerce').fillna(np.inf).to_numpy() if cfg.get('missingness_priority_column') else None
    if cfg.get('predictions') and predictions is not None:
        pred_df=align_predictions(df,predictions,id_col)
    else:
        pred_df=df
    coords=coords_b=None
    if cfg['pca_overview']:
        if min(z.shape)>=2:
            coords=PCA(n_components=2,svd_solver='randomized',random_state=cfg['seed']).fit_transform(z)
            pd.DataFrame({'row_position':np.arange(len(df)),id_col:df[id_col].to_numpy(),
                          'pc1':coords[:,0],'pc2':coords[:,1]}).to_csv(out/'pca_coordinates.csv',index=False)
        if zb is not None and min(zb.shape)>=2:
            coords_b=PCA(n_components=2,svd_solver='randomized',random_state=cfg['seed']).fit_transform(zb)
            pd.DataFrame({'row_position':np.arange(len(df)),id_col:df[id_col].to_numpy(),
                          'pc1':coords_b[:,0],'pc2':coords_b[:,1]}).to_csv(out/'comparison_pca_coordinates.csv',index=False)
    manifest=dict(run_id=uuid.uuid4().hex,n_rows=len(df),n_input_rows=len(sample_df),n_excluded_embedding_rows=len(excluded_ids),
        embedding_dimensions=z.shape[1],embedding_memory_mib=z.nbytes/2**20,
        graph_source=graph_source,requested_ks=cfg['ks'],evaluated_ks=ks,config=cfg,features=specs,
        metadata_features=list(metadata),
        reference_scope='Supplied sample; external prevalence only where explicitly configured',
        embedding_preprocessing=cfg['embedding_normalization'],
        missingness_predictor='user callback' if predictor is not None else 'embedding-based kNN demo; not independent evidence',
        neighborhood_missingness='fixed full-sample neighborhoods; purity denominator uses known neighbors only',
        limitations=['Exact graph construction costs approximately O(N²D); permutations reuse it.',
            'Probe scores use one fixed held-out split. Macro tag scores exclude unsupported training tags.',
            'Tag absence is only a negative among rows configured as annotated; incomplete annotation can bias it.',
            'No prediction accuracy is claimed for naturally missing labels without observed truth.',
            'Permutation p-values are empirical; z-scores are descriptive, not a normality guarantee.'])
    if comparison is not None:
        manifest['comparison']=dict(name=cfg['comparison_name'],dimensions=zb.shape[1],graph_source=comparison_source,
            scope='Same common valid cohort, candidate pool, annotations, metric and k; per-anchor paired geometry.',
            uncertainty='Percentile bootstrap conditional on fixed graphs; group_column controls group resampling. '
                        'Shared neighborhoods may induce dependence; no retraining or search-error uncertainty.')
    write_json(manifest,out/'manifest.json')
    results={}; overview=[]; combined_null=[]; encoded={}
    for number,(name,spec) in enumerate(specs.items()):
        if cfg['verbose']: print(f'[{number+1}/{len(specs)}] {name}',flush=True)
        # Stable numeric folders avoid path traversal or nonportable column names.
        folder=out/'features'/f'{number+1:03d}'
        tables=folder/'tables'; tables.mkdir(parents=True,exist_ok=True)
        plots=folder/'plots'; plots.mkdir(exist_ok=True)
        kind=spec['kind']
        label_metadata=metadata.get(name)
        results[name]=dict(folder=folder,kind=kind,status='ok',metadata=label_metadata)
        def save_feature_table(table,path):
            save_table(annotate_labels(table,label_metadata),path)
        if label_metadata:
            write_json(label_metadata,folder/'cluster_metadata.json')
            save_table(metadata_label_table(df[name],label_metadata,spec),tables/'labels_metadata.csv')
        y,valid,names=encode_feature(df[name],spec)
        encoded[name]=dict(values=y,valid=valid,names=names,kind=kind)
        source,prevalence,co=feature_source(df[name],y,valid,names,kind,spec)
        results[name]['source']=source
        write_json(source,folder/'source_summary.json'); write_json(dict(column=name,kind=kind,spec=spec),folder/'feature.json')
        prevalence=annotate_labels(prevalence,label_metadata)
        if label_metadata:
            unknown=sorted(set(y[valid])-set(metadata_key(row['id'],label_metadata) for row in label_metadata['labels']))
            source['unmapped_label_ids']=unknown
            source['missing_values']=spec.get('missing_values',[])
            write_json(source,folder/'source_summary.json')
        save_feature_table(prevalence,tables/'source_prevalence.csv'); save_feature_table(co,tables/'cooccurrence.csv')
        if name in cfg.get('source_only_targets',[]):
            results[name].update(status='source_only',probe_status='Geometry and fitting disabled for this feature')
            overview.append(dict(feature=name,kind=kind,status='source_only',coverage=source['coverage']))
            continue
        if valid.sum()<2 or kind!='continuous' and not names:
            results[name].update(status='skipped',error='Not enough known rows or no labels meet the configured support threshold')
            overview.append(dict(feature=name,kind=kind,status='skipped',coverage=source['coverage']))
            continue
        baseline=cfg['population_prevalence'].get(name)
        if baseline is not None:
            baseline={(cluster_id(k) if label_metadata else str(k)):float(v) for k,v in baseline.items()}
            if set(names)-set(baseline) or any(not 0<v<=1 for v in baseline.values()):
                raise ValueError(f'{name}: external prevalence must cover every audited label with values in (0,1]')
        summary,per_label=score_geometry(y,valid,names,kind,graph,ks,baseline)
        if summary.empty:
            results[name].update(status='skipped',error='No anchors have known neighbors at requested k')
            continue
        per_label=annotate_labels(per_label,label_metadata)
        save_feature_table(summary,tables/'geometry.csv'); save_feature_table(per_label,tables/'per_label_geometry.csv')
        if comparison is not None:
            b_summary,b_labels=score_geometry(y,valid,names,kind,comparison['graph'],ks,baseline)
            results[name]['comparison_geometry']=b_summary
            save_feature_table(b_summary,tables/'comparison_geometry.csv')
            save_feature_table(b_labels,tables/'comparison_per_label_geometry.csv')
        null_score=pd.DataFrame()
        for mode in cfg['null_modes']:
            if mode not in ['permutation','prevalence-random','stratified']:
                raise ValueError(f'Unknown null mode {mode}')
            if mode=='stratified' and strata is None:
                raise ValueError('stratified null requires strata_column')
            if cfg['permutations']:
                nulls,scored=permutation_analysis(y,valid,names,kind,graph,ks,summary,
                    cfg['permutations'],cfg['seed'],mode,strata if mode=='stratified' else None,baseline)
                save_table(nulls,tables/f'null_{mode}.csv'); save_table(scored,tables/f'null_comparison_{mode}.csv')
                if mode=='permutation' or null_score.empty:
                    null_score=scored; save_table(scored,tables/'null_comparison.csv')
                scored['feature']=name; scored['null_mode']=mode; combined_null.append(scored)
        if kind=='continuous':
            relationship=distance_relationship(z,y,valid,cfg['metric'],cfg['seed'],cap=cfg.get('distance_cap',500),pairs=cfg.get('distance_pairs',20000))
            write_json(relationship,folder/'distance_relationship.json')
        if strata is not None:
            conditional=[]; conditional_prevalence=[]
            for group in pd.unique(strata):
                rows=np.flatnonzero(strata==group)
                group_source,group_counts,_=feature_source(df[name].iloc[rows],y[rows],valid[rows],names,kind,spec)
                conditional.append(dict(group=group,**group_source))
                if len(group_counts):
                    group_counts['group']=group; conditional_prevalence.append(group_counts)
            save_table(pd.DataFrame(conditional),tables/'conditional_source_summary.csv')
            if conditional_prevalence:
                save_feature_table(pd.concat(conditional_prevalence,ignore_index=True),tables/'conditional_prevalence.csv')
        probes=pd.DataFrame()
        if selected(name,cfg['probe_features']):
            if cfg['verbose']: print(f'  {name}: training held-out probes…',flush=True)
            py,pnames=y,names
            if kind=='multilabel':
                pnames=names[:cfg['max_probe_labels']]; py=y[:,:len(pnames)]
            try:
                models=list(dict.fromkeys(['baseline']+list(cfg['probe_models'])))
                probes,detail,pred,train,test=run_probes(z,py,valid,pnames,kind,cfg['seed'],groups,split,cfg['trees'],models,regularization_grid=cfg.get('probe_regularization_grid'))
                results[name].update(probe_predictions=pred,probe_labels=pnames)
                save_feature_table(probes,tables/'probes.csv'); save_feature_table(detail,tables/'probe_per_label.csv')
                save_feature_table(pred,tables/'probe_predictions.csv')
                save_table(pd.DataFrame({'row_position':np.r_[train,test],'split':['train']*len(train)+['test']*len(test)}),tables/'probe_split.csv')
                write_json(dict(labels=pnames,selection='first selected source labels; full geometry vocabulary may be larger'),folder/'probe_scope.json')
                if zb is not None:
                    b_probes,b_detail,b_pred,b_train,b_test=run_probes(zb,py,valid,pnames,kind,cfg['seed'],groups,split,cfg['trees'],models,regularization_grid=cfg.get('probe_regularization_grid'))
                    if not np.array_equal(train,b_train) or not np.array_equal(test,b_test):
                        raise ValueError('Comparison probes must share the identical split')
                    results[name].update(comparison_probes=b_probes,comparison_probe_predictions=b_pred)
                    save_feature_table(b_probes,tables/'comparison_probes.csv')
                    save_feature_table(b_detail,tables/'comparison_probe_per_label.csv')
                    save_feature_table(b_pred,tables/'comparison_probe_predictions.csv')
            except ValueError as exc:
                results[name]['probe_status']=str(exc)
                write_json({'status':'skipped','reason':str(exc)},folder/'probe_status.json')
        pred_geometry=None
        if name in cfg['predictions']:
            mapping=cfg['predictions'][name]
            mapping={'column':mapping} if isinstance(mapping,str) else mapping
            col=mapping['column']
            if col not in pred_df:
                raise ValueError(f'Missing predicted column {col}')
            results[name]['prediction_series']=pred_df[col]
            # Empty prediction means a known negative tag set; NA means unavailable.
            pred_spec={**spec,'empty_is_missing':False,'missing_values':mapping.get('missing_values',spec.get('missing_values',[]))}
            yp,pvalid,pnames=encode_feature(pred_df[col],pred_spec,names if kind=='multilabel' else None)
            joint=valid & pvalid
            accuracy,detail=prediction_metrics(y[joint],yp[joint],kind,names)
            accuracy.update(observed_coverage=float(valid.mean()),prediction_coverage=float(pvalid.mean()),
                            predictor_inputs=mapping.get('uses_embedding','unknown'),
                            evaluation_scope='matched observed and predicted rows; selected tag vocabulary only' if kind=='multilabel' else 'matched observed and predicted rows')
            if kind=='multilabel':
                from .dataframe import tag_sets
                pred_sets,_=tag_sets(pred_df[col],False)
                extra=set().union(*(pred_sets[i] for i in np.flatnonzero(joint)))-set(names)
                accuracy['predicted_tags_outside_audited_vocabulary']=len(extra)
            write_json(accuracy,folder/'prediction_metrics.json'); save_feature_table(detail,tables/'prediction_per_label.csv')
            results[name]['prediction_metrics']=accuracy
            if joint.sum()>=2:
                compare_names=names if kind!='categorical' else sorted(set(y[joint])|set(yp[joint]))
                true_geom,_=score_geometry(y,joint,compare_names,kind,graph,ks)
                predicted_geom,pred_label=score_geometry(yp,joint,compare_names,kind,graph,ks)
                if len(true_geom) and len(predicted_geom):
                    pred_geometry=geometry_delta(true_geom,predicted_geom,kind)
                    save_table(pred_geometry,tables/'prediction_geometry.csv'); save_feature_table(pred_label,tables/'predicted_per_label_geometry.csv')
            # Predicted distribution across missing truth is reported separately, not accuracy.
            predicted_source,pred_prevalence,_=feature_source(pred_df[col],yp,pvalid,pnames,kind,pred_spec)
            write_json(predicted_source,folder/'predicted_source_summary.json'); save_feature_table(pred_prevalence,tables/'predicted_source_prevalence.csv')
        missing_summary=pd.DataFrame(); missing_geometry=pd.DataFrame()
        if selected(name,cfg['missingness_features']) and cfg['missingness_fractions']:
            if cfg['verbose']: print(f'  {name}: running missingness experiments…',flush=True)
            missing_summary,missing_geometry=missingness_experiments(z,y,valid,names,kind,graph,ks,
                cfg['missingness_fractions'],cfg['missingness_modes'],cfg['seed'],predictor,priority,groups)
            save_table(missing_summary,tables/'missingness.csv'); save_table(missing_geometry,tables/'missingness_geometry.csv')
        feature_plots(plots,prevalence,summary,per_label,null_score,probes,missing_summary,missing_geometry,
                      y,valid,kind,coords,names,pred_geometry,co,label_metadata)
        row=summary.iloc[(summary.k-50).abs().argmin()].to_dict()
        row.update(feature=name,kind=kind,status='ok',coverage=source['coverage'],n_source_labels=source.get('n_labels'))
        overview.append(row)
        results[name].update(source=source,geometry=summary,per_label=per_label,probes=probes,null_comparison=null_score,
                             missingness=missing_summary)
    overview=pd.DataFrame(overview)
    enhancements=dict(normalization={'status':'disabled'},topology={'status':'disabled'})
    if norm_options['enabled']:
        norm_scores=[];norm_predictions=[];norm_splits=[];norm_status={}
        for name,data in encoded.items():
            if not selected(name,norm_options['features']): continue
            if cfg['verbose']: print(f'{name}: comparing raw, unit and magnitude-only probe inputs…',flush=True)
            for model,matrix in [('A',raw_z),('B',raw_zb)]:
                if matrix is None: continue
                status,scores,predicted,split_rows=normalization_probes(matrix,data,norm_options,seed=cfg['seed'],groups=groups,split=split,max_labels=cfg['max_probe_labels'])
                norm_status[f'{name}:{model}']=status
                for table,collection in [(scores,norm_scores),(predicted,norm_predictions),(split_rows,norm_splits)]:
                    if len(table):
                        table['feature']=name;table['embedding']=model;collection.append(table)
        scores=pd.concat(norm_scores,ignore_index=True) if norm_scores else pd.DataFrame()
        if len(scores) and (scores.groupby('feature').split_signature.nunique()>1).any():
            raise RuntimeError('Normalization A/B comparisons changed held-out rows')
        predicted=pd.concat(norm_predictions,ignore_index=True) if norm_predictions else pd.DataFrame()
        splits_table=pd.concat(norm_splits,ignore_index=True) if norm_splits else pd.DataFrame()
        save_table(scores,out/'normalization_probe_scores.csv');save_table(predicted,out/'normalization_probe_predictions.csv');save_table(splits_table,out/'normalization_probe_splits.csv')
        for name,result in results.items():
            result['normalization_scores']=scores[scores.feature==name] if len(scores) else pd.DataFrame()
        enhancements['normalization']=dict(status='ok' if len(scores) else 'skipped',options=norm_options,
            scores=scores.to_dict('records'),feature_status=norm_status,
            scope='Same evaluated rows and train/test split per feature across input variants; source norms measured before optional main normalization.',
            interpretation='Direction-only gains or losses describe the selected probe pipeline; fit telemetry does not prove general training stability.')
    if norm_options['enabled'] and enhancements['normalization']['status']=='skipped':
        reasons=sorted(set(value.get('reason','No estimable variants; inspect errors.') for value in norm_status.values()))
        enhancements['normalization']['reason']='; '.join(reasons) or 'No configured features selected for normalization analysis.'
    if topo_options['enabled']:
        if cfg['verbose']: print('Calculating sampled component filtrations and bounded H1 persistence…',flush=True)
        models={'A':raw_z}
        if raw_zb is not None: models['B']=raw_zb
        progress=(lambda text: print(text,flush=True)) if cfg['verbose'] else None
        status,summary,trials,structure,bars,merges,samples=audit_topology(models,encoded,ids,topo_options,seed=cfg['seed'],strata=strata,progress=progress)
        for table,file in [(summary,'topology_feature_summary.csv'),(trials,'topology_feature_trials.csv'),(structure,'topology_structure.csv'),(bars,'topology_barcodes.csv'),(merges,'topology_merges.csv'),(samples,'topology_sample_rows.csv')]:save_table(table,out/file)
        for name,result in results.items():
            result['topology_summary']=summary[summary.feature==name] if len(summary) else pd.DataFrame()
        enhancements['topology']=dict(**status,features=summary.to_dict('records'),structure=structure.to_dict('records'),barcodes=bars.to_dict('records'))
    enhancement_plots(out,enhancements,embedding_names={'A':cfg['embedding_name'],'B':cfg['comparison_name']})
    write_json(enhancements,out/'enhancement_measurements.json')
    first=['feature','kind','status','coverage','k','macro_lift','micro_lift','macro_purity','micro_purity']
    overview=overview[[c for c in first if c in overview]+[c for c in overview if c not in first]]
    save_table(overview,out/'feature_overview.csv')
    if combined_null:
        all_null=pd.concat(combined_null,ignore_index=True)
        for col in all_null:
            if col.endswith('_p') and cfg.get('null_correction','bh')=='bh':
                mask=all_null[col].notna()
                all_null.loc[mask,col[:-2]+'_q_bh']=false_discovery_control(all_null.loc[mask,col].to_numpy(),method='bh')
        save_table(all_null,out/'all_null_comparisons.csv')
    continuous=[name for name,spec in specs.items() if spec['kind']=='continuous']
    if continuous:
        numeric=df[continuous].apply(pd.to_numeric,errors='coerce')
        numeric.corr(method='spearman').to_csv(out/'source_numeric_spearman.csv')
    manifest['feature_status']={name:result['status'] for name,result in results.items()}
    manifest['feature_folders']={name:str(result['folder'].relative_to(out)) for name,result in results.items()}
    if comparison is not None:
        if cfg['verbose']: print('Calculating paired embedding changes and conditional bootstrap intervals…',flush=True)
        comparison['paired']=compare_graphs({n:r for n,r in encoded.items() if n not in cfg.get('source_only_targets',[])},graph,comparison['graph'],ks,groups,cfg['comparison_bootstraps'],cfg['seed'])
        save_table(comparison['paired'],out/'embedding_comparison.csv')
        for name,result in results.items():
            result['paired_comparison']=comparison['paired'][comparison['paired'].feature==name]
    for name,result in results.items():
        result['explanation']=explain_feature(name,result,manifest)
        write_json(result['explanation'],result['folder']/'explanation.json')
    explorer=build_explorer_data(df,z,graph,distances,specs,results,{**cfg,'_evaluated_ks':ks,'_run_id':manifest['run_id'],
                                 '_input_rows':len(sample_df),'_excluded_rows':len(excluded_ids)},coords,comparison)
    if explorer.get('available') and comparison is not None and coords_b is not None:
        explorer['comparison']['coordinates']=np.round(coords_b,6).tolist()
    write_json(explorer,out/'explorer_data.json')
    if (out/'explorer_data.json').stat().st_size>cfg['explorer_max_bytes']:
        explorer={'available':False,'reason':'Explorer snapshot exceeds explorer_max_bytes. Aggregate scorecards remain available; use a smaller explicit sample.'}
        write_json(explorer,out/'explorer_data.json')
    if cfg['verbose']: print('Summarizing feature diversity across all neighborhood sizes…',flush=True)
    feature_view,entropy_tables=feature_overview(encoded,results,graph,ks,cfg,comparison,explorer['available'])
    for name,table in entropy_tables.items():
        save_table(table,results[name]['folder']/'tables'/'local_entropy.csv')
    write_json(feature_view,out/'feature_measurements.json')
    explorer['overview']=feature_view
    write_json(explorer,out/'explorer_data.json')
    if explorer['available'] and (out/'explorer_data.json').stat().st_size>cfg['explorer_max_bytes']:
        explorer={'available':False,'reason':'Row snapshot with feature measurements exceeds explorer_max_bytes; aggregate overview remains available.'}
        for record in feature_view['features'].values():
            for key in ['entropy_rows','comparison_entropy_rows','probe_predictions','comparison_probe_predictions']:
                record.pop(key,None)
        explorer['overview']=feature_view
        write_json(explorer,out/'explorer_data.json'); write_json(feature_view,out/'feature_measurements.json')
    explorer['enhancements']=enhancements
    write_json(explorer,out/'explorer_data.json')
    if explorer['available'] and (out/'explorer_data.json').stat().st_size>cfg['explorer_max_bytes']:
        for record in feature_view['features'].values():
            for key in ['entropy_rows','comparison_entropy_rows','probe_predictions','comparison_probe_predictions']: record.pop(key,None)
        explorer={'available':False,'reason':'Complete snapshot including enhancement measurements exceeds explorer_max_bytes; aggregate analyses remain available.',
                  'overview':feature_view,'enhancements':enhancements}
        write_json(explorer,out/'explorer_data.json');write_json(feature_view,out/'feature_measurements.json')
    manifest['enhancements']={key:dict(status=value['status'],scope=value.get('scope',value.get('sample_scope'))) for key,value in enhancements.items()}
    manifest['explorer']=dict(available=explorer['available'],max_rows=cfg['explorer_max_rows'],
                              reason=explorer.get('reason'),scope='Full evaluated sample; no hidden row subsampling')
    copy_explorer_assets(out)
    write_json(manifest,out/'manifest.json'); write_report(out,json.loads((out/'manifest.json').read_text()),overview,results)
    return dict(out_dir=out,report=out/'report.html',overview=overview,features=results,neighbors=graph,
                comparison=comparison,explorer_data=out/'explorer_data.json',enhancements=enhancements)
