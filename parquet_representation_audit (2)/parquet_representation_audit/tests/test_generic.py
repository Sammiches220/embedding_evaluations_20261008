"""Contract tests use independent expected values and real Parquet/CLI paths."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from parquet_audit import validate_bundle,run_audit
from parquet_audit.config import read_metadata
from parquet_audit.ingest import canonical_identity,external_vectors,supplied_graph,source_statistics
from parquet_audit.semantics import normalize_series,normalize_value,parse_tags,typed_key
from embedding_feature_audit.geometry import purity_lift,continuous_geometry,distance_relationship,make_neighbors
from embedding_feature_audit.normalization import safe_norms,l2_normalize

ROOT=Path(__file__).resolve().parents[1]


def bundle(tmp,domain='telemetry',*,small=True):
    src=ROOT/'examples'/domain;schema=json.loads((src/'dataset_schema.json').read_text());cfg=json.loads((src/'audit_config.json').read_text())
    schema['dataset']['path']=str(src/schema['dataset']['path'])
    for spec in schema['columns'].values():
        if 'metadata_path' in spec: spec['metadata_path']=str((src/spec['metadata_path']).resolve())
    for e in schema['embeddings']:
        for key in ['matrix_path','ids_path']:
            if key in e: e[key]=str(src/e[key])
    cfg['verbose']=False
    if small:
        cfg['sample']['max_rows']=36;cfg['geometry']['ks']=[2,5];cfg['report']['overview']['trusted_ks']=[2,5]
        cfg['nulls']['permutations']=0;cfg['normalization']['enabled']=False;cfg['topology']['enabled']=False;cfg['missingness']['enabled']=False
        cfg['comparison']['bootstraps']=0
    def save():
        a=tmp/'schema.json';b=tmp/'config.json';a.write_text(json.dumps(schema));b.write_text(json.dumps(cfg));return a,b
    return schema,cfg,save


def test_counts_and_clusters_are_semantically_distinct_and_missing_first():
    count={'kind':'count','missing_values':[-1]};cluster={'kind':'cluster_label','id_type':'integer'}
    assert normalize_series(pd.Series([-1,0,3],name='counts'),count).tolist()==[None,0,3]
    with pytest.raises(ValueError,match='nonnegative'):normalize_value(-2,count)
    assert normalize_series(pd.Series([0,2,-1]),cluster).tolist()==['0','2','-1']
    assert normalize_value(False,{'kind':'boolean'})=='false'
    assert normalize_value(0,{'kind':'numeric'})==0
    with pytest.raises(ValueError):normalize_value(1,{'kind':'boolean'})


def test_unknown_known_empty_and_literal_delimiters():
    spec={'kind':'multilabel','encoding':'list','null_is_unknown':True,'empty_is_unknown':False}
    assert parse_tags(None,spec) is None and parse_tags([],spec)==[]
    assert parse_tags(['A','A','b c'],spec)==['A','b c']
    assert parse_tags('x | y',{'encoding':'delimited','delimiter':'|','strip':True})==['x','y']
    assert parse_tags('["x y","z"]',{'encoding':'json'})==['x y','z']
    with pytest.raises(ValueError):parse_tags('[1]',{'encoding':'json'})
    with pytest.raises(ValueError):parse_tags('x||y',{'encoding':'delimited','delimiter':'|'})
    assert parse_tags('',{'encoding':'delimited','delimiter':'|','empty_is_unknown':False})==[]


def test_exact_identity_large_int_leading_zeros_composites_and_float_rejection():
    big=9007199254741099
    assert canonical_identity([big],[{'id_type':'integer'}])==str(big)
    assert canonical_identity(['0001'],[{'id_type':'string'}])=='0001'
    assert canonical_identity(['0001',2],[{},{}])!='00012'
    with pytest.raises(ValueError):canonical_identity([float(big)],[{}])
    with pytest.raises(ValueError):typed_key('1','integer')
    with pytest.raises(ValueError):typed_key(True,'integer')


def test_metadata_typed_duplicate_names_remain_distinct_and_mismatch_fails(tmp_path):
    meta={'schema_version':1,'name':'code','id_type':'string','excluded_from_coverage':[],
          'labels':[{'id':'01','name':'same'},{'id':'1','name':'same'},{'id':'a <script>','name':'odd'}]}
    p=tmp_path/'metadata.json';p.write_text(json.dumps(meta));assert len(read_metadata(p)['labels'])==3
    meta['labels'].append({'id':'1','name':'different'});p.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='duplicate'):read_metadata(p)
    meta['labels']=meta['labels'][:3];meta['excluded_from_coverage']=['1'];p.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='disagree'):read_metadata(p,{'kind':'categorical','id_type':'string','missing_values':[]})
    assert read_metadata(p,{'kind':'categorical','id_type':'string','missing_values':[],'metadata_missing_override':'Schema keeps code 1 known'})


def test_external_id_shuffle_and_absent_policy(tmp_path):
    np.save(tmp_path/'vectors.npy',np.array([[30.,3],[10.,1],[20.,2]]));np.save(tmp_path/'ids.npy',np.array(['003','001','002']))
    asset={'name':'test','matrix_path':str(tmp_path/'vectors.npy'),'ids_path':str(tmp_path/'ids.npy')}
    cfg={'limits':{'external_batch_rows':1}}
    z,missing=external_vectors(asset,np.array(['001','003']),{'id_type':'string'},cfg)
    np.testing.assert_array_equal(z,[[10.,1],[30.,3]]);assert not missing.any()
    with pytest.raises(ValueError,match='absent'):external_vectors(asset,np.array(['004']),{'id_type':'string'},cfg)
    z,missing=external_vectors({**asset,'missing_ids':'exclude'},np.array(['004']),{'id_type':'string'},cfg);assert missing[0] and np.isnan(z).all()
    np.save(tmp_path/'ids.npy',np.array(['003','003','002']))
    with pytest.raises(ValueError,match='duplicate'):external_vectors(asset,np.array(['003']),{'id_type':'string'},cfg)


def test_positive_tag_micro_macro_and_known_neighbor_denominators():
    values=np.array([[1,1],[1,0],[0,1],[0,0]],dtype=np.uint8);valid=np.array([1,1,1,0],bool)
    graph=np.array([[1,3],[0,3],[0,1],[0,2]])
    table,labels=purity_lift(values,valid,['x','y'],'multilabel',graph,[2])
    # x: 2 eligible positive anchor-label pairs with purity 1,1; y: 2 with 0,0.5.
    assert labels.set_index('label').loc['x','purity']==1
    assert labels.set_index('label').loc['y','purity']==.25
    assert table.iloc[0].micro_purity==.625 and table.iloc[0].macro_purity==.625
    assert table.iloc[0].known_neighbor_share==4/6
    # Unequal positive supports exercise weighting, and mean lift != purity/chance.
    v=np.array([[1,1],[1,0],[1,0],[0,0]],dtype=np.uint8);g=np.array([[1],[0],[3],[2]])
    summary,detail=purity_lift(v,np.ones(4,bool),['x','y'],'multilabel',g,[1])
    assert summary.iloc[0].micro_purity==.5 and summary.iloc[0].macro_purity==1/3
    assert np.isclose(summary.iloc[0].micro_lift,2/3)
    assert not np.isclose(summary.iloc[0].micro_lift,summary.iloc[0].micro_purity/summary.iloc[0].chance_purity)


def test_numeric_units_constant_nan_offset_stability_and_distance_name():
    y=np.array([0.,2.,10.,12.]);graph=np.array([[1],[0],[3],[2]])
    a=continuous_geometry(y,np.ones(4,bool),graph,[1]);b=continuous_geometry(y*100,np.ones(4,bool),graph,[1])
    assert a.iloc[0].mean_neighbor_abs_difference==2 and b.iloc[0].mean_neighbor_abs_difference==200
    assert a.iloc[0].variance_ratio_to_global==b.iloc[0].variance_ratio_to_global==0
    c=continuous_geometry(np.ones(4),np.ones(4,bool),graph,[1]);assert np.isnan(c.iloc[0].morans_i)
    c=continuous_geometry(y+1e12,np.ones(4,bool),np.array([[1,2],[0,3],[0,3],[1,2]]),[2])
    d=continuous_geometry(y,np.ones(4,bool),np.array([[1,2],[0,3],[0,3],[1,2]]),[2]);np.testing.assert_allclose(c.mean_local_variance,d.mean_local_variance)
    rel=distance_relationship(np.array([[1.,0],[1,1],[0,1],[-1,1]]),y,np.ones(4,bool),cap=4,pairs=100)
    assert 'centered_distance_association' in rel and 'distance_correlation' not in rel


def test_safe_norm_extremes_and_normalized_direction():
    z=np.array([[1e300,1e300],[1e-300,0]])
    n=safe_norms(z);assert np.isfinite(n).all() and (n>0).all()
    unit=l2_normalize(z);np.testing.assert_allclose(np.linalg.norm(unit,axis=1),1)
    np.testing.assert_allclose(unit[0],[1/np.sqrt(2)]*2)


@pytest.mark.parametrize('domain',['retail','telemetry','source_only'])
def test_prepared_examples_validate(domain):
    src=ROOT/'examples'/domain
    valid=validate_bundle(src/'dataset_schema.json',src/'audit_config.json');assert valid['status']=='valid'


def test_limits_unknown_columns_draft_and_weights_fail_early(tmp_path):
    schema,cfg,save=bundle(tmp_path)
    cfg['sample']['strategy']='all';cfg['sample']['max_rows']=10;a,b=save()
    with pytest.raises(ValueError,match='max_rows'):validate_bundle(a,b)
    cfg['sample']['strategy']='uniform_rows';schema['status']='draft';a,b=save()
    with pytest.raises(ValueError,match='Draft'):validate_bundle(a,b)
    assert validate_bundle(a,b,check_data=False,allow_draft=True)['status']=='draft'
    schema['status']='validated';schema['weights']='weighted';a,b=save()
    with pytest.raises(ValueError):validate_bundle(a,b)
    schema.pop('weights');schema['columns'].pop('engineer_note');a,b=save()
    with pytest.raises(ValueError,match='Unmapped'):validate_bundle(a,b)


def test_ragged_text_and_null_vector_behavior(tmp_path):
    from parquet_audit.ingest import embedded_vectors
    spec={'expected_dimension':2};asset={'name':'encoder','column':'v'}
    with pytest.raises(ValueError):embedded_vectors(pd.DataFrame({'v':[[1,2],[1]]}),asset,spec)
    with pytest.raises(ValueError):embedded_vectors(pd.DataFrame({'v':['1 ... 2']}),asset,spec)
    z=embedded_vectors(pd.DataFrame({'v':[[1,None],None]}),asset,spec);assert np.isnan(z[0,1]) and np.isnan(z[1]).all()


def test_source_identity_duplicate_detected_outside_evaluation_sample(tmp_path):
    schema,cfg,save=bundle(tmp_path);frame=pd.read_parquet(schema['dataset']['path']);frame.loc[143,'reading_ref']=frame.loc[0,'reading_ref']
    p=tmp_path/'duplicate.parquet';frame.to_parquet(p,index=False);schema['dataset']['path']=str(p);a,b=save()
    with pytest.raises(ValueError,match='Duplicate IDs'):run_audit(a,b,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_graph_contract_order_range_duplicates_and_distances(tmp_path):
    np.save(tmp_path/'ids.npy',np.array(['a','b','c']));np.save(tmp_path/'indices.npy',np.array([[1],[2],[0]]));np.save(tmp_path/'dist.npy',np.ones((3,1)))
    spec={'index_kind':'positions','ids_path':str(tmp_path/'ids.npy'),'indices_path':str(tmp_path/'indices.npy'),'distances_path':str(tmp_path/'dist.npy')};cfg={'geometry':{'ks':[1]}}
    assert supplied_graph(spec,np.array(['a','b','c']),{'id_type':'string'},cfg)[0].shape==(3,1)
    with pytest.raises(ValueError,match='order'):supplied_graph(spec,np.array(['b','a','c']),{'id_type':'string'},cfg)
    np.save(tmp_path/'indices.npy',np.array([[0],[2],[0]]))
    with pytest.raises(ValueError,match='self'):supplied_graph(spec,np.array(['a','b','c']),{'id_type':'string'},cfg)
    np.save(tmp_path/'indices.npy',np.array([[4],[2],[0]]))
    with pytest.raises(ValueError,match='POSITIONS'):supplied_graph(spec,np.array(['a','b','c']),{'id_type':'string'},cfg)


def test_generic_ab_run_common_cohort_and_report_contract(tmp_path):
    schema,cfg,save=bundle(tmp_path);a,b=save();result=run_audit(a,b,tmp_path/'audit');out=result['out_dir']
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['contract_version']==1 and manifest['weights']=='unweighted'
    ids=np.load(out/'row_ids.npy');assert all(len(v)==6 for v in ids)
    payload=json.loads((out/'explorer_data.json').read_text());assert len(payload['overview']['features'])==6
    assert 'engineer_note' in payload['overview']['features']
    assert payload['overview']['features']['temperature']['units']=='degrees C'
    assert payload['enhancements']['topology']['status']=='disabled'
    doc=(out/'report.html').read_text();assert doc.index('Features at a glance')<doc.index('Direction, magnitude and topology')<doc.index('<label for="filter">Find feature')
    assert '<th>External pipeline</th>' in doc and '#f5f7fa' in doc
    for n in ['condition','pulse_count','temperature','alarms','calibration_ok']:
        folder=result['features'][n]['folder'];pa=pd.read_csv(folder/'tables/probe_predictions.csv');pb=pd.read_csv(folder/'tables/comparison_probe_predictions.csv')
        assert set(pa.row_position)==set(pb.row_position)
    with pytest.raises(ValueError,match='fresh'):run_audit(a,b,out)


def test_source_only_has_no_fake_geometry(tmp_path):
    schema,cfg,save=bundle(tmp_path,'source_only');a,b=save();result=run_audit(a,b,tmp_path/'audit')
    payload=json.loads((result['out_dir']/'explorer_data.json').read_text());assert not payload['available'] and not payload['overview']['ks']
    assert not (result['out_dir']/'neighbors.npy').exists()
    assert all(not r['geometry'] for r in payload['overview']['features'].values())


def test_probability_contract_matched_truth_and_label_order():
    from parquet_audit.predictions import probability_metrics
    truth=pd.Series(['a','b',None],dtype=object);p=pd.Series([[.8,.2],[.1,.9],[.3,.7]],dtype=object)
    result=probability_metrics(truth,p,{'kind':'categorical'},{'label_order':['a','b']})
    assert result['probability_n_matched']==2;assert np.isclose(result['log_loss'],-np.log([.8,.9]).mean())
    with pytest.raises(ValueError,match='sum'):probability_metrics(truth,pd.Series([[.2,.2],None,None]),{'kind':'categorical'},{'label_order':['a','b']})


def test_cli_real_inspect_and_structure_only(tmp_path):
    src=ROOT/'examples/telemetry'
    cmd=[sys.executable,'-m','parquet_audit','inspect','--parquet',str(src/'readings.parquet'),'--out',str(tmp_path/'inspection'),'--sample-rows','5']
    done=subprocess.run(cmd,text=True,capture_output=True);assert done.returncode==0,done.stderr
    inspection=json.loads((tmp_path/'inspection/inspection.json').read_text());assert inspection['footer_rows']==144 and inspection['columns']['condition']['sample_n']==5
    done=subprocess.run([sys.executable,'-m','parquet_audit','validate','--schema',str(src/'dataset_schema.json'),'--config',str(src/'audit_config.json'),'--structure-only'],text=True,capture_output=True)
    assert done.returncode==0,done.stderr


def test_group_leakage_is_a_run_contract_error(tmp_path):
    schema,cfg,save=bundle(tmp_path,'retail');frame=pd.read_parquet(schema['dataset']['path']);frame.loc[1,'assignment']='holdout'
    # Customer 0 now appears in fit and holdout: row 0 already holdout; move row 2 to fit.
    frame.loc[2,'assignment']='fit';p=tmp_path/'leak.parquet';frame.to_parquet(p,index=False);schema['dataset']['path']=str(p)
    cfg['sample']['strategy']='all';cfg['sample']['max_rows']=160;a,b=save()
    with pytest.raises(ValueError,match='leaks'):run_audit(a,b,tmp_path/'out')


def test_auxiliary_stratum_retains_numeric_feature_and_source_only_switch(tmp_path):
    schema,cfg,save=bundle(tmp_path);schema['columns']['pulse_count']['auxiliary_roles']=['stratum'];cfg['nulls'].update(modes=['stratified'],stratum_column='pulse_count')
    schema['columns']['temperature'].update(geometry=False,probe=False)
    a,b=save();result=run_audit(a,b,tmp_path/'audit');payload=json.loads((result['out_dir']/'explorer_data.json').read_text())
    assert 'pulse_count' in payload['overview']['features'];assert not payload['overview']['features']['temperature']['geometry']
    assert not payload['overview']['features']['temperature']['probes']


def test_text_derived_datetime_and_exact_sample_scope(tmp_path):
    schema,cfg,save=bundle(tmp_path,'source_only');cfg['source']['scope']='evaluation';a,b=save();result=run_audit(a,b,tmp_path/'audit')
    stats=json.loads((result['out_dir']/'source_population/summaries.json').read_text());assert stats['department']['n_items']==36
    assert stats['product_note']['scope']=='Retained evaluation sample' and stats['note_length']['kind']=='count'
    assert stats['purchase_year']['kind']=='numeric'
    assert result['features']['department']['prediction_metrics']['probability_n_matched']>0


def test_skipped_tag_support_still_renders_without_optional_plots(tmp_path):
    schema,cfg,save=bundle(tmp_path);schema['columns']['alarms']['min_count']=10000;a,b=save();result=run_audit(a,b,tmp_path/'audit')
    payload=json.loads((result['out_dir']/'explorer_data.json').read_text());assert payload['overview']['features']['alarms']['status']=='skipped'
    assert (result['out_dir']/'report.html').exists()


def test_long_untrusted_name_and_explicit_struct_extract(tmp_path):
    schema,cfg,save=bundle(tmp_path);frame=pd.read_parquet(schema['dataset']['path']);name='very long numerical feature '+('with readable units '*8)+'</script><script>alert(1)</script>'
    frame=frame.rename(columns={'pulse_count':name});schema['columns'][name]=schema['columns'].pop('pulse_count')
    frame['payload']=[{'reading':i%7} for i in range(len(frame))];schema['columns']['payload']={'role':'feature','kind':'count','struct_path':['reading'],'units':'readings'}
    p=tmp_path/'odd.parquet';frame.to_parquet(p,index=False);schema['dataset']['path']=str(p);a,b=save();result=run_audit(a,b,tmp_path/'audit')
    html=result['report'].read_text();assert '</script><script>alert(1)' not in html and '\\u003c/script' in html
    assert result['features']['payload']['source']['n_known']>0
    assert name in json.loads((result['out_dir']/'feature_measurements.json').read_text())['features']
