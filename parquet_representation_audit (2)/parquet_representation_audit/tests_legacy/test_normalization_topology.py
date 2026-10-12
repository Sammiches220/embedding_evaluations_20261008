"""Hand-computed persistence/entropy cases and same-split pipeline contracts."""
import json
import re
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from embedding_feature_audit import audit_dataframe
from embedding_feature_audit.geometry import make_neighbors
from embedding_feature_audit.normalization import l2_normalize,normalization_options,normalization_probes
from embedding_feature_audit.topology import component_filtration,rips_h1,component_measure,shuffled_values,topology_options,audit_topology


def test_normalization_removes_radius_and_preserves_cosine_neighbors():
    z=np.random.default_rng(2).normal(size=(20,5)).astype('float32')
    unit=l2_normalize(z)
    np.testing.assert_allclose(np.linalg.norm(unit,axis=1),1,atol=1e-7)
    np.testing.assert_array_equal(make_neighbors(z,4)[0],make_neighbors(unit,4)[0])
    tiny=l2_normalize(np.array([[1e-30,0]],dtype='float32'))
    np.testing.assert_array_equal(tiny,[[1,0]])
    with pytest.raises(ValueError):l2_normalize(np.zeros((2,3)))


def test_radial_information_and_identical_probe_splits():
    y=np.array(['a','b']*40);z=np.column_stack([np.where(y=='a',1.,2.),np.zeros(80)])
    data=dict(values=y,valid=np.ones(80,bool),names=['a','b'],kind='categorical')
    status,scores,pred,split=normalization_probes(z,data,normalization_options({}),seed=42)
    assert status['status']=='ok'
    assert scores.split_signature.nunique()==1
    linear=scores[scores.model=='linear'].set_index('variant')
    assert linear.loc['raw_standardized','balanced_accuracy']==1
    assert linear.loc['norm_only','balanced_accuracy']==1
    assert linear.loc['unit','balanced_accuracy']==.5
    assert set(split.variant)=={'raw_standardized','unit','unit_standardized','norm_only'}
    assert (linear.convergence_warnings>=0).all()


def test_group_aware_training_only_regularization_tuning():
    rng=np.random.default_rng(8);z=rng.normal(size=(48,3));y=np.array(['a','b']*24)
    groups=np.arange(48)//2
    data=dict(values=y,valid=np.ones(48,bool),names=['a','b'],kind='categorical')
    options=normalization_options({'normalization_analysis':{'variants':['raw_standardized','unit'],'regularization_grid':[.1,1.]}})
    status,scores,_,split=normalization_probes(z,data,options,groups=groups)
    assert status['status']=='ok' and not status['errors']
    assert set(scores[scores.model=='linear'].tuning)=={'training-only CV'}
    for variant,table in split.groupby('variant'):
        train=table[table.split=='train'].row_position.to_numpy();test=table[table.split=='test'].row_position.to_numpy()
        assert not set(groups[train]) & set(groups[test])


def test_normalization_row_limit_is_explicit():
    data=dict(values=np.arange(10.),valid=np.ones(10,bool),names=[],kind='continuous')
    status,*_=normalization_probes(np.ones((10,2)),data,normalization_options({'normalization_analysis':{'max_rows':8}}))
    assert status['status']=='skipped' and 'explicit evaluation sample' in status['reason']


def test_h0_merges_and_censoring():
    d=cdist(np.array([[0.],[1.],[3.]]),np.array([[0.],[1.],[3.]]))
    table,partitions,bars,merges=component_filtration(d,[0,.5,1,2,3])
    assert table.beta0.tolist()==[3,3,2,1,1]
    assert sorted(bars[bars.status=='finite'].death)==[1.,2.]
    assert sum(bars.status=='essential')==1
    assert merges.merged_size.tolist()==[2,3]
    _,_,limited,_=component_filtration(d,[0,.5])
    assert sum(limited.status=='right-censored')==2


def test_h1_square_dies_when_triangles_fill():
    z=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    bars,n=rips_h1(cdist(z,z),2)
    assert n==4+6+4
    assert len(bars)==1 and bars.iloc[0].status=='finite'
    assert bars.iloc[0].birth==1
    assert np.isclose(bars.iloc[0].death,np.sqrt(2))
    assert len(bars.iloc[0].cycle_edges)==4
    assert set(sum(bars.iloc[0].cycle_edges,[]))==set(range(4))
    limited,_=rips_h1(cdist(z,z),1.1)
    assert len(limited)==1 and limited.iloc[0].status=='right-censored'
    assert np.isnan(limited.iloc[0].death)


def test_h1_triangle_and_duplicate_points_have_no_nonzero_loop():
    triangle=np.ones((3,3))-np.eye(3)
    bars,_=rips_h1(triangle,1)
    assert bars.empty
    bars,_=rips_h1(np.zeros((4,4)),0)
    assert bars.empty


def test_h1_hexagon_has_expected_lifetime():
    theta=np.arange(6)*np.pi/3;z=np.column_stack([np.cos(theta),np.sin(theta)])
    bars,_=rips_h1(cdist(z,z),2.01)
    longest=bars.iloc[(bars.death-bars.birth).argmax()]
    assert np.isclose(longest.birth,1) and np.isclose(longest.death,np.sqrt(3))


def test_h1_budget_guards():
    with pytest.raises(ValueError,match='budget'):rips_h1(np.ones((8,8))-np.eye(8),1,max_simplices=20)
    with pytest.raises(ValueError,match='128'):rips_h1(np.zeros((129,129)),1)


def test_component_entropy_tags_and_numeric_variance():
    components=np.array([0,0,1,1]);valid=np.ones(4,bool)
    category=component_measure(np.array(['a','a','b','b']),valid,['a','b'],'categorical',components)
    assert category['value']==0 and category['retained_share']==1
    assert component_measure(np.array(['a','a','b','b']),valid,['a','b'],'categorical',np.zeros(4,int))['value']==1
    tags=np.array([[1,0],[1,1],[0,1],[0,0]],dtype='uint8')
    assert component_measure(tags,valid,['x','y'],'multilabel',components)['value']==.5
    numeric=component_measure(np.array([0.,2.,10.,12.]),valid,[],'continuous',components,26.)
    assert np.isclose(numeric['value'],1/26)
    dropped=component_measure(tags,valid,['x','y'],'multilabel',components,min_known=3)
    assert dropped['retained_share']==0 and np.isnan(dropped['value'])


def test_whole_tag_shuffle_preserves_unknown_rows_and_strata():
    y=np.array([[1,1],[1,0],[0,1],[0,0],[1,0]],dtype='uint8');valid=np.array([1,1,1,1,0],bool)
    strata=np.array(['a','a','b','b','a'])
    shuffled=shuffled_values(y,valid,np.random.default_rng(5),strata)
    np.testing.assert_array_equal(shuffled[-1],y[-1])
    for key in ['a','b']:
        rows=np.flatnonzero(valid&(strata==key))
        assert sorted(map(tuple,y[rows]))==sorted(map(tuple,shuffled[rows]))


def test_shared_samples_radii_and_identical_models():
    z=np.random.default_rng(9).normal(size=(30,3));ids=np.arange(30).astype(str)
    data={'kind':dict(values=np.array(['a','b']*15),valid=np.ones(30,bool),names=['a','b'],kind='categorical')}
    options=topology_options({'topology':{'sample_size':16,'repetitions':2,'permutations':3,'h1_sample_size':8,'spaces':['raw_euclidean','unit_chordal']}})
    status,summary,raw,structure,bars,merges,samples=audit_topology({'A':z,'B':z},data,ids,options)
    assert status['actual_repetitions']==2 and len(samples)==32
    a=summary[summary.embedding=='A'].sort_values(['space','radius']);b=summary[summary.embedding=='B'].sort_values(['space','radius'])
    np.testing.assert_allclose(a.organization_gap,b.organization_gap,equal_nan=True)
    for space in status['radii']:
        for model in ['A','B']:
            rows=raw[(raw.space==space)&(raw.embedding==model)]
            assert set(rows.radius)==set(status['radii'][space])
    assert all(row['status']=='ok' for row in status['h1_status'])


def test_full_sample_does_not_pretend_to_be_repeated_samples():
    z=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    options=topology_options({'topology':{'sample_size':20,'repetitions':3,'permutations':0,'h1':False}})
    status,*_=audit_topology({'A':z},{},np.arange(4).astype(str),options)
    assert status['actual_repetitions']==1 and status['requested_repetitions']==3


def test_enhanced_report_and_main_l2_contract():
    rng=np.random.default_rng(10);z=rng.normal(size=(24,3));df=pd.DataFrame({'id':np.arange(24),'label':['a','b']*12,'embedding':list(z)})
    config={'ks':[2,4],'permutations':0,'probe_features':False,'missingness_features':False,'verbose':False,
        'embedding_normalization':'l2','normalization_analysis':{'enabled':True},
        'topology':{'enabled':True,'sample_size':12,'repetitions':2,'h1_sample_size':8,'permutations':2}}
    with tempfile.TemporaryDirectory() as tmp:
        result=audit_dataframe(df,tmp,features={'label':'categorical'},config=config)
        payload=json.loads(result['explorer_data'].read_text())
        assert payload['enhancements']['normalization']['status']=='ok'
        assert payload['enhancements']['topology']['status']=='ok'
        assert (Path(tmp)/'embedding_norms.csv').exists()
        assert (Path(tmp)/'topology_barcodes.csv').exists()
        assert (Path(tmp)/'normalization_probe_scores.csv').exists()
        manifest=json.loads((Path(tmp)/'manifest.json').read_text())
        assert manifest['embedding_preprocessing']=='l2'
        # Norm diagnostics retain original magnitude, even when main analysis is unit-normalized.
        np.testing.assert_allclose(pd.read_csv(Path(tmp)/'embedding_norms.csv').embedding_a_norm,np.linalg.norm(z,axis=1),rtol=1e-6)
        html=result['report'].read_text();assert 'Unit-normalization comparisons' in html and 'Filtration and feature organization' in html
        script=(Path(tmp)/'assets/enhancements.js').read_text()
        for control in re.findall(r"\$\('([^']+)'\)",script):assert f'id="ea-{control}"' in html


def test_euclidean_supplied_graph_rejected_for_main_normalization():
    df=pd.DataFrame({'id':[1,2],'embedding':[[1.,0.],[0.,1.]],'label':['a','b']})
    with tempfile.TemporaryDirectory() as temp:
        with pytest.raises(ValueError,match='Euclidean neighbors'):
            audit_dataframe(df,temp,features={'label':'categorical'},neighbors=np.array([[1],[0]]),config={'metric':'euclidean','embedding_normalization':'l2'})


def test_topology_memory_guard_and_numeric_offset_stability():
    z=np.random.default_rng(7).normal(size=(20,2))
    options=topology_options({'topology':{'sample_size':20,'max_distance_bytes':1024}})
    status,*tables=audit_topology({'A':z},{},np.arange(20).astype(str),options)
    assert status['status']=='skipped' and 'memory' in status['reason']
    assert all(table.empty for table in tables)
    y=np.array([0.,2.,10.,12.])+1e12
    measured=component_measure(y,np.ones(4,bool),[],'continuous',np.array([0,0,1,1]),26.)
    assert np.isclose(measured['value'],1/26)


def test_cli_enhancement_flags_actual_parquet(tmp_path):
    import subprocess,sys
    z=np.random.default_rng(4).normal(size=(16,3))
    df=pd.DataFrame({'id':np.arange(16),'category':['a','b']*8,'embedding':list(z)})
    data=tmp_path/'posts.parquet';df.to_parquet(data,index=False)
    config=tmp_path/'config.json';config.write_text(json.dumps({'ks':[1,2],'permutations':0,'probe_features':False,
        'missingness_features':False,'verbose':False,'features':{'category':'categorical'},
        'topology':{'permutations':1,'h1_sample_size':8}}))
    output=tmp_path/'audit'
    done=subprocess.run([sys.executable,'-m','embedding_feature_audit.dataframe_cli','--dataframe',str(data),
        '--config',str(config),'--normalize-embeddings','l2','--normalization-analysis','--topology','--out',str(output)],capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['embedding_preprocessing']=='l2'
    assert manifest['enhancements']['normalization']['status']=='ok'
    assert manifest['enhancements']['topology']['status']=='ok'
