"""Entropy definitions, shared-anchor comparisons and fixed held-out probe contracts."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from embedding_feature_audit import audit_dataframe
from embedding_feature_audit.feature_overview import local_entropy, feature_overview
from embedding_feature_audit.geometry import continuous_geometry, purity_lift
from embedding_feature_audit.modeling import run_probes


def test_category_entropy_fixed_vocabulary_unknowns():
    y=np.array(['a','a','b','?']);valid=np.array([1,1,1,0],bool)
    graph=np.array([[1,2],[0,3],[0,1],[0,2]])
    entropy=local_entropy(y,valid,['a','b'],'categorical',graph,[1,2],batch_size=2)
    np.testing.assert_allclose(entropy[:3,0],0)
    np.testing.assert_allclose(entropy[:3,1],[1,0,0])
    assert np.isnan(entropy[3]).all()
    assert np.isnan(local_entropy(y,np.zeros(4,bool),['a','b'],'categorical',graph,[2])).all()


def test_tags_use_binary_entropy_not_occurrence_entropy():
    y=np.array([[1,1],[1,0],[0,0]],dtype='uint8');valid=np.ones(3,bool)
    graph=np.array([[1,2],[0,2],[0,1]])
    result=local_entropy(y,valid,['x','y'],'multilabel',graph,[2])
    np.testing.assert_allclose(result[:,0],[.5,1,.5])
    assert np.isnan(local_entropy(np.arange(3.),valid,[],'continuous',graph,[2])).all()


def test_anchor_filter_does_not_filter_candidate_neighbors_or_baselines():
    y=np.array(['a','a','b','b']);valid=np.ones(4,bool)
    graph=np.array([[1,2],[0,2],[0,3],[1,2]])
    mask=np.array([1,0,0,0],bool)
    geom,labels=purity_lift(y,valid,['a','b'],'categorical',graph,[2],anchor_mask=mask)
    assert geom.iloc[0].micro_purity==.5
    assert labels.iloc[0].baseline_prevalence==.5
    numeric=continuous_geometry(np.array([0.,10.,20.,30.]),valid,graph,[2],mask)
    assert numeric.iloc[0].mean_neighbor_abs_difference==15
    assert numeric.iloc[0].variance_ratio_to_global==25/125


def test_models_selection_keeps_identical_split_between_embeddings():
    y=np.array(['a','b']*20);valid=np.ones(40,bool);z=np.column_stack([np.arange(40.),np.arange(40)%2])
    a=run_probes(z,y,valid,['a','b'],'categorical',models=['baseline','linear'])
    b=run_probes(np.column_stack([z,np.ones(40)]),y,valid,['a','b'],'categorical',models=['baseline','linear'])
    assert set(a[0].model)=={'baseline','linear'}
    np.testing.assert_array_equal(a[3],b[3]);np.testing.assert_array_equal(a[4],b[4])
    assert set(a[2].row_position)==set(b[2].row_position)


def test_aggregate_view_survives_limits_and_compares_probe_b():
    rng=np.random.default_rng(7);z=rng.normal(size=(40,3));y=np.where(z[:,0]>0,'a','b')
    df=pd.DataFrame({'id':np.arange(40),'category':y,'embedding':list(z)})
    with tempfile.TemporaryDirectory() as temp:
        result=audit_dataframe(df,temp,features={'category':'categorical'},config={
            'ks':[2,5],'permutations':0,'probe_features':True,'missingness_features':False,
            'probe_models':['linear'],'explorer_max_rows':10,'comparison_bootstraps':0,'verbose':False},comparison_embeddings=z)
        payload=json.loads(result['explorer_data'].read_text())
        assert payload['available'] is False
        record=payload['overview']['features']['category']
        assert 'entropy_rows' not in record
        assert {r['model'] for r in record['probes']}=={'baseline','linear'}
        assert len(record['comparison_probes'])==2
        assert [r['delta'] for r in record['paired_entropy_delta']]==[0,0]
        assert [r['k'] for r in record['entropy']]==[2,5]
        assert (Path(temp)/'features/001/tables/local_entropy.csv').exists()
        text=result['report'].read_text()
        assert 'Features at a glance' in text and 'Tag / feature queries' not in text
        assert 'id="ex-k"' not in text and 'id="ex-label"' not in text


def test_paired_entropy_excludes_nonshared_anchors():
    y=np.array(['a','a','b','?']);valid=np.array([1,1,1,0],bool)
    a=np.array([[1,2],[3,0],[0,1],[0,2]])
    b=np.array([[1,2],[3,2],[3,1],[0,2]])
    encoded={'f':dict(values=y,valid=valid,names=['a','b'],kind='categorical')}
    results={'f':dict(kind='categorical',folder=Path('001'),status='ok')}
    view,_=feature_overview(encoded,results,a,[1,2],{},dict(graph=b),False)
    row=view['features']['f']['paired_entropy_delta']
    assert row[0]['n_anchors']==1 and row[0]['delta']==0
    assert row[1]['n_anchors']==3 and row[1]['delta']==0
