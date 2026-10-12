"""Regression tests for substantive metric/alignment failures, using unittest."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from embedding_feature_audit.dataframe import encode_feature,align_predictions,embedding_matrix,default_features
from embedding_feature_audit.geometry import make_neighbors,validate_neighbors,purity_lift,continuous_geometry
from embedding_feature_audit.statistics import permutation_analysis,prediction_metrics
from embedding_feature_audit.neighborhood import neighbor_indices
from embedding_feature_audit import audit_dataframe


class DataFrameTests(unittest.TestCase):
    def test_no_self_and_no_dropped_closest_neighbor(self):
        z=np.array([[0.,0.],[.1,0.],[10.,0.],[11.,0.]])
        graph,_=make_neighbors(z,3,'euclidean',batch_size=2)
        self.assertEqual(graph[0,0],1)
        self.assertFalse(np.any(graph==np.arange(4)[:,None]))
        old=neighbor_indices(z,3)
        self.assertEqual(old[0,0],1)
        self.assertEqual(old.shape,(4,3))

    def test_duplicate_embedding_ties_exclude_actual_self(self):
        z=np.ones((5,2))
        graph,_=make_neighbors(z,4,'euclidean',batch_size=2)
        self.assertFalse(np.any(graph==np.arange(5)[:,None]))
        for i,row in enumerate(graph):
            self.assertEqual(set(row),set(range(5))-{i})

    def test_zero_is_a_category_unless_explicitly_missing(self):
        s=pd.Series([0,1,0,np.nan])
        y,valid,names=encode_feature(s,{'kind':'categorical'})
        self.assertEqual(valid.tolist(),[True,True,True,False])
        self.assertEqual(len(names),2)
        _,valid,_=encode_feature(s,{'kind':'categorical','missing_values':[0]})
        self.assertEqual(valid.tolist(),[False,True,False,False])

    def test_hand_calculated_lift_imbalance(self):
        y=np.array(['a','a','a','b']); valid=np.ones(4,dtype=bool)
        graph=np.array([[1],[0],[0],[0]])
        summary,labels=purity_lift(y,valid,['a','b'],'categorical',graph,[1])
        a=labels[labels.label=='a'].iloc[0]
        self.assertAlmostEqual(a.purity,1.)
        self.assertAlmostEqual(a.lift,4/3)
        self.assertAlmostEqual(summary.iloc[0].micro_purity,.75)
        self.assertAlmostEqual(summary.iloc[0].macro_purity,.5)
        self.assertAlmostEqual(summary.iloc[0].micro_lift,1.)

    def test_multilabel_not_whole_string_or_negative_agreement(self):
        s=pd.Series(['a b','b a','c',''])
        y,valid,names=encode_feature(s,{'kind':'multilabel','min_count':1,'max_labels':None})
        self.assertTrue(np.array_equal(y[0],y[1]))
        self.assertFalse(valid[3])
        graph=np.array([[1],[0],[0],[0]])
        summary,labels=purity_lift(y,valid,names,'multilabel',graph,[1])
        self.assertAlmostEqual(labels.loc[labels.label=='a','purity'].iloc[0],1.)
        self.assertAlmostEqual(labels.loc[labels.label=='a','lift'].iloc[0],1.5)
        self.assertAlmostEqual(summary.iloc[0].macro_purity,2/3)

    def test_unknown_neighbors_removed_from_denominator(self):
        y=np.array(['a','a','ignored']); valid=np.array([1,1,0],dtype=bool)
        graph=np.array([[2,1],[2,0],[0,1]])
        s,l=purity_lift(y,valid,['a'],'categorical',graph,[1,2])
        self.assertEqual(s.k.tolist(),[2])
        self.assertEqual(s.iloc[0].micro_purity,1.)
        self.assertEqual(l.iloc[0].effective_neighbors,1.)

    def test_prediction_alignment_by_id_not_dataframe_index(self):
        df=pd.DataFrame({'id':[20,10]},index=[7,3])
        pred=pd.DataFrame({'id':[10,20],'value':['x','y']})
        a=align_predictions(df,pred)
        self.assertEqual(a.value.tolist(),['y','x'])
        self.assertEqual(a.index.tolist(),[7,3])
        with self.assertRaises(ValueError): align_predictions(df,pd.concat([pred,pred]))

    def test_stratified_null_preserves_group_labels(self):
        y=np.array(['a','a','b','b']); valid=np.ones(4,dtype=bool)
        graph=np.array([[1],[0],[3],[2]])
        observed,_=purity_lift(y,valid,['a','b'],'categorical',graph,[1])
        nulls,score=permutation_analysis(y,valid,['a','b'],'categorical',graph,[1],observed,
                                       repetitions=4,mode='stratified',strata=np.array([0,0,1,1]))
        self.assertTrue((nulls.micro_purity==1).all())
        self.assertTrue(np.isnan(score.iloc[0].micro_purity_z))

    def test_continuous_constant_geometry_not_infinite(self):
        y=np.ones(4); valid=np.ones(4,dtype=bool); graph=np.array([[1],[0],[3],[2]])
        s=continuous_geometry(y,valid,graph,[1])
        self.assertTrue(np.isnan(s.iloc[0].morans_i))
        self.assertEqual(s.iloc[0].mean_neighbor_abs_difference,0.)

    def test_graph_and_embedding_input_validation(self):
        with self.assertRaises(ValueError): validate_neighbors(np.array([[0],[1]]),2,1)
        with self.assertRaises(ValueError): embedding_matrix(pd.DataFrame({'embedding':['[1 ... 2]','[3 ... 4]']}))
        with self.assertRaises(ValueError): embedding_matrix(pd.DataFrame({'embedding':[[1,2],[3]]}))
        with self.assertRaises(ValueError): embedding_matrix(pd.DataFrame({'embedding':[[1,2],[3,np.nan]]}))

    def test_conditional_source_and_external_population_lift(self):
        df=pd.DataFrame({'id':range(12),'embedding':list(np.random.default_rng(2).normal(size=(12,3))),
                         'target':['a']*6+['b']*6,
                         'year_group':pd.Categorical(['old']*6+['new']*5+[None])})
        with tempfile.TemporaryDirectory() as directory:
            r=audit_dataframe(df,directory,features={'target':'categorical'},config={
                'ks':[2], 'permutations':2,'probe_features':False,'missingness_features':False,
                'strata_column':'year_group','null_modes':['stratified','prevalence-random'],
                'population_prevalence':{'target':{'a':.25,'b':.75}},'verbose':False})
            p=r['out_dir']/'features/001/tables'
            conditional=pd.read_csv(p/'conditional_source_summary.csv')
            self.assertEqual(set(conditional.group),{'old','new','__missing__'})
            labels=r['features']['target']['per_label']
            self.assertAlmostEqual(labels[labels.label=='a'].iloc[0].baseline_prevalence,.25)
            self.assertTrue((p/'conditional_prevalence.csv').is_file())

    def test_dataframe_report_and_matched_prediction_coverage(self):
        df=pd.DataFrame({'id':range(20),'embedding':list(np.random.default_rng(0).normal(size=(20,3))),
                         'target': ['a']*10+['b']*10})
        pred=pd.DataFrame({'id':range(10),'pred':['a']*10})
        with tempfile.TemporaryDirectory() as directory:
            result=audit_dataframe(df,directory,features={'target':'categorical'},predictions=pred,
                config={'ks':[2,3],'permutations':2,'probe_features':False,'missingness_features':False,
                        'predictions':{'target':'pred'},'verbose':False})
            import json
            metrics=json.loads((Path(directory)/'features/001/prediction_metrics.json').read_text())
            self.assertEqual(metrics['n_compared'],10)
            self.assertEqual(metrics['accuracy'],1.)
            self.assertTrue(result['report'].is_file())
            self.assertNotIn('NaN',(Path(directory)/'manifest.json').read_text())


if __name__=='__main__': unittest.main()
