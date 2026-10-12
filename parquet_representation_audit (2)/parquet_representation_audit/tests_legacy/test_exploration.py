"""Hand-calculated selection, comparison, and row-scope regression cases."""
import json
import tempfile
import unittest
import subprocess
import sys
import re
from html.parser import HTMLParser
from pathlib import Path
import numpy as np
import pandas as pd
from embedding_feature_audit import audit_dataframe, query_dataframe, compare_cohorts
from embedding_feature_audit.comparison import anchor_scores, paired_difference, align_embedding_rows
from embedding_feature_audit.exploration import row_feature, edge_distances


class ExplorationTests(unittest.TestCase):
    def sample(self):
        return pd.DataFrame({'id':[10,20,30,40], 'score':[0,10,20,np.nan],
            'tag_string_general':['water night','water','sky',None],
            'category':['a','b','b',None],
            'embedding':list(np.array([[1.,0.],[.9,.1],[0.,1.],[.1,.9]],dtype='float32'))})

    def config(self, **extra):
        return {'ks':[1,2],'permutations':0,'probe_features':False,
                'missingness_features':False,'comparison_bootstraps':10,'verbose':False,**extra}

    def features(self):
        return {'score':'continuous','tag_string_general':{'kind':'multilabel','min_count':2},'category':'categorical'}

    def test_joint_queries_unknown_logic_and_unselected_tags(self):
        df=self.sample()
        rules=[{'feature':'score','min':5},{'feature':'tag_string_general','include':['water']}]
        self.assertEqual(query_dataframe(df,rules,features=self.features()).tolist(),[False,True,False,False])
        self.assertEqual(query_dataframe(df,[{'feature':'tag_string_general','include':['night']}],features=self.features()).tolist(),[True,False,False,False])
        self.assertEqual(query_dataframe(df,[{'feature':'tag_string_general','known':'unknown'}],features=self.features()).tolist(),[False,False,False,True])
        self.assertEqual(query_dataframe(df,[{'feature':'tag_string_general','exclude':['water']}],features=self.features()).tolist(),[False,False,True,False])

    def test_cohort_counts_percentage_points_coverage_and_overlap(self):
        table=compare_cohorts(self.sample(),[10,20,40],[20,30],features=self.features())
        water=table[(table.feature=='tag_string_general') & (table.label=='water')].iloc[0]
        self.assertEqual(water.count_a,2); self.assertEqual(water.count_b,1)
        self.assertAlmostEqual(water.coverage_a,2/3)
        self.assertAlmostEqual(water.percentage_point_delta,50)
        self.assertEqual(water.prevalence_ratio,2)
        self.assertEqual(water.overlap,1)
        numeric=table[table.feature=='score'].iloc[0]
        self.assertEqual(numeric.mean_a,5); self.assertEqual(numeric.mean_b,15)
        self.assertEqual(numeric.mean_delta,-10)
        with self.assertRaises(ValueError): compare_cohorts(self.sample(),[99],[10],features=self.features())
        with self.assertRaises(ValueError): compare_cohorts(self.sample(),[10,10],[20],features=self.features())

    def test_tag_positive_pair_weights_and_unknown_denominators(self):
        values=np.array([[1,1],[1,0],[0,1],[0,0]],dtype='uint8')
        valid=np.array([1,1,1,0],dtype=bool)
        graph=np.array([[1,3],[0,2],[0,3],[1,2]])
        scores,weights=anchor_scores(values,valid,['a','b'],'multilabel',graph,2)
        self.assertEqual(weights.tolist(),[2,1,1,0])
        self.assertEqual(scores[:3].tolist(),[.5,.5,1.])
        result=paired_difference(scores,scores,weights,repetitions=10)
        self.assertEqual(result['a'],.625); self.assertEqual(result['delta'],0.)
        self.assertEqual(result['ci_low'],0.)

    def test_continuous_anchor_vs_edge_average_is_explicit(self):
        values=np.array([0.,10.,20.,np.nan]); valid=np.isfinite(values)
        graph=np.array([[1,3],[0,2],[0,3],[0,1]])
        scores,_=anchor_scores(values,valid,[],'continuous',graph,2)
        self.assertEqual(scores[:3].tolist(),[10.,10.,20.])
        self.assertTrue(np.isnan(scores[3]))

    def test_group_bootstrap_and_missing_paired_evidence(self):
        a=np.array([0.,0.,np.nan,1.]); b=np.array([1.,1.,1.,0.]); weights=np.ones(4)
        result=paired_difference(a,b,weights,groups=np.array(['x','x','y','z']),repetitions=40)
        self.assertEqual(result['n_paired_anchors'],3)
        self.assertEqual(result['n_resampling_units'],2)
        self.assertAlmostEqual(result['delta'],1/3)
        result=paired_difference(a,b,weights,groups=np.array(['x']*4),repetitions=40)
        self.assertIsNone(result['ci_low'])
        result=paired_difference(np.array([np.nan]),np.array([1.]),np.ones(1))
        self.assertIsNone(result['delta'])

    def test_comparison_alignment_and_different_dimensions(self):
        b=np.array([[1.,2.,3.],[4.,5.,6.]])
        self.assertTrue(np.array_equal(align_embedding_rows(b,[20,10],[10,20]),b[::-1]))
        with self.assertRaises(ValueError): align_embedding_rows(b,[20,20],[10,20])
        with self.assertRaises(ValueError): align_embedding_rows(b,[20,30],[10,20])

    def test_analytic_edge_distances(self):
        z=np.array([[1.,0.],[0.,1.],[-1.,0.]])
        graph=np.array([[1,2],[0,2],[0,1]])
        self.assertEqual(edge_distances(z,graph,'cosine')[0].tolist(),[1.,2.])
        self.assertAlmostEqual(edge_distances(z,graph,'euclidean')[0,0],np.sqrt(2),places=6)

    def test_full_report_snapshot_safe_labels_ids_and_comparison(self):
        df=self.sample(); df['category']=['</script><script>alert(1)</script>','b','b',None]
        z=np.stack(df.embedding)
        with tempfile.TemporaryDirectory() as tmp:
            r=audit_dataframe(df,tmp,features=self.features(),config=self.config(),
                comparison_embeddings=z[::-1],comparison_ids=df.id.to_numpy()[::-1])
            payload=json.loads(r['explorer_data'].read_text())
            self.assertEqual(payload['ids'],['10','20','30','40'])
            self.assertEqual(payload['features']['tag_string_general']['values'][0],['night','water'])
            self.assertNotIn('night',payload['features']['tag_string_general']['audited_labels'])
            self.assertEqual(payload['features']['category']['values'][-1],None)
            self.assertTrue(payload['comparison'])
            self.assertTrue((Path(tmp)/'assets/explorer.js').exists())
            text=r['report'].read_text(); embedded=text.split('<script id="explorer-payload" type="application/json">')[1].split('</script>')[0]
            self.assertNotIn('</script>',embedded); self.assertEqual(json.loads(embedded)['ids'],payload['ids'])
            self.assertIn('What these results mean',text)
            self.assertTrue((Path(tmp)/'features/001/explanation.json').exists())
            self.assertTrue((Path(tmp)/'embedding_comparison.csv').exists())
            self.assertTrue((r['comparison']['paired'].delta==0).all())
            class Controls(HTMLParser):
                def __init__(self):
                    super().__init__(); self.ids=[]
                def handle_starttag(self,tag,attrs):
                    self.ids.extend(value for key,value in attrs if key=='id')
            parser=Controls(); parser.feed(text)
            self.assertEqual(len(parser.ids),len(set(parser.ids)))
            script=(Path(tmp)/'assets/explorer.js').read_text()
            for control in re.findall(r"\$\('([^']+)'\)",script):
                self.assertIn('ex-'+control,parser.ids)
            self.assertIn('src="assets/explorer.js"',text)
            self.assertIn('href="assets/explorer.css"',text)

    def test_common_valid_row_filter_alignment_and_graph_rejection(self):
        df=self.sample(); a=np.stack(df.embedding); a[1,0]=np.nan
        b=np.stack(df.embedding); b[2]=0
        with tempfile.TemporaryDirectory() as tmp:
            r=audit_dataframe(df,tmp,features=self.features(),embeddings=a,comparison_embeddings=b,
                config=self.config(drop_invalid_embeddings=True))
            payload=json.loads(r['explorer_data'].read_text())
            self.assertEqual(payload['ids'],['10','40'])
            self.assertEqual(payload['ks'],[1])
            excluded=pd.read_csv(Path(tmp)/'excluded_embedding_rows.csv')
            self.assertEqual(excluded.id.tolist(),[20,30])
            manifest=json.loads((Path(tmp)/'manifest.json').read_text())
            self.assertEqual(manifest['n_input_rows'],4); self.assertEqual(manifest['n_rows'],2)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                audit_dataframe(df,tmp,features=self.features(),embeddings=a,neighbors=np.array([[1],[0],[3],[2]]),config=self.config(drop_invalid_embeddings=True))

    def test_explorer_row_and_byte_limits_are_visible_not_hidden_sampling(self):
        for cfg in [self.config(explorer_max_rows=3),self.config(explorer_max_bytes=10),self.config(explorer=False)]:
            with tempfile.TemporaryDirectory() as tmp:
                r=audit_dataframe(self.sample(),tmp,features={'score':'continuous'},config=cfg)
                payload=json.loads(r['explorer_data'].read_text())
                self.assertFalse(payload['available']); self.assertTrue(payload['reason'])
                self.assertTrue(r['report'].exists())

    def test_cli_comparison_reordered_ids_and_invalid_rows(self):
        df=self.sample(); z=np.stack(df.embedding); z[1,0]=np.nan; df['embedding']=list(z)
        b=np.column_stack([np.nan_to_num(z),np.ones(len(df))])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); df.to_parquet(root/'posts.parquet',index=False)
            np.save(root/'b.npy',b[::-1],allow_pickle=False)
            np.save(root/'ids.npy',df.id.astype(str).to_numpy(dtype=str)[::-1],allow_pickle=False)
            (root/'config.json').write_text(json.dumps(self.config(features={'category':'categorical'})))
            command=[sys.executable,'-m','embedding_feature_audit.dataframe_cli','--dataframe',str(root/'posts.parquet'),
                     '--config',str(root/'config.json'),'--compare-embeddings',str(root/'b.npy'),
                     '--compare-embedding-ids',str(root/'ids.npy'),'--drop-invalid-embeddings','--out',str(root/'audit')]
            done=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            payload=json.loads((root/'audit/explorer_data.json').read_text())
            self.assertEqual(payload['ids'],['10','30','40'])
            self.assertTrue(payload['comparison'])
            command.remove('--compare-embedding-ids'); command.remove(str(root/'ids.npy'))
            done=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(done.returncode,0)
            self.assertIn('--compare-embedding-ids is required',done.stderr)


if __name__=='__main__':
    unittest.main()
