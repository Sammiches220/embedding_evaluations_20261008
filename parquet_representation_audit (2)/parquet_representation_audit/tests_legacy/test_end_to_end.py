import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from embedding_feature_audit import audit_dataframe, export_dataframe
from embedding_feature_audit.dataframe import load_dataframe
from embedding_feature_audit.experiments import missingness_experiments
from embedding_feature_audit.geometry import make_neighbors


class WorkflowTests(unittest.TestCase):
    def sample(self):
        rng=np.random.default_rng(1); n=40
        return pd.DataFrame({'id':range(n),'rating':['a']*20+['b']*20,
            'tags':['one two']*20+['three']*20,'score':np.arange(n,dtype=float),
            'embedding':list(rng.normal(size=(n,4))), 'media_asset':[{'nested':1}]*n})

    def test_parquet_export_and_cli_embedded_and_separate_embeddings(self):
        import subprocess,sys
        df=self.sample()
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); export_dataframe(df,root/'input')
            restored=load_dataframe(root/'input/features.parquet')
            self.assertNotIn('media_asset',restored)
            config={'features':{'rating':'categorical','score':'continuous'},'ks':[2,5],
                'permutations':2,'probe_features':['rating','score'],'trees':5,
                'missingness_features':['score'],'missingness_fractions':[.5],
                'null_modes':['permutation','prevalence-random'],'verbose':False}
            (root/'config.json').write_text(json.dumps(config))
            cmd=[sys.executable,'-m','embedding_feature_audit.dataframe_cli',
                '--dataframe',str(root/'input/features.parquet'),'--embeddings',str(root/'input/embeddings.npy'),
                '--embedding-ids',str(root/'input/embedding_ids.npy'),
                '--config',str(root/'config.json'),'--out',str(root/'report')]
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((root/'report/report.html').exists())
            self.assertTrue((root/'report/features/002/distance_relationship.json').exists())
            self.assertTrue((root/'report/features/001/tables/probes.csv').exists())
            # Embedded-array Parquet route should load as arrays, not text.
            df.drop(columns='media_asset').to_parquet(root/'embedded.parquet',index=False)
            result=subprocess.run([sys.executable,'-m','embedding_feature_audit.dataframe_cli',
                '--dataframe',str(root/'embedded.parquet'),'--config',str(root/'config.json'),
                '--out',str(root/'report2')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            # A shuffled ID file must fail before analysis.
            np.save(root/'bad_ids.npy',np.arange(39,-1,-1).astype(str))
            cmd[cmd.index('--embedding-ids')+1]=str(root/'bad_ids.npy')
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('do not match',result.stderr)

    def test_hidden_targets_not_in_training_callback_and_groups_disjoint(self):
        z=np.random.default_rng(1).normal(size=(40,4)); y=np.arange(40,dtype=float)
        valid=np.ones(40,dtype=bool); graph,_=make_neighbors(z,3,'euclidean')
        groups=np.repeat(np.arange(20),2); calls=[]
        def predictor(z,y,train,hidden,kind):
            self.assertFalse(set(train)&set(hidden))
            self.assertFalse(set(groups[train])&set(groups[hidden]))
            calls.append((train,hidden))
            return np.full(len(hidden),y[train].mean())
        a,b=missingness_experiments(z,y,valid,[],'continuous',graph,[2,3],[.1,.5],
                                   ['random','rare-first'],predictor=predictor,groups=groups)
        self.assertEqual(len(calls),4); self.assertEqual(len(a),4); self.assertEqual(len(b),8)

    def test_multilabel_probes_predictions_and_priority_missingness(self):
        df=self.sample(); df['priority']=np.arange(len(df))
        pred=pd.DataFrame({'id':df.id.astype(str),'predicted_tags':df.tags})
        with tempfile.TemporaryDirectory() as root:
            r=audit_dataframe(df,Path(root)/'audit',features={'tags':{'kind':'multilabel','min_count':1}},predictions=pred,
                config={'ks':[2,5],'permutations':2,'probe_features':['tags'],'trees':5,
                    'predictions':{'tags':'predicted_tags'},'missingness_features':['tags'],
                    'missingness_fractions':[.1], 'missingness_modes':['priority-first'],
                    'missingness_priority_column':'priority','verbose':False})
            metrics=json.loads((r['out_dir']/'features/001/prediction_metrics.json').read_text())
            self.assertEqual(metrics['subset_accuracy'],1.)
            self.assertEqual(metrics['n_compared'],len(df))
            self.assertTrue((r['out_dir']/'features/001/tables/probe_per_label.csv').is_file())
            with self.assertRaises(ValueError):
                audit_dataframe(df,r['out_dir'],features={'rating':'categorical'})


if __name__=='__main__': unittest.main()
