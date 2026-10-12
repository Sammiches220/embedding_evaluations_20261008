import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from embedding_feature_audit import audit_dataframe
from embedding_feature_audit.cluster_metadata import (read_cluster_metadata,configure_cluster_metadata,
    annotate_labels,resolve_config_metadata_paths)
from embedding_feature_audit.dataframe import encode_feature

METADATA=Path(__file__).resolve().parent.parent/'examples/cluster_metadata'


class MetadataTests(unittest.TestCase):
    def sample(self):
        n=24
        return pd.DataFrame({'id':np.arange(n),'embedding':list(np.random.default_rng(0).normal(size=(n,4))),
            'artist_avg_fav_count_cluster':np.tile([0,1,2,3],6),
            'artist_avg_score_cluster':np.tile([0,1,2,3],6),
            'is_pending_cluster':np.tile([0,1,2],8)})

    def test_metadata_missing_policy_differs_for_artists_and_flags(self):
        specs={name:{'kind':'categorical'} for name in self.sample() if name.endswith('_cluster')}
        specs,meta=configure_cluster_metadata(specs,{'cluster_metadata_dir':METADATA})
        self.assertEqual(specs['artist_avg_fav_count_cluster']['missing_values'],[0])
        self.assertEqual(specs['is_pending_cluster']['missing_values'],[2])
        _,known,names=encode_feature(pd.Series([0.,1.,2.,None]),specs['is_pending_cluster'])
        self.assertEqual(known.tolist(),[True,True,False,False]); self.assertEqual(names,['0','1'])
        self.assertEqual(meta['is_pending_cluster']['labels'][0]['name'],'false')

    def test_missing_override_and_explicit_file(self):
        specs={'artist_avg_fav_count_cluster':{'kind':'categorical','missing_values':[]}}
        specs,meta=configure_cluster_metadata(specs,{'cluster_metadata':{
            'artist_avg_fav_count_cluster':METADATA/'artist_avg_fav_count.json'}})
        self.assertEqual(specs['artist_avg_fav_count_cluster']['missing_values'],[])
        _,valid,_=encode_feature(pd.Series([0,1]),specs['artist_avg_fav_count_cluster'])
        self.assertTrue(valid.all())

    def test_duplicate_names_do_not_collapse_ids_and_unknown_stays_visible(self):
        meta={'name':'test','labels':[{'id':1,'name':'same'},{'id':2,'name':'same'}]}
        table=annotate_labels(pd.DataFrame({'label':['1','2','9'],'lift':[2.,3.,1.]}),meta)
        self.assertEqual(table.label.tolist(),['1','2','9'])
        self.assertEqual(table.label_name.tolist(),['same','same','unmapped'])
        self.assertEqual(table.label_metadata_known.tolist(),[True,True,False])
        self.assertEqual(table.lift.tolist(),[2.,3.,1.])
        self.assertNotEqual(table.label_display[0],table.label_display[1])

    def test_invalid_or_duplicate_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json'
            p.write_text(json.dumps({'name':'test','labels':[{'id':1,'name':'a'},{'id':1.0,'name':'b'}]}))
            with self.assertRaises(ValueError): read_cluster_metadata(p)
            data=read_cluster_metadata(METADATA/'is_pending.json')
            for name in ['a.json','b.json']: (Path(tmp)/name).write_text(json.dumps(data))
            p.unlink()
            with self.assertRaises(ValueError): configure_cluster_metadata({'is_pending_cluster':{'kind':'categorical'}},{'cluster_metadata_dir':tmp})

    def test_report_table_names_criteria_and_prediction_id_normalization(self):
        df=self.sample()
        pred=pd.DataFrame({'id':df.id,'predicted':df.is_pending_cluster.astype(float)})
        with tempfile.TemporaryDirectory() as root:
            result=audit_dataframe(df,Path(root)/'audit',predictions=pred,config={
                'cluster_metadata_dir':METADATA,'ks':[2,3],'permutations':2,
                'probe_features':False,'missingness_features':False,
                'predictions':{'is_pending_cluster':'predicted'},'verbose':False})
            labels=result['features']['is_pending_cluster']['per_label']
            self.assertEqual(set(labels.label_name),{'false','true'})
            self.assertNotIn('2',set(labels.label_id))
            f=result['out_dir']/'features/003'
            metrics=json.loads((f/'prediction_metrics.json').read_text())
            self.assertEqual(metrics['n_compared'],16); self.assertEqual(metrics['accuracy'],1.)
            self.assertAlmostEqual(result['features']['artist_avg_fav_count_cluster']['source']['coverage'],.75)
            table=pd.read_csv(f/'tables/labels_metadata.csv')
            self.assertEqual(table.loc[table.label==2,'n_items'].iloc[0],8)
            text=result['report'].read_text()
            self.assertIn('is_pending = false',text)
            self.assertIn('third label is reserved',text)
            self.assertTrue((f/'cluster_metadata.json').exists())

    def test_cli_actual_filename_and_metadata_paths_relative_to_config(self):
        df=self.sample()
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); (root/'data').mkdir(); (root/'settings').mkdir()
            df.to_parquet(root/'data/posts-sample-populated.parquet',index=False)
            config={'cluster_metadata_dir':str(METADATA),'ks':[2,3],'permutations':0,
                'probe_features':False,'missingness_features':False,'verbose':False}
            # Resolve relative explicit paths against config directory, not cwd.
            meta_dir=root/'settings/meta'; meta_dir.mkdir()
            (meta_dir/'is_pending.json').write_text((METADATA/'is_pending.json').read_text())
            config['cluster_metadata_dir']='meta'
            config['cluster_metadata']={'artist_avg_fav_count_cluster':str(METADATA/'artist_avg_fav_count.json')}
            config['features']={name:'categorical' for name in df if name.endswith('_cluster')}
            config_path=root/'settings/run.json'; config_path.write_text(json.dumps(config))
            command=[sys.executable,'-m','embedding_feature_audit.dataframe_cli',
                '--dataframe','data/posts-sample-populated.parquet','--config','settings/run.json','--out','audit']
            done=subprocess.run(command,cwd=root,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            table=pd.read_csv(root/'audit/features/003/tables/source_prevalence.csv',dtype={'label_name':'string'})
            self.assertEqual(set(table.label_name),{'false','true'})
            # CLI directory overrides config-relative automatic discovery.
            command += ['--cluster-metadata-dir',str(METADATA)]
            command[command.index('--out')+1]='audit2'
            done=subprocess.run(command,cwd=root,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)


if __name__=='__main__': unittest.main()
