"""Two unrelated synthetic domains. The generated data are illustrative, never real measurements."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def write(value,path):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def configuration(title):
    return {'schema_version':1,'mode':'embedding','seed':19,
      'sample':{'strategy':'uniform_rows','max_rows':120},
      'source':{'scope':'full','batch_rows':64,'max_vocabulary':10000,'quantile_sample_size':1000,'max_feature_columns':100},
      'limits':{'max_evaluation_rows':5000,'max_embedding_bytes':512000000,'max_exact_pairs':12500000,'max_dense_feature_bytes':64000000,'max_external_ids':10000000,'external_batch_rows':128,'max_source_batch_bytes':128000000,'max_source_batch_cells':1000000,'max_work_units':200000000},
      'geometry':{'enabled':True,'metric':'cosine','ks':[3,8,20],'invalid_embeddings':'exclude','reference':'evaluation','distance_cap':100,'distance_pairs':2000},
      'nulls':{'permutations':3,'modes':['permutation'],'correction':'bh','testing_family':'all_features_scales_modes_per_metric'},
      'probes':{'enabled':True,'models':['baseline','linear'],'max_labels':6,'trees':40,'regularization_grid':[1.]},
      'normalization':{'enabled':True,'max_rows':5000,'regularization_grid':[1.]},
      'topology':{'enabled':True,'spaces':['raw_euclidean','unit_chordal'],'sample_size':32,'repetitions':2,'permutations':3,'h1':True,'h1_sample_size':12,'max_simplices':5000,'quantiles':[0,.05,.2,.5,1.]},
      'missingness':{'enabled':True,'features':['department'],'fractions':[.2],'modes':['random']},
      'report':{'title':title,'row_explorer':True,'max_rows':500,'max_bytes':8000000,'pca':True,'overview':{'min_known':15,'min_test':10,'min_known_neighbor_share':.5,'trusted_ks':[3,8,20]}},
      'comparison':{'bootstraps':30},'output_policy':'fresh','verbose':True}


def generate():
    rng=np.random.default_rng(19);n=160;group=np.arange(n)//4;department=np.array(['kitchen','outdoors','books'])[np.arange(n)%3]
    embedding=rng.normal(scale=.2,size=(n,8));embedding[:,:3]+=np.eye(3)[np.arange(n)%3]*2;embedding[7]=np.nan
    basket=[None if i%19==0 else [] if i%7==0 else ['gift','seasonal'] if i%3==0 else ['daily'] for i in range(n)]
    frame=pd.DataFrame({'order_key':np.arange(9007199254741099,9007199254741099+n,dtype='int64'),'encoded_basket':list(embedding),
        'department':pd.Categorical(department),'partition_code':np.where(np.arange(n)%17==0,-1,np.arange(n)%3),
        'units_sold':np.arange(n)%8,'net_margin':rng.normal(5,2,n),'returned':np.arange(n)%5==0,
        'priority':np.array(['low','medium','high'])[np.arange(n)%3],'basket_labels':basket,
        'purchased_time':pd.date_range('2024-01-01',periods=n,freq='7D',tz='UTC'),
        'customer_group':['customer-'+str(g) for g in group],'assignment':np.where(group%4==0,'holdout','fit'),
        'region':np.where(group%2==0,'north','south'),'product_note':['Sample product '+str(i) for i in range(n)],
        'predicted_department':department.copy(),'department_scores':[list(np.eye(3)[i%3]*.8+.2/3) for i in range(n)],
        'source_link':['https://example.invalid/order/'+str(i) for i in range(n)],'payload':[{'nested':[1,2]}]*n})
    retail=ROOT/'retail';retail.mkdir(exist_ok=True);frame.to_parquet(retail/'orders.parquet',index=False,row_group_size=40)
    columns={'order_key':{'role':'id','id_type':'integer','canonicalization':'lossless_string'},'encoded_basket':{'role':'embedding','expected_dimension':8,'format':'list'},
      'department':{'role':'feature','kind':'categorical','id_type':'string','description':'Observed product department'},
      'partition_code':{'role':'feature','kind':'cluster_label','id_type':'integer','missing_values':[-1],'metadata_path':'metadata/partition.json'},
      'units_sold':{'role':'feature','kind':'count','units':'items'},'net_margin':{'role':'feature','kind':'numeric','units':'USD'},
      'returned':{'role':'feature','kind':'boolean'},'priority':{'role':'feature','kind':'ordinal','order':['low','medium','high'],'derive':[{'name':'priority_value','operation':'ordinal_value','kind':'numeric','mapping':{'low':0,'medium':1,'high':2},'reasoning':'Explicit equal step coding for this synthetic operational urgency example','units':'priority steps'}]},
      'basket_labels':{'role':'feature','kind':'multilabel','encoding':'list','null_is_unknown':True,'empty_is_unknown':False,'min_count':2,'max_labels':8},
      'purchased_time':{'role':'feature','kind':'datetime','timezone':'UTC','derive':[{'name':'purchase_year','operation':'year','kind':'numeric','units':'calendar year'}]},
      'customer_group':{'role':'group'},'assignment':{'role':'split','values':{'train':'fit','test':'holdout'}},'region':{'role':'stratum'},
      'product_note':{'role':'feature','kind':'text','derive':[{'name':'note_length','operation':'text_length','kind':'count','units':'characters'}]},
      'predicted_department':{'role':'prediction','target':'department','pipeline':'supplied_classifier'},
      'department_scores':{'role':'probabilities','target':'department','pipeline':'supplied_classifier','label_order':['kitchen','outdoors','books'],'encoding':'list'},
      'source_link':{'role':'display'},'payload':{'role':'ignore','reason':'Uninterpreted nested payload'}}
    schema={'schema_version':1,'status':'validated','dataset':{'name':'Synthetic retail orders','path':'orders.parquet','unmapped_columns':'error'},'columns':columns,
      'embeddings':[{'name':'Basket encoder','column':'encoded_basket','main_normalization':'none'}],
      'prediction_pipelines':{'supplied_classifier':{'uses_embedding':'unknown','description':'Synthetic provided predictions; illustrative only','provenance':{'synthetic':True}}},'weights':'unweighted'}
    write(schema,retail/'dataset_schema.json');cfg=configuration('Retail representation audit · synthetic');cfg['probes'].update(group_column='customer_group',split_column='assignment');cfg['nulls'].update(modes=['permutation','stratified'],stratum_column='region');write(cfg,retail/'audit_config.json')
    write({'schema_version':1,'name':'partition_code','id_type':'integer','origin':'external_assignment','description':'Synthetic assignment IDs, no arithmetic order','source_fields':[],'excluded_from_coverage':[-1],
      'labels':[{'id':-1,'name':'Unassigned','criteria':'Synthetic missing assignment'},{'id':0,'name':'Segment A','criteria':'Declared assignment code 0'},{'id':1,'name':'Segment A','criteria':'Distinct computational ID 1 despite duplicate display name'},{'id':2,'name':'Segment C','criteria':'Declared assignment code 2'}]},retail/'metadata'/'partition.json')
    source=ROOT/'source_only';source.mkdir(exist_ok=True)
    sc=copy.deepcopy(schema);sc['dataset']['path']='../retail/orders.parquet';sc['embeddings']=[];sc['columns']['partition_code']['metadata_path']='../retail/metadata/partition.json';write(sc,source/'dataset_schema.json')
    so=copy.deepcopy(cfg);so['mode']='source_only';so['report']['title']='Retail source audit · synthetic'
    for section in ['geometry','probes','normalization','topology','missingness']: so[section]['enabled']=False
    write(so,source/'audit_config.json')
    telemetry=ROOT/'telemetry';telemetry.mkdir(exist_ok=True);m=144;keys=np.array([f'{i:06d}' for i in range(m)]);fault=np.array(['nominal','warning','fault'])[np.arange(m)%3]
    a=rng.normal(scale=.2,size=(m,6));a[:,:3]+=np.eye(3)[np.arange(m)%3];b=np.column_stack([a,rng.normal(scale=.1,size=(m,4))]);a[11]=0;b[22]=np.nan
    pd.DataFrame({'reading_ref':keys,'condition':fault,'pulse_count':np.arange(m)%9,'temperature':20+a[:,0]*4,
      'alarms':['' if i%11==0 else 'hot|vibration' if i%3==0 else 'pressure' for i in range(m)],
      'calibration_ok':np.arange(m)%4!=0,'station_id':['station-'+str(i//3) for i in range(m)],'engineer_note':['Voltage normal; no automatic tag parsing']*m}).to_parquet(telemetry/'readings.parquet',index=False,row_group_size=36)
    order=rng.permutation(m);np.save(telemetry/'encoder_a.npy',a[order]);np.save(telemetry/'encoder_a_ids.npy',keys[order]);np.save(telemetry/'encoder_b.npy',b[order[::-1]]);np.save(telemetry/'encoder_b_ids.npy',keys[order[::-1]])
    cols={'reading_ref':{'role':'id','id_type':'string','canonicalization':'lossless_string'},'condition':{'role':'feature','kind':'categorical','id_type':'string'},'pulse_count':{'role':'feature','kind':'count','units':'pulses'},'temperature':{'role':'feature','kind':'numeric','units':'degrees C','invalid_values':'unknown'},
      'alarms':{'role':'feature','kind':'multilabel','encoding':'delimited','delimiter':'|','case':'preserve','empty_is_unknown':False,'min_count':2,'max_labels':6},'calibration_ok':{'role':'feature','kind':'boolean'},'station_id':{'role':'group'},'engineer_note':{'role':'feature','kind':'text'}}
    ts={'schema_version':1,'status':'validated','dataset':{'name':'Synthetic instrument readings','path':'readings.parquet','unmapped_columns':'error'},'columns':cols,
      'embeddings':[{'name':'Instrument A','matrix_path':'encoder_a.npy','ids_path':'encoder_a_ids.npy','expected_dimension':6,'main_normalization':'none'}, {'name':'Instrument B','matrix_path':'encoder_b.npy','ids_path':'encoder_b_ids.npy','expected_dimension':10,'main_normalization':'none'}]}
    write(ts,telemetry/'dataset_schema.json');tc=configuration('Telemetry A/B representation audit · synthetic');tc['probes']['group_column']='station_id';tc['missingness']['features']=['condition'];write(tc,telemetry/'audit_config.json')
    for folder in [retail,source,telemetry]:
        (folder/'run_audit.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd "$(dirname "$0")"\nparquet-representation-audit validate --schema dataset_schema.json --config audit_config.json\nparquet-representation-audit run --schema dataset_schema.json --config audit_config.json --out "${1:-audit_output}"\n')
        (folder/'PREPARATION_NOTES.md').write_text('# Prepared synthetic bundle\n\nAll row/column semantics are explicitly declared in dataset_schema.json. This is generated demonstration data. Missing embeddings are intentionally present; invalid_embeddings=exclude records them. Scalar feature missingness does not remove other evaluation rows. No real-world performance claims apply.\n\nIDs, category meanings, count units, tag delimiters and known-empty policy are documented by the JSON. Retail uses large integer IDs, embedded list vectors, explicit group-safe splits and Arrow dictionary categories. Telemetry uses leading-zero strings, pipe-delimited tags and shuffled external A/B IDs. Source-only disables all embedding analyses.\n')
        (folder/'VALIDATION.md').write_text('# Local validation\n\nRun `parquet-representation-audit validate --schema dataset_schema.json --config audit_config.json`, then `bash run_audit.sh fresh_output`. See the root VALIDATION.md for checks actually performed on the delivered bundle.\n')
    print('Generated retail, telemetry A/B and source-only dataset bundles.')


if __name__=='__main__': generate()
