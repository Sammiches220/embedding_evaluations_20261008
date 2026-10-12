"""Cross-language numerical verification on both models and explicit row subsets."""
import json, os, subprocess
from pathlib import Path
import numpy as np
from embedding_feature_audit.geometry import purity_lift, continuous_geometry
from embedding_feature_audit.feature_overview import local_entropy
from embedding_feature_audit.statistics import prediction_metrics
root=Path(__file__).resolve().parents[1]
payload=json.loads((root/'demo_telemetry_ab/explorer_data.json').read_text())
tasks=[];expected=[]
for name,record in payload['overview']['features'].items():
    feature=payload['features'][name];labels=record['labels'];kind=feature['kind']
    if kind not in ['categorical','continuous','multilabel']: continue
    valid=np.array([v is not None for v in feature['values']])
    if kind=='multilabel': values=np.array([[int(label in (v or [])) for label in labels] for v in feature['values']],dtype='uint8')
    elif kind=='continuous': values=np.array([np.nan if v is None else v for v in feature['values']],float)
    else: values=np.array(['?' if v is None else str(v) for v in feature['values']])
    global_source={'variance':np.var(values[valid]) if kind=='continuous' else None,'mean':np.mean(values[valid]) if kind=='continuous' else None}
    baseline=np.array([np.mean(values[valid,c]) if kind=='multilabel' else np.mean(values[valid]==label) for c,label in enumerate(labels)])
    for which in ['A','B']:
        graph=np.array(payload['graph'] if which=='A' else payload['comparison']['graph'])
        ent=local_entropy(values,valid,labels,kind,graph,payload['ks'])
        for selected in [list(range(len(valid))),list(range(0,len(valid),3))]:
            mask=np.zeros(len(valid),bool);mask[selected]=True
            table=continuous_geometry(values,valid,graph,payload['ks'],mask) if kind=='continuous' else purity_lift(values,valid,labels,kind,graph,payload['ks'],anchor_mask=mask)[0]
            for col,k in enumerate(payload['ks']):
                row=table[table.k==k].iloc[0].to_dict()
                if kind!='continuous':
                    e=ent[selected,col];e=e[np.isfinite(e)];row['entropy']=float(e.mean()) if len(e) else None
                keys=['known_neighbor_share']+(['mean_neighbor_abs_difference','mean_local_variance','variance_ratio_to_global','morans_i'] if kind=='continuous' else ['micro_purity','macro_purity','micro_lift','macro_lift','entropy'])
                tasks.append(dict(feature=feature,labels=labels,graph=graph.tolist(),rows=selected,k=k,source=global_source,baseline=baseline.tolist()))
                expected.append({key:row.get(key) for key in keys})
script="const c=require(process.argv[1]);let s='';process.stdin.on('data',v=>s+=v);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(s).map(t=>c.geometry(t.feature,t.labels,t.graph,t.rows,t.k,t.source,t.baseline)))));"
run=subprocess.run([os.environ.get('CODEX_PRIMARY_RUNTIME_NODE','node'),'-e',script,str(root/'embedding_feature_audit/assets/explorer.js')],input=json.dumps(tasks),capture_output=True,text=True,check=True)
actual=json.loads(run.stdout);checks=0
for want,got in zip(expected,actual):
    for key,value in want.items():
        if value is None or not np.isfinite(value): assert got[key] is None,(key,value,got[key])
        else: assert np.isclose(value,got[key],atol=1e-10,rtol=1e-10),(key,value,got[key])
        checks+=1
# Prediction scorer agrees with all saved linear and baseline summaries, A and B.
tasks=[];expected=[]
for record in payload['overview']['features'].values():
    for which in ['A','B']:
        summary=record['probes'] if which=='A' else record['comparison_probes']
        preds=record.get('probe_predictions',[]) if which=='A' else record.get('comparison_probe_predictions',[])
        for row in summary:
            tasks.append(dict(records=preds,model=row['model'],kind=record['kind']))
            expected.append({k:row[k] for k in ['balanced_accuracy','macro_f1','mae','r2'] if k in row})
script="const c=require(process.argv[1]);let s='';process.stdin.on('data',v=>s+=v);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(s).map(t=>c.probe(t.records,null,t.kind,t.model)))));"
run=subprocess.run([os.environ.get('CODEX_PRIMARY_RUNTIME_NODE','node'),'-e',script,str(root/'embedding_feature_audit/assets/explorer.js')],input=json.dumps(tasks),capture_output=True,text=True,check=True)
for want,got in zip(expected,json.loads(run.stdout)):
    for key,value in want.items():
        if value is None: assert got[key] is None
        else: assert np.isclose(value,got[key],atol=1e-10,rtol=1e-10),(key,value,got[key])
        checks+=1
print(f'{checks} JS/Python numerical comparisons passed across A/B, full/subset geometry, entropy and probes.')
