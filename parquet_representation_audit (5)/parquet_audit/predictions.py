"""ID-aligned external predictions and explicitly ordered probability metrics."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score,average_precision_score
from .semantics import normalize_series, typed_key,is_null
from .ingest import canonical_identity


def attach_predictions(frame: pd.DataFrame,schema: dict,cfg: dict) -> tuple[pd.DataFrame,dict]:
    columns={n:s for n,s in schema['columns'].items() if s['role'] in ['prediction','probabilities']}
    idcols=schema['dataset'].get('identity_columns',[n for n,s in schema['columns'].items() if s['role']=='id'])
    for asset in schema.get('prediction_assets',[]):
        file=pq.ParquetFile(asset['path'])
        if file.metadata.num_rows>cfg['limits']['max_external_ids']: raise ValueError('External prediction ID budget exceeded')
        seen=set();wanted={k:i for i,k in enumerate(frame['_audit_item_identity'])};collected={n:[None]*len(frame) for n in asset['columns']}
        for batch in file.iter_batches(batch_size=cfg['source']['batch_rows'],columns=[asset['id_column']]+list(asset['columns'])):
            if batch.nbytes>cfg['limits']['max_source_batch_bytes']: raise ValueError('External prediction batch byte limit exceeded')
            rows=batch.to_pylist()
            for row in rows:
                key=canonical_identity([row[asset['id_column']]],[schema['columns'][idcols[0]]])
                if key in seen: raise ValueError('Duplicate external prediction IDs')
                seen.add(key)
                if key in wanted:
                    for n in collected: collected[n][wanted[key]]=row[n]
        for n,values in collected.items(): frame[n]=pd.Series(values,dtype=object)
        columns.update(asset['columns'])
    for name,spec in columns.items():
        if spec['role']=='prediction':
            target=schema['columns'][spec['target']]
            frame[name]=normalize_series(frame[name],{**target,'missing_values':spec.get('missing_values',target.get('missing_values',[]))})
    return frame,columns


def probability_metrics(truth: pd.Series,probabilities: pd.Series,target_spec: dict,spec: dict) -> dict:
    typ='string' if target_spec['kind']=='multilabel' else 'boolean' if target_spec['kind']=='boolean' else target_spec.get('id_type','string')
    labels=[typed_key(v,typ) for v in spec['label_order']];rows=[];vectors=[]
    for i,value in enumerate(probabilities):
        if is_null(value): continue
        a=np.asarray(value)
        if a.ndim!=1 or len(a)!=len(labels) or not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all() or (a<0).any() or (a>1).any(): raise ValueError(f'Probability row {i} violates numeric length/range contract')
        if target_spec['kind']!='multilabel' and not np.isclose(a.sum(),1,atol=1e-6): raise ValueError('Categorical probabilities must sum to one')
        if not is_null(truth.iloc[i]): rows.append(i);vectors.append(a)
    if not rows: return {'probability_n_matched':0,'probability_status':'unavailable'}
    p=np.stack(vectors);values=truth.iloc[rows].tolist()
    if target_spec['kind']=='multilabel': binary=np.array([[int(label in tags) for label in labels] for tags in values])
    else:
        outside=set(values)-set(labels)
        if outside: raise ValueError(f'Known truth absent from probability label_order: {outside}')
        binary=np.array([[int(label==value) for label in labels] for value in values])
    output={'probability_n_matched':len(rows),'probability_label_order':labels,'probability_scope':'Matched known truth and supplied scores; no accuracy on missing truth'}
    if target_spec['kind']!='multilabel': output['log_loss']=float(-np.log(np.clip(p[np.arange(len(rows)),[labels.index(v) for v in values]],1e-15,1)).mean())
    output['mean_binary_brier']=float(np.mean((binary-p)**2))
    per=[]
    for i,label in enumerate(labels):
        b=binary[:,i];score=p[:,i]
        per.append({'label':label,'positive_support':int(b.sum()),'roc_auc':float(roc_auc_score(b,score)) if len(np.unique(b))==2 else None,'average_precision':float(average_precision_score(b,score)) if b.sum() else None})
    output['probability_per_label']=per
    return output
