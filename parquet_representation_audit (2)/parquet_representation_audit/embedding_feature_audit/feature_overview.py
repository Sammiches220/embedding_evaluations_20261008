"""Feature-level measurements: fixed vocabularies, batched local entropy, explicit scope."""
import numpy as np
import pandas as pd
from .geometry import membership, purity_lift, continuous_geometry


def entropy_probabilities(p, binary=False):
    p = np.asarray(p, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = np.where(p > 0, -p*np.log2(p), 0.)
        if binary:
            h += np.where(p < 1, -(1-p)*np.log2(1-p), 0.)
    return h


def local_entropy(values, valid, names, kind, graph, ks, batch_size=256):
    """N×len(ks) normalized plug-in entropy; unknown anchors/neighbors are excluded.

    Categories: H/log2(L), using the fixed full-run vocabulary L. Tags: mean
    Bernoulli entropy over audited tags (0..1 bits). This is not tag-occurrence
    entropy in the existing source summary. Numeric fields have no discretized
    entropy: local/global variance remains their measure of diversity.
    Temporary memory is O(batch*K + batch*L), rather than N*K*L.
    """
    out = np.full((len(valid),len(ks)), np.nan)
    if kind == 'continuous' or not names:
        return out
    members = membership(values,names,kind)
    divisor = np.log2(len(names)) if kind == 'categorical' and len(names)>1 else 1.
    for start in range(0,len(valid),batch_size):
        end = min(start+batch_size,len(valid))
        for column,k in enumerate(ks):
            idx = graph[start:end,:k]
            known = valid[idx]
            denom = known.sum(axis=1)
            h = np.zeros(end-start)
            for label in range(len(names)):
                count = (members[idx,label]*known).sum(axis=1)
                p = np.divide(count,denom,out=np.zeros(end-start),where=denom>0)
                h += entropy_probabilities(p,kind=='multilabel')
            h /= len(names) if kind=='multilabel' else divisor
            usable = valid[start:end] & (denom>0)
            out[start:end,column] = np.where(usable,h,np.nan)
    return out


def overview_settings(ks,cfg):
    """Validate descriptive assumptions before graph building or model fitting."""
    settings = dict(min_known=30,min_test=30,min_known_neighbor_share=.5,
                    neutral_delta=.02,probe_neutral_delta=.02,trusted_ks=ks,metric_thresholds={})
    if not isinstance(cfg.get('overview',{}),dict):
        raise ValueError('overview must be a configuration object')
    settings.update(cfg.get('overview',{}))
    for name in ['min_known','min_test']:
        if not isinstance(settings[name],int) or settings[name]<1:
            raise ValueError(f'overview.{name} must be a positive integer')
    for name in ['min_known_neighbor_share','neutral_delta','probe_neutral_delta']:
        if not isinstance(settings[name],(int,float)) or not 0<=settings[name]<=1:
            raise ValueError(f'overview.{name} must be in [0,1]')
    thresholds=settings['metric_thresholds']
    if not isinstance(thresholds,dict) or any(not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 for v in thresholds.values()):
        raise ValueError('overview.metric_thresholds must map metric names to finite nonnegative margins')
    trusted = settings['trusted_ks']
    if not isinstance(trusted,list) or any(k not in cfg.get('ks',ks) for k in trusted):
        raise ValueError('overview.trusted_ks must be a list of evaluated k values')
    settings['trusted_ks']=[k for k in trusted if k in ks]
    return settings


def feature_overview(encoded, results, graph, ks, cfg, comparison=None, include_rows=True):
    """Small aggregate overview always available; per-anchor data only in bounded runs."""
    settings=overview_settings(ks,cfg)
    features = {}; entropy_tables = {}
    for name,data in encoded.items():
        result = results[name]
        record = dict(kind=data['kind'],labels=data['names'],source=result.get('source',{}),
            geometry=result.get('geometry',pd.DataFrame()).to_dict('records'),
            comparison_geometry=result.get('comparison_geometry',pd.DataFrame()).to_dict('records'),
            probes=result.get('probes',pd.DataFrame()).to_dict('records'),
            comparison_probes=result.get('comparison_probes',pd.DataFrame()).to_dict('records'),
            probe_status=result.get('probe_status'),probe_labels=result.get('probe_labels',[]),
            external=result.get('prediction_metrics'),paired=result.get('paired_comparison',pd.DataFrame()).to_dict('records'),
            folder=str(result['folder'].name),status=result['status'],entropy=[],comparison_entropy=[],
            paired_geometry_a=[],paired_geometry_b=[],paired_entropy_delta=[])
        known=data['values'][data['valid']]
        if data['kind']!='continuous' and len(known) and data['names']:
            p=membership(known,data['names'],data['kind']).mean(axis=0)
            h=entropy_probabilities(p,data['kind']=='multilabel').sum()
            record['diversity']=float(h/len(p) if data['kind']=='multilabel' else h/(np.log2(len(p)) if len(p)>1 else 1.))
        else:
            record['diversity']=None
        tables=[]; entropies={}
        if data['kind']!='continuous' and name not in cfg.get('source_only_targets',[]):
            for key,g in [('entropy',graph),('comparison_entropy',comparison['graph'] if comparison else None)]:
                if g is None: continue
                local = local_entropy(data['values'],data['valid'],data['names'],data['kind'],g,ks)
                entropies[key]=local
                for j,k in enumerate(ks):
                    values=local[:,j]; known=np.isfinite(values)
                    row=dict(k=k,normalized_entropy=float(values[known].mean()) if known.any() else None,
                             n_anchors=int(known.sum()),model='B' if key.startswith('comparison') else 'A')
                    record[key].append(row); tables.append(row)
                if include_rows: record[key+'_rows']=local
        if include_rows:
            record['probe_predictions']=result.get('probe_predictions',pd.DataFrame()).to_dict('records')
            record['comparison_probe_predictions']=result.get('comparison_probe_predictions',pd.DataFrame()).to_dict('records')
        if comparison is not None and name not in cfg.get('source_only_targets',[]):
            for k in ks:
                mask=data['valid'] & data['valid'][graph[:,:k]].any(axis=1) & data['valid'][comparison['graph'][:,:k]].any(axis=1)
                for key,g in [('paired_geometry_a',graph),('paired_geometry_b',comparison['graph'])]:
                    if data['kind']=='continuous':
                        table=continuous_geometry(data['values'],data['valid'],g,[k],mask)
                    elif data['names']:
                        label_table=result.get('per_label',pd.DataFrame())
                        baseline=dict(zip(label_table.label,label_table.baseline_prevalence)) if len(label_table) else None
                        table,_=purity_lift(data['values'],data['valid'],data['names'],data['kind'],g,[k],baseline,mask)
                    else: table=pd.DataFrame()
                    record[key].extend(table.to_dict('records'))
                if data['kind']!='continuous':
                    # Fixed shared anchors for entropy too, rather than subtracting different pools.
                    ea=entropies['entropy'][:,ks.index(k)]
                    eb=entropies['comparison_entropy'][:,ks.index(k)]
                    shared=mask & np.isfinite(ea) & np.isfinite(eb)
                    record['paired_entropy_delta'].append(dict(k=k,n_anchors=int(shared.sum()),delta=float((eb[shared]-ea[shared]).mean()) if shared.any() else None))
        features[name]=record; entropy_tables[name]=pd.DataFrame(tables)
    return dict(ks=ks,settings=settings,features=features,embedding_name=cfg.get('embedding_name','Embedding A'),comparison_name=cfg.get('comparison_name','Embedding B') if comparison else None,scope='Same full-run vocabulary and candidate pool for every subset; fixed held-out probes are never refit on selection.'),entropy_tables
