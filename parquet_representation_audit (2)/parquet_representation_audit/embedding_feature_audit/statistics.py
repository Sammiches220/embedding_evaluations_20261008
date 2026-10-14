"""Source statistics, prediction comparisons, and fixed-graph null models."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                            mean_absolute_error, r2_score, precision_recall_fscore_support)
from .geometry import membership, purity_lift, continuous_geometry


def entropy(counts):
    counts = np.asarray(counts, dtype=float)
    if counts.sum() == 0:
        return 0., 0.
    p = counts[counts > 0] / counts.sum()
    h = -np.sum(p*np.log(p))
    return float(h), float(h / np.log(len(p))) if len(p) > 1 else 0.


def feature_source(series, values, valid, names, kind, spec):
    """Source baseline covers all labels; audited tag subset is disclosed separately."""
    summary = dict(n_items=len(series), n_known=int(valid.sum()),
                   coverage=float(valid.mean()), missing_share=float(1-valid.mean()))
    if kind == 'continuous':
        s = pd.Series(values[valid])
        summary.update(mean=float(s.mean()), std=float(s.std()), variance=float(s.var(ddof=0)),
                       min=float(s.min()), max=float(s.max()))
        summary.update({f'q{q:02d}': float(s.quantile(q/100)) for q in [5,25,50,75,95]})
        return summary, pd.DataFrame(), pd.DataFrame()
    if kind == 'multilabel':
        from .dataframe import tag_sets
        sets, _ = tag_sets(series, spec.get('empty_is_missing', True))
        counts = {}
        for tags, known in zip(sets, valid):
            if known:
                for tag in tags:
                    counts[tag] = counts.get(tag, 0)+1
        labels = sorted(counts, key=lambda tag: (-counts[tag], tag))
        h, hn = entropy(list(counts.values()))
        total = sum(counts.values())
        table = pd.DataFrame([dict(label=tag, count=counts[tag], prevalence=counts[tag]/max(valid.sum(),1),
                   occurrence_share=counts[tag]/max(total,1), audited=tag in names) for tag in labels])
        summary.update(n_labels=len(labels), n_audited_labels=len(names), entropy=h, normalized_entropy=hn,
            largest_label_share=max(counts.values(), default=0)/max(total,1),
            largest_item_prevalence=max(counts.values(), default=0)/max(valid.sum(),1),
            mean_labels_per_known_item=total/max(valid.sum(),1),
            audited_occurrence_share=sum(counts.get(t,0) for t in names)/max(total,1))
        # Co-occurrence only for the explicitly selected label vocabulary.
        m = values[valid].astype(np.int64)
        co = m.T @ m
        rows = [dict(label_a=names[a], label_b=names[b], count=int(co[a,b]))
                for a in range(len(names)) for b in range(a+1,len(names)) if co[a,b]]
        return summary, table, pd.DataFrame(rows, columns=['label_a','label_b','count'])
    counts = np.array([(values[valid] == name).sum() for name in names])
    h, hn = entropy(counts)
    summary.update(n_labels=len(names), n_audited_labels=len(names), entropy=h, normalized_entropy=hn,
                   largest_label_share=float(counts.max()/max(valid.sum(),1)) if len(counts) else 0.)
    table = pd.DataFrame({'label':names,'count':counts,'prevalence':counts/max(valid.sum(),1)})
    return summary, table, pd.DataFrame()


def prediction_metrics(y, yp, kind, names):
    if not len(y):
        return {'n_compared':0}, pd.DataFrame()
    out = {'n_compared':len(y)}
    if kind == 'continuous':
        out.update(r2=float(r2_score(y,yp)) if len(y)>1 and np.var(y)>0 else np.nan,
                   mae=float(mean_absolute_error(y,yp)),
                   spearman=float(spearmanr(y,yp).statistic) if np.unique(y).size>1 and np.unique(yp).size>1 else np.nan)
        return out, pd.DataFrame()
    if kind == 'categorical':
        out.update(accuracy=float(accuracy_score(y,yp)),
            balanced_accuracy=float(np.mean([np.mean(yp[y==v]==v) for v in np.unique(y)])),
            macro_f1=float(f1_score(y,yp,average='macro',zero_division=0)),
            micro_f1=float(f1_score(y,yp,average='micro',zero_division=0)))
        labels = sorted(set(y)|set(yp))
        p,r,f,s = precision_recall_fscore_support(y,yp,labels=labels,zero_division=0)
        tc = np.array([(y==v).sum() for v in labels]); pc = np.array([(yp==v).sum() for v in labels])
    else:
        out.update(subset_accuracy=float(accuracy_score(y,yp)),
            macro_f1=float(f1_score(y,yp,average='macro',zero_division=0)),
            micro_f1=float(f1_score(y,yp,average='micro',zero_division=0)))
        labels = names
        p,r,f,s = precision_recall_fscore_support(y,yp,average=None,zero_division=0)
        tc,pc = y.sum(0),yp.sum(0)
        out['mean_label_cardinality_delta'] = float(yp.sum(1).mean()-y.sum(1).mean())
        out['macro_prevalence_absolute_error'] = float(np.abs(pc-tc).mean()/len(y))
    table = pd.DataFrame(dict(label=labels,precision=p,recall=r,f1=f,support=s,
                             true_prevalence=tc/len(y),predicted_prevalence=pc/len(y)))
    ht,hnt = entropy(tc); hp,hnp = entropy(pc)
    out.update(entropy_delta=hp-ht, normalized_entropy_delta=hnp-hnt,
        largest_label_share_delta=float(pc.max()/pc.sum()-tc.max()/tc.sum()) if pc.sum() and tc.sum() else np.nan,
        n_labels_delta=int((pc>0).sum()-(tc>0).sum()),
        jensen_shannon_divergence=float(jensenshannon(tc/tc.sum(),pc/pc.sum(),base=2)**2)
            if tc.sum() and pc.sum() else np.nan)
    return out, table


def geometry_delta(true, predicted, kind):
    merged = true.merge(predicted,on='k',suffixes=('_true','_pred'))
    metrics = (['micro_purity','macro_purity','micro_lift','macro_lift'] if kind!='continuous'
               else ['mean_neighbor_abs_difference','mean_local_variance','variance_ratio_to_global','morans_i'])
    for metric in metrics:
        merged[metric+'_delta'] = merged[metric+'_pred']-merged[metric+'_true']
        merged[metric+'_ratio'] = merged[metric+'_pred']/merged[metric+'_true'].replace(0,np.nan)
    return merged


def score_geometry(y, valid, names, kind, neighbors, ks, baseline=None):
    if kind == 'continuous':
        return continuous_geometry(y,valid,neighbors,ks), pd.DataFrame()
    return purity_lift(y,valid,names,kind,neighbors,ks,baseline)


def permutation_analysis(y, valid, names, kind, neighbors, ks, observed,
                         repetitions=20, seed=42, mode='permutation', strata=None, baseline=None):
    """Shuffle known rows only; multilabel shuffles preserve complete tag sets."""
    rng = np.random.default_rng(seed)
    known = np.flatnonzero(valid)
    rows=[]
    for repeat in range(repetitions):
        yp = y.copy()
        if mode == 'prevalence-random':
            # Empirical row bootstrap: prevalence in expectation; co-occurrence retained.
            yp[known] = y[rng.choice(known,len(known),replace=True)]
        elif strata is None:
            yp[known] = y[rng.permutation(known)]
        else:
            for group in pd.unique(strata[known]):
                ids = known[strata[known] == group]
                yp[ids] = y[rng.permutation(ids)]
        summary,_ = score_geometry(yp,valid,names,kind,neighbors,ks,baseline)
        summary['repeat']=repeat; summary['null_mode']=mode
        rows.append(summary)
    nulls = pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
    scored = observed.copy()
    metrics = (['micro_purity','macro_purity','micro_lift','macro_lift'] if kind!='continuous'
               else ['mean_neighbor_abs_difference','variance_ratio_to_global','morans_i'])
    for metric in metrics:
        if nulls.empty:
            continue
        stats = nulls.groupby('k')[metric].agg(['mean','std'])
        scored['null_'+metric+'_mean'] = scored.k.map(stats['mean'])
        scored['null_'+metric+'_std'] = scored.k.map(stats['std'])
        scored[metric+'_z'] = (scored[metric]-scored['null_'+metric+'_mean']) / scored['null_'+metric+'_std'].replace(0,np.nan)
        # One-sided empirical tail; low differences/variance are the enriched direction.
        low = metric in ['mean_neighbor_abs_difference','variance_ratio_to_global']
        probabilities=[];effective=[]
        for _,row in scored.iterrows():
            draws=nulls.loc[nulls.k==row.k,metric].to_numpy()
            draws=draws[np.isfinite(draws)];effective.append(len(draws))
            tail=(draws<=row[metric]) if low else (draws>=row[metric])
            probabilities.append((1+tail.sum())/(1+len(draws)) if np.isfinite(row[metric]) and len(draws) else np.nan)
        scored[metric+'_p']=probabilities
        scored[metric+'_null_repetitions']=effective
    return nulls, scored
