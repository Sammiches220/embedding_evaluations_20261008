"""Artificial annotation loss, scored on hidden truth and reconstructed geometry."""
import numpy as np
import pandas as pd
from .geometry import membership
from .modeling import reconstruct
from .statistics import prediction_metrics, score_geometry, geometry_delta


def missingness_experiments(z,y,valid,names,kind,neighbors,ks,fractions,modes,seed=42,
                           predictor=None,priority=None,groups=None):
    rng=np.random.default_rng(seed); known=np.flatnonzero(valid)
    baseline,_=score_geometry(y,valid,names,kind,neighbors,ks)
    summaries=[]; geometry=[]
    for mode in modes:
        if mode=='random':
            order=rng.permutation(known)
        elif mode=='rare-first':
            if kind=='continuous':
                median=np.median(y[known]); rarity=-np.abs(y[known]-median)
            else:
                m=membership(y,names,kind); counts=m[valid].sum(0)
                rarity=np.array([min(counts[np.flatnonzero(m[i])],default=len(known)) for i in known])
            order=known[np.lexsort((rng.random(len(known)),rarity))]
        elif mode=='priority-first':
            if priority is None:
                raise ValueError('priority-first requires numeric missingness_priority_column (smaller hides first)')
            order=known[np.argsort(priority[known],kind='stable')]
        else:
            raise ValueError(f'Unknown missingness mode: {mode}')
        for fraction in fractions:
            if not 0<fraction<1:
                raise ValueError('Missingness fractions must be strictly between 0 and 1')
            count=max(1,int(round(fraction*len(known))))
            hidden=order[:count]; keep=np.setdiff1d(known,hidden)
            # Remove entire groups from training when a hidden member shares one.
            if groups is not None:
                keep=keep[~np.isin(groups[keep],groups[hidden])]
            if not len(keep):
                summaries.append(dict(mode=mode,missing_fraction=fraction,status='skipped',reason='No training rows remain'))
                continue
            pred=(predictor or reconstruct)(z,y,keep,hidden,kind)
            rebuilt=y.copy(); rebuilt[hidden]=pred
            accuracy,_=prediction_metrics(y[hidden],pred,kind,names)
            distortion,_=prediction_metrics(y[known],rebuilt[known],kind,names)
            row=dict(mode=mode,missing_fraction=fraction,n_hidden=len(hidden),n_train=len(keep),status='ok',**accuracy)
            for key,value in distortion.items():
                if 'delta' in key or key=='jensen_shannon_divergence':
                    row['reconstructed_'+key]=value
            summaries.append(row)
            reconstructed,_=score_geometry(rebuilt,valid,names,kind,neighbors,ks)
            delta=geometry_delta(baseline,reconstructed,kind)
            delta['mode']=mode; delta['missing_fraction']=fraction
            geometry.append(delta)
    return pd.DataFrame(summaries),pd.concat(geometry,ignore_index=True) if geometry else pd.DataFrame()
