"""Unit vectors and same-split probe pipelines; plain arrays and tables."""
import hashlib
import numpy as np
import pandas as pd
from .modeling import run_probes, validate_probe_models

VARIANTS=['raw_standardized','unit','unit_standardized','norm_only']


def l2_normalize(z):
    """Normalize nonzero finite rows using float64 norms; reject undefined directions."""
    z=np.asarray(z)
    if z.ndim!=2 or not np.isfinite(z).all(): raise ValueError('L2 normalization requires a finite N×D matrix')
    norms=safe_norms(z)
    if np.any(norms==0): raise ValueError('Zero vectors have no unit direction; filter them explicitly')
    scale=np.max(np.abs(z.astype(np.float64)),axis=1)
    scaled=z/scale[:,None]
    return scaled/np.sqrt(np.sum(scaled*scaled,axis=1))[:,None]


def normalization_options(cfg):
    raw=cfg.get('normalization_analysis',{})
    if not isinstance(raw,dict): raise ValueError('normalization_analysis must be an object')
    options=dict(enabled=False,features=True,variants=VARIANTS,max_rows=10000,regularization_grid=[1.0],models=cfg.get('probe_models',['baseline','linear']),trees=cfg.get('trees',100))
    options.update(raw)
    if not isinstance(options['enabled'],bool): raise ValueError('normalization_analysis.enabled must be boolean')
    if not isinstance(options['max_rows'],int) or options['max_rows']<8: raise ValueError('normalization_analysis.max_rows must be >=8')
    if not isinstance(options['variants'],list) or not options['variants'] or set(options['variants'])-set(VARIANTS):
        raise ValueError(f'normalization variants must select {VARIANTS}')
    grid=options['regularization_grid']
    if not isinstance(grid,list) or not grid or any(not isinstance(x,(float,int)) or not np.isfinite(x) or x<=0 for x in grid):
        raise ValueError('normalization regularization_grid must contain positive finite values')
    validate_probe_models(options['models'])
    if not isinstance(options['trees'],int) or options['trees']<1: raise ValueError('normalization trees must be positive')
    if not isinstance(options['features'],(bool,list)): raise ValueError('normalization features must be boolean or a feature list')
    return options


def normalization_probes(z, data, options, *, seed=42, groups=None, split=None, max_labels=10):
    """Compare pipelines on identical known rows; record splits, timing and convergence.

    A grid of more than one value tunes Ridge alpha / logistic C using training
    folds only; group_column keeps groups separate in both outer and inner splits.
    No row sampling or test-set tuning occurs here.
    """
    if len(z)>options['max_rows']:
        return dict(status='skipped',reason=f'Normalization probes exceed max_rows={options["max_rows"]}; use an explicit evaluation sample.'),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    direction=bool(set(options['variants']) & {'unit','unit_standardized'})
    if not np.isfinite(z).all() or direction and np.any(safe_norms(z)==0):
        return dict(status='skipped',reason='Direction comparisons require finite nonzero vectors on the same cohort.'),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    y=data['values'];names=data['names'];kind=data['kind'];valid=data['valid']
    if kind=='multilabel': names=names[:max_labels];y=y[:,:len(names)]
    unit=l2_normalize(z) if direction else None;norm=safe_norms(z)[:,None]
    inputs={'raw_standardized':(z,'standardize'),'unit':(unit,'none'),
            'unit_standardized':(unit,'standardize'),'norm_only':(norm,'standardize')}
    scores=[];predictions=[];splits=[];reference=None;errors=[]
    for variant in dict.fromkeys(options['variants']):
        matrix,preprocess=inputs[variant]
        try:
            summary,detail,pred,train,test=run_probes(matrix,y,valid,names,kind,seed,groups,split,
                trees=options['trees'],models=list(dict.fromkeys(['baseline']+options['models'])),preprocess=preprocess,regularization_grid=options['regularization_grid'])
            signature=hashlib.sha256(np.r_[train,-1,test].astype('int64').tobytes()).hexdigest()
            if reference is not None and reference!=signature: raise RuntimeError('Normalization variants changed the held-out split')
            reference=signature
            for row in summary.to_dict('records'):
                row.update(variant=variant,split_signature=signature,kind=kind,preprocess=preprocess)
                scores.append(row)
            if len(pred): pred['variant']=variant;predictions.append(pred)
            train_set=set(train)
            splits.extend(dict(variant=variant,row_position=int(i),split='train' if i in train_set else 'test') for i in np.r_[train,test])
        except ValueError as exc: errors.append(dict(variant=variant,reason=str(exc)))
    status=dict(status='ok' if scores else 'skipped',labels=names,errors=errors,grid=options['regularization_grid'],
        scope='Same full evaluated cohort and feature-specific held-out rows; each pipeline fitted on training rows only. ',
        stability='Fit timings, solver iterations and convergence warnings are observations, not guarantees of training stability.',
        regularization='Grid values are Ridge alpha or logistic C (inverse penalty strength); >1 value enables training-only CV.')
    if not scores: status['reason']='No auditable probe targets or variants; inspect per-variant errors.'
    return status,pd.DataFrame(scores),pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame(),pd.DataFrame(splits)


def safe_norms(z):
    """Scaled float64 L2 norms avoid underflow/overflow in the sum of squares."""
    a=np.asarray(z,dtype=np.float64)
    scale=np.max(np.abs(a),axis=1)
    scaled=np.divide(a,scale[:,None],out=np.zeros_like(a),where=scale[:,None]>0)
    with np.errstate(over='ignore',invalid='ignore'):
        return scale*np.sqrt(np.sum(scaled*scaled,axis=1))
