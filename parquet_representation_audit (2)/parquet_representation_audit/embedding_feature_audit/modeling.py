"""Held-out probes with per-label results; no feature-wrapper classes."""
import numpy as np
import pandas as pd
import warnings
import json
from time import perf_counter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV, GroupKFold, StratifiedKFold, KFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .statistics import prediction_metrics


def split_rows(valid, values, kind, seed=42, groups=None, explicit_split=None):
    ids = np.flatnonzero(valid)
    if len(ids)<8:
        raise ValueError('Need >=8 known rows for a held-out probe')
    if explicit_split is not None:
        train = ids[explicit_split[ids]=='train']; test = ids[explicit_split[ids]=='test']
        if not len(train) or not len(test):
            raise ValueError('split_column must contain train and test on known rows')
        if groups is not None and set(groups[train]) & set(groups[test]):
            raise ValueError('Explicit train/test split leaks groups')
        return train,test
    if groups is not None:
        a,b = next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed).split(ids,groups=groups[ids]))
        return ids[a],ids[b]
    stratify = None
    if kind=='categorical':
        _,counts = np.unique(values[ids],return_counts=True)
        ntest = int(np.ceil(len(ids)*.25))
        if counts.min()>=2 and len(counts)<=min(ntest,len(ids)-ntest):
            stratify = values[ids]
    return train_test_split(ids,test_size=.25,random_state=seed,stratify=stratify)


def probe_models(kind, n_train, seed=42, trees=100, preprocess='standardize'):
    if preprocess not in ['standardize','none']:
        raise ValueError('preprocess must be standardize or none')
    def linear(model):
        return make_pipeline(StandardScaler(),model) if preprocess=='standardize' else model
    if kind=='continuous':
        return {'baseline':DummyRegressor(strategy='mean'),
            'linear':linear(Ridge(alpha=1.)),
            'knn':KNeighborsRegressor(n_neighbors=min(15,n_train),weights='distance'),
            'random_forest':RandomForestRegressor(n_estimators=trees,random_state=seed,n_jobs=1)}
    return {'baseline':DummyClassifier(strategy='most_frequent'),
        'linear':linear(LogisticRegression(max_iter=1000,class_weight='balanced',random_state=seed)),
        'knn':KNeighborsClassifier(n_neighbors=min(15,n_train),weights='distance'),
        'random_forest':RandomForestClassifier(n_estimators=trees,class_weight='balanced_subsample',random_state=seed,n_jobs=1)}


def run_probes(z, y, valid, names, kind, seed=42, groups=None, explicit_split=None, trees=100, models=None,
               preprocess='standardize', regularization_grid=None):
    requested = ['baseline','linear','knn','random_forest'] if models is None else list(models)
    if not requested or set(requested)-{'baseline','linear','knn','random_forest'}:
        raise ValueError('probe_models must select baseline, linear, knn or random_forest')
    def selected_models(model_kind):
        return {name:model for name,model in probe_models(model_kind,len(train),seed,trees,preprocess).items() if name in requested}
    train,test = split_rows(valid,y,kind,seed,groups,explicit_split)
    def fit_model(model, target, model_name, model_kind):
        grid=regularization_grid or []
        if model_name=='linear' and grid:
            parameter='alpha' if model_kind=='continuous' else 'C'
            if preprocess=='standardize': parameter=('ridge__' if model_kind=='continuous' else 'logisticregression__')+parameter
            if len(grid)>1:
                if groups is not None:
                    count=min(3,len(np.unique(groups[train])))
                    if count<2: raise ValueError('Regularization tuning requires >=2 training groups')
                    cv=list(GroupKFold(count).split(z[train],target,groups[train]))
                elif model_kind=='continuous': cv=KFold(min(3,len(train)),shuffle=True,random_state=seed)
                else:
                    count=min(3,int(np.unique(target,return_counts=True)[1].min()))
                    if count<2: raise ValueError('Regularization tuning requires >=2 training examples per class')
                    cv=StratifiedKFold(count,shuffle=True,random_state=seed)
                model=GridSearchCV(model,{parameter:grid},cv=cv,
                    scoring='neg_mean_absolute_error' if model_kind=='continuous' else 'balanced_accuracy',
                    error_score='raise',n_jobs=1)
            else: model.set_params(**{parameter:grid[0]})
        start=perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(z[train],target)
        estimator=getattr(model,'best_estimator_',model)
        final=estimator.steps[-1][1] if hasattr(estimator,'steps') else estimator
        iterations=getattr(final,'n_iter_',None)
        info=dict(fit_seconds=perf_counter()-start,convergence_warnings=sum(issubclass(w.category,ConvergenceWarning) for w in caught),
                  max_iterations=float(np.max(iterations)) if iterations is not None else np.nan,
                  selected_regularization=float(getattr(final,'alpha',getattr(final,'C',np.nan))) if model_name=='linear' else np.nan,
                  tuning='training-only CV' if isinstance(model,GridSearchCV) else 'fixed')
        return model,info
    rows,details,predictions=[],[],[]
    if kind=='multilabel':
        # Independent binary probes keep tag semantics explicit and reveal unsupported labels.
        for col,name in enumerate(names):
            if len(np.unique(y[train,col]))<2:
                rows.append(dict(model='all',label=name,status='skipped',reason='Training set lacks positives or negatives',n_train=len(train),n_test=len(test)))
                continue
            for model_name,model in selected_models('categorical').items():
                model,fit_info=fit_model(model,y[train,col],model_name,'categorical'); pred = model.predict(z[test])
                metrics,_ = prediction_metrics(y[test,col],pred,'categorical',[])
                p = model.predict_proba(z[test])[:,list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(test))
                metrics['roc_auc'] = float(roc_auc_score(y[test,col],p)) if len(np.unique(y[test,col]))==2 else np.nan
                metrics['average_precision'] = float(average_precision_score(y[test,col],p)) if y[test,col].sum() else np.nan
                metrics['test_positive_support'] = int(y[test,col].sum())
                rows.append(dict(model=model_name,label=name,status='ok',n_train=len(train),n_test=len(test),**metrics,**fit_info))
                predictions.extend(dict(row_position=int(i),model=model_name,label=name,true=int(t),predicted=int(pv),probability=float(prob))
                                   for i,t,pv,prob in zip(test,y[test,col],pred,p))
        detail = pd.DataFrame(rows)
        ok = detail[detail.status=='ok'] if len(detail) else pd.DataFrame()
        summary = ok.groupby('model').mean(numeric_only=True).reset_index() if len(ok) else pd.DataFrame()
        if len(summary):
            summary['aggregation']='macro across auditable binary tags'
            summary['n_labels']=int(ok.label.nunique())
            support=ok[(ok.test_positive_support>0)&(ok.test_positive_support<ok.n_test)]
            summary['n_supported_labels']=int(support.label.nunique())
            for metric in ['fit_seconds','convergence_warnings']:
                summary[metric]=summary.model.map(ok.groupby('model')[metric].sum())
            summary['max_iterations']=summary.model.map(ok.groupby('model').max_iterations.max())
            summary['selected_regularization_by_label']=summary.model.map({model:json.dumps({str(row['label']):row['selected_regularization'] for row in table.to_dict('records') if np.isfinite(row['selected_regularization'])}) for model,table in ok.groupby('model')})
            summary['selected_regularization']=np.nan
            summary['tuning']='training-only CV' if regularization_grid and len(regularization_grid)>1 else 'fixed'
        return summary,detail,pd.DataFrame(predictions),train,test
    for name,model in selected_models(kind).items():
        if kind=='categorical' and len(np.unique(y[train]))<2 and name=='linear':
            rows.append(dict(model=name,status='skipped',reason='Only one training class'))
            continue
        model,fit_info=fit_model(model,y[train],name,kind); pred=model.predict(z[test])
        metrics,per_label=prediction_metrics(y[test],pred,kind,names)
        row=dict(model=name,status='ok',n_train=len(train),n_test=len(test),**metrics,**fit_info)
        if kind=='categorical': row['n_classes']=int(np.unique(y[test]).size)
        if kind=='categorical':
            proba=model.predict_proba(z[test]); classes=list(model.classes_)
            aucs=[]; aps=[]
            for label in classes:
                binary=(y[test]==label).astype(int); p=proba[:,classes.index(label)]
                auc=float(roc_auc_score(binary,p)) if len(np.unique(binary))==2 else np.nan
                ap=float(average_precision_score(binary,p)) if binary.sum() else np.nan
                aucs.append(auc); aps.append(ap)
                per_label.loc[per_label.label==label,'roc_auc']=auc
                per_label.loc[per_label.label==label,'average_precision']=ap
            row['roc_auc']=float(np.nanmean(aucs)) if np.isfinite(aucs).any() else np.nan
            row['average_precision']=float(np.nanmean(aps)) if np.isfinite(aps).any() else np.nan
        if len(per_label):
            per_label['model']=name; details.append(per_label)
        rows.append(row)
        predictions.extend(dict(row_position=int(i),model=name,true=t,predicted=p) for i,t,p in zip(test,y[test],pred))
    return pd.DataFrame(rows),pd.concat(details,ignore_index=True) if details else pd.DataFrame(),pd.DataFrame(predictions),train,test


def reconstruct(z, y, train, hidden, kind):
    """A replaceable demo predictor; embedding-derived reconstruction is not independent."""
    if kind=='continuous' or kind=='multilabel':
        model=KNeighborsRegressor(n_neighbors=min(15,len(train)),weights='distance')
    else:
        model=KNeighborsClassifier(n_neighbors=min(15,len(train)),weights='distance')
    model.fit(z[train],y[train]); pred=model.predict(z[hidden])
    if kind=='multilabel':
        pred=(pred>=.5).astype(np.uint8)
    return pred
