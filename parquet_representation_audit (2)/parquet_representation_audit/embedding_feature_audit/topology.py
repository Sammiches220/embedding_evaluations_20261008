"""Sampled distance filtrations, H0 mergers and bounded exact H1 over GF(2).

No wrappers: arrays in, dictionaries and pandas tables out. The small H1 reducer
includes triangles; it is not graph cycle counting or an 8m-row persistence engine.
"""
import itertools
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from .normalization import l2_normalize
from .feature_overview import entropy_probabilities
from .geometry import membership

SPACES=['unit_chordal','raw_euclidean']


def topology_options(cfg):
    raw=cfg.get('topology',{})
    if not isinstance(raw,dict): raise ValueError('topology must be an object')
    options=dict(enabled=False,features=True,spaces=['unit_chordal'],sample_size=512,repetitions=3,
        quantiles=[0,.001,.005,.01,.025,.05,.1,.25,.5,1],radii={},permutations=20,
        min_component_known=1,min_retained_share=.5,null_mode='permutation',h1=True,h1_sample_size=64,max_simplices=100000,max_distance_bytes=256000000)
    options.update(raw)
    if not isinstance(options['enabled'],bool) or not isinstance(options['h1'],bool): raise ValueError('topology enabled/h1 must be boolean')
    if not isinstance(options['features'],(bool,list)): raise ValueError('topology features must be boolean or a list')
    if not isinstance(options['spaces'],list) or not options['spaces'] or set(options['spaces'])-set(SPACES): raise ValueError(f'topology spaces must select {SPACES}')
    for key,minimum,maximum in [('sample_size',2,2000),('repetitions',1,20),('permutations',0,1000),('min_component_known',1,2000),('h1_sample_size',3,128),('max_simplices',10,500000)]:
        if not isinstance(options[key],int) or not minimum<=options[key]<=maximum: raise ValueError(f'topology.{key} must be an integer in [{minimum},{maximum}]')
    if not isinstance(options['max_distance_bytes'],int) or options['max_distance_bytes']<1024: raise ValueError('topology.max_distance_bytes must be an integer >=1024')
    if not isinstance(options['min_retained_share'],(int,float)) or not 0<=options['min_retained_share']<=1: raise ValueError('topology.min_retained_share must be in [0,1]')
    q=options['quantiles']
    if not isinstance(q,list) or len(q)<2 or any(not isinstance(x,(float,int)) or not np.isfinite(x) or not 0<=x<=1 for x in q): raise ValueError('topology quantiles require >=2 finite values in [0,1]')
    if not isinstance(options['radii'],dict) or set(options['radii'])-set(SPACES): raise ValueError('topology radii must map supported spaces to radius lists')
    for radii in options['radii'].values():
        if not isinstance(radii,list) or len(radii)<2 or any(not isinstance(x,(float,int)) or not np.isfinite(x) or x<0 for x in radii): raise ValueError('Explicit topology radii must be finite nonnegative lists with >=2 values')
    if options['null_mode'] not in ['permutation','stratified']: raise ValueError('topology null_mode must be permutation or stratified')
    return options


def distance_matrix(z, space):
    matrix=l2_normalize(z) if space=='unit_chordal' else np.asarray(z)
    return cdist(matrix,matrix,'euclidean')


def component_filtration(distances, radii):
    """Nested full-metric threshold graphs. Save actual merges and memberships."""
    d=np.asarray(distances,float);n=len(d)
    if d.shape!=(n,n) or not np.isfinite(d).all() or np.any(d<0) or not np.allclose(d,d.T): raise ValueError('Expected a symmetric finite nonnegative distance matrix')
    if not np.allclose(np.diag(d),0): raise ValueError('Distance matrix diagonal must be zero')
    radii=sorted(set(map(float,radii)))
    if not radii or radii[0]<0: raise ValueError('Filtration radii must be nonnegative')
    i,j=np.triu_indices(n,1);weights=d[i,j];order=np.lexsort((j,i,weights))
    parent=np.arange(n);size=np.ones(n,int);cursor=0;bars=[];merges=[];snapshots=[];rows=[]
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return int(x)
    for radius in radii:
        while cursor<len(order) and weights[order[cursor]]<=radius:
            edge=order[cursor];a=root(i[edge]);b=root(j[edge]);cursor+=1
            if a==b: continue
            a,b=sorted([a,b]);parent[b]=a;size[a]+=size[b]
            weight=float(weights[edge]);bars.append(dict(dimension=0,birth=0.,death=weight,status='finite',representative=b))
            merges.append(dict(radius=weight,left=a,right=b,edge_i=int(i[edge]),edge_j=int(j[edge]),merged_size=int(size[a])))
        labels=np.array([root(x) for x in range(n)]);snapshots.append(labels)
        _,counts=np.unique(labels,return_counts=True)
        rows.append(dict(radius=radius,beta0=len(counts),largest_component_share=float(counts.max()/n),n_edges=cursor,n_rows=n))
    roots=np.unique(snapshots[-1])
    for a in roots: bars.append(dict(dimension=0,birth=0.,death=np.nan,status='essential' if a==roots.min() else 'right-censored',representative=int(a)))
    return pd.DataFrame(rows),snapshots,pd.DataFrame(bars),pd.DataFrame(merges)


def rips_h1(distances, cutoff, max_simplices=100000):
    """Exact finite Rips H1 through cutoff over GF(2), with an explicit size guard.

    Faces precede cofaces even at tied distances. Boundary columns use Python
    integer bitsets and standard left-to-right pivot reduction. H2 is not computed.
    Alive H1 intervals are right-censored because the radius calculation stops.
    """
    if not isinstance(max_simplices,int) or max_simplices<1 or not np.isfinite(cutoff) or cutoff<0: raise ValueError('H1 requires a finite nonnegative cutoff and positive simplex budget')
    d=np.asarray(distances,float);n=len(d)
    if n>128: raise ValueError('Built-in H1 is limited to 128 explicitly sampled points')
    if d.shape!=(n,n) or not np.isfinite(d).all() or np.any(d<0) or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0): raise ValueError('Expected a symmetric finite nonnegative distance matrix')
    simplices=[(0.,0,(i,)) for i in range(n)]
    for i,j in itertools.combinations(range(n),2):
        if d[i,j]<=cutoff: simplices.append((float(d[i,j]),1,(i,j)))
    if len(simplices)>max_simplices: raise ValueError('H1 simplex budget exceeded before triangle expansion')
    for i,j,k in itertools.combinations(range(n),3):
        value=max(d[i,j],d[i,k],d[j,k])
        if value<=cutoff:
            simplices.append((float(value),2,(i,j,k)))
            if len(simplices)>max_simplices: raise ValueError('H1 simplex budget exceeded during triangle expansion; use fewer points or a smaller cutoff')
    simplices.sort(key=lambda item:(item[0],item[1],item[2]))
    index={vertices:i for i,(_,_,vertices) in enumerate(simplices)}
    reduced={};chains={};cycles={};born={};paired=set();bars=[]
    def cycle_edges(bits):
        edges=[]
        while bits:
            low=bits & -bits;i=low.bit_length()-1;edges.append(list(simplices[i][2]));bits^=low
        return edges
    for column,(value,dimension,vertices) in enumerate(simplices):
        boundary=0;chain=(1<<column) if dimension==1 else 0
        if dimension:
            for face in itertools.combinations(vertices,dimension): boundary^=1<<index[face]
        while boundary:
            pivot=boundary.bit_length()-1
            if pivot not in reduced: break
            boundary^=reduced[pivot]
            if dimension==1: chain^=chains[pivot]
        if boundary:
            pivot=boundary.bit_length()-1;reduced[pivot]=boundary;paired.add(pivot)
            if dimension==1: chains[pivot]=chain
            birth,dim=born[pivot]
            if dim==1 and value>birth:
                edges=cycle_edges(cycles[pivot])
                bars.append(dict(dimension=1,birth=birth,death=value,status='finite',birth_simplex_index=pivot,cycle_edges=edges))
        else:
            born[column]=(value,dimension)
            if dimension==1: cycles[column]=chain
    for column,(birth,dim) in born.items():
        if dim==1 and column not in paired:
            edges=cycle_edges(cycles[column])
            bars.append(dict(dimension=1,birth=birth,death=np.nan,status='right-censored',birth_simplex_index=column,cycle_edges=edges))
    return pd.DataFrame(bars,columns=['dimension','birth','death','status','birth_simplex_index','cycle_edges']),len(simplices)


def component_measure(values,valid,names,kind,labels,global_variance=None,min_known=1):
    """Known-row weighted conditional entropy/variance; full vocabulary stays fixed."""
    ids=np.flatnonzero(valid)
    if not len(ids): return dict(value=np.nan,n_known=0,retained_known=0,retained_share=np.nan,n_components_measured=0)
    group_ids,inverse=np.unique(labels[ids],return_inverse=True)
    counts=np.bincount(inverse);keep=counts>=min_known;retained=int(counts[keep].sum())
    if not retained: return dict(value=np.nan,n_known=len(ids),retained_known=0,retained_share=0.,n_components_measured=0)
    if kind=='continuous':
        y=np.asarray(values[ids],float);y=y-y.mean();sums=np.bincount(inverse,weights=y);sums2=np.bincount(inverse,weights=y*y)
        means=sums/counts;measure=np.maximum(0,sums2/counts-means*means)
        if global_variance is None: global_variance=float(np.var(y))
        if global_variance<=0: value=np.nan
        else: value=float(np.average(measure[keep],weights=counts[keep])/global_variance)
    else:
        if not names: return dict(value=np.nan,n_known=len(ids),retained_known=retained,retained_share=retained/len(ids),n_components_measured=int(keep.sum()))
        members=membership(values[ids],names,kind);measure=np.zeros(len(counts))
        for col in range(len(names)):
            successes=np.bincount(inverse,weights=members[:,col],minlength=len(counts))
            measure+=entropy_probabilities(successes/counts,kind=='multilabel')
        measure/=len(names) if kind=='multilabel' else (np.log2(len(names)) if len(names)>1 else 1.)
        value=float(np.average(measure[keep],weights=counts[keep]))
    return dict(value=value,n_known=len(ids),retained_known=retained,retained_share=retained/len(ids),n_components_measured=int(keep.sum()))


def shuffled_values(values,valid,rng,strata=None):
    out=values.copy();known=np.flatnonzero(valid)
    blocks=[known] if strata is None else [known[strata[known]==key] for key in np.unique(strata[known])]
    for rows in blocks: out[rows]=values[rng.permutation(rows)]
    return out


def audit_topology(models, encoded, ids, options, *, seed=42, strata=None, progress=None):
    """Matched samples and shared radii across A/B; descriptive repeated-sample bands."""
    n=len(ids);count=min(n,options['sample_size']);reps=1 if count==n else options['repetitions']
    if options['null_mode']=='stratified' and strata is None: raise ValueError('Stratified topology requires aligned strata')
    estimated=count*count*len(models)*len(set(options['spaces']))*reps*12
    if estimated>options['max_distance_bytes']:
        status=dict(status='skipped',reason=f'Estimated distance/pooling memory {estimated:,} bytes exceeds topology.max_distance_bytes={options["max_distance_bytes"]:,}; use fewer sampled rows or repetitions.',options=options)
        return (status,)+(pd.DataFrame(),)*6
    rng=np.random.default_rng(seed);samples=[np.sort(rng.choice(n,count,replace=False)) for _ in range(reps)]
    spaces=list(dict.fromkeys(options['spaces']));radii={};matrices={}
    for space in spaces:
        pool=[]
        for model,z in models.items():
            for repeat,rows in enumerate(samples):
                d=distance_matrix(z[rows],space);matrices[(space,model,repeat)]=d;pool.append(d[np.triu_indices(count,1)])
        if space in options['radii']: radii[space]=sorted(set(map(float,options['radii'][space])))
        else:
            values=np.concatenate(pool);radii[space]=sorted(set([0.]+np.quantile(values,options['quantiles']).tolist()))
    feature_names=[name for name in encoded if options['features'] is True or isinstance(options['features'],list) and name in options['features']]
    feature_rows=[];structure=[];barcodes=[];merges=[];h1_status=[];sample_records=[]
    # Independent of model/space so item correspondence is retained for every H1 comparison.
    h1_local=np.sort(np.random.default_rng(seed+97).choice(count,min(count,options['h1_sample_size']),replace=False))
    for repeat,rows in enumerate(samples):
        h1_index={int(local):int(position) for position,local in enumerate(h1_local)}
        sample_records.extend(dict(repeat=repeat,sample_position=i,row_position=int(row),id=str(ids[row]),
            h1_selected=bool(options['h1'] and repeat==0 and i in h1_index),h1_position=h1_index.get(i) if options['h1'] and repeat==0 else None) for i,row in enumerate(rows))
        for space in spaces:
            for model in models:
                if progress: progress(f'Topology {space}, embedding {model}, sample {repeat+1}/{reps}')
                d=matrices[(space,model,repeat)]
                table,partitions,bars,merge=component_filtration(d,radii[space])
                for record in table.to_dict('records'): structure.append(dict(space=space,embedding=model,repeat=repeat,**record))
                for record in bars.to_dict('records'): barcodes.append(dict(space=space,embedding=model,repeat=repeat,n_points=count,**record))
                for record in merge.to_dict('records'): merges.append(dict(space=space,embedding=model,repeat=repeat,**record))
                if repeat==0 and options['h1']:
                    try:
                        h1,simplices=rips_h1(d[np.ix_(h1_local,h1_local)],radii[space][-1],options['max_simplices'])
                        for record in h1.to_dict('records'): barcodes.append(dict(space=space,embedding=model,repeat=0,n_points=len(h1_local),**record))
                        h1_status.append(dict(space=space,embedding=model,status='ok',n_points=len(h1_local),n_simplices=simplices,cutoff=radii[space][-1]))
                    except ValueError as exc: h1_status.append(dict(space=space,embedding=model,status='skipped',reason=str(exc),n_points=len(h1_local)))
                for name in feature_names:
                    data=encoded[name];y=data['values'][rows];valid=data['valid'][rows];kind=data['kind'];names=data['names']
                    variance=float(np.var(data['values'][data['valid']])) if kind=='continuous' and data['valid'].any() else None
                    null_rng=np.random.default_rng(seed+repeat*1009+feature_names.index(name)*37)
                    nulls=[shuffled_values(y,valid,null_rng,strata[rows] if options['null_mode']=='stratified' else None) for _ in range(options['permutations'])]
                    for radius,labels in zip(radii[space],partitions):
                        observed=component_measure(y,valid,names,kind,labels,variance,options['min_component_known'])
                        null_values=np.array([component_measure(yn,valid,names,kind,labels,variance,options['min_component_known'])['value'] for yn in nulls])
                        finite=null_values[np.isfinite(null_values)];null_mean=float(finite.mean()) if len(finite) else np.nan
                        feature_rows.append(dict(feature=name,kind=kind,space=space,embedding=model,repeat=repeat,radius=radius,
                            measure='component_variance_ratio' if kind=='continuous' else 'component_entropy',
                            observed=observed['value'],null_mean=null_mean,null_sd=float(finite.std()) if len(finite) else np.nan,
                            organization_gap=null_mean-observed['value'],**{k:v for k,v in observed.items() if k!='value'}))
    raw=pd.DataFrame(feature_rows);summary=[]
    if len(raw):
        for keys,group in raw.groupby(['feature','kind','space','embedding','radius','measure'],sort=False):
            record=dict(zip(['feature','kind','space','embedding','radius','measure'],keys));record['n_samples']=len(group)
            for column in ['observed','null_mean','organization_gap','retained_share','n_known','retained_known','n_components_measured']:
                values=group[column].to_numpy(float);values=values[np.isfinite(values)]
                record[column]=float(values.mean()) if len(values) else np.nan
                if column=='organization_gap':
                    record['sample_min']=float(values.min()) if len(values) else np.nan;record['sample_max']=float(values.max()) if len(values) else np.nan
            summary.append(record)
    status=dict(status='ok',options=options,n_population=n,n_sample=count,requested_repetitions=options['repetitions'],actual_repetitions=reps,
        radii=radii,h1_status=h1_status,sample_scope='Seeded uniform samples of evaluated rows; identical IDs across models and spaces. H1 uses an explicit smaller subset of the first sample.',
        metric_scope='Exact complete pairwise distances within each sample; no ANN graph or projection is used.',
        null_scope='Whole observed label/tag rows shuffled within the sample; known mask fixed; strata respected only in stratified mode.',
        interpretation='Positive null-minus-observed gap suggests annotation organization across scales. It does not establish linear separability or semantic usefulness.',
        uncertainty='Repeated-sample min/max bands are descriptive, not confidence intervals. No persistence significance test is claimed.')
    return status,pd.DataFrame(summary),raw,pd.DataFrame(structure),pd.DataFrame(barcodes),pd.DataFrame(merges),pd.DataFrame(sample_records)
