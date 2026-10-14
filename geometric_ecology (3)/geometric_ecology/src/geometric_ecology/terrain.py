"""Measured terrain channels, original-neighborhood lenses and route-set entropy.

These channels remain separate. More local directions is not synonymous with
higher route cost, ambiguity, semantic meaning, or human value.
"""
from collections import defaultdict
import time
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.sparse.csgraph import connected_components, dijkstra, yen
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from .core import percentile, sparse_graph, knn, normalize


def route_set(graph, source, target, limit=12, detour=1.25, temperature=.10):
    """Entropy of enumerated simple routes, not local degree or all route entropy.

Retain up to `limit` shortest loopless routes whose cost is at most detour*best.
Softmax weights use excess cost / (temperature*best). A cap flag exposes truncation.
    """
    if source==target:
        return dict(paths=[[int(source)]],costs=[0.],probabilities=[1.],entropy_bits=0.,effective_routes=1.,capped=False,reachable=True)
    distances,predecessors=yen(graph,int(source),int(target),limit,directed=False,return_predecessors=True)
    if not len(distances):
        return dict(paths=[],costs=[],probabilities=[],entropy_bits=None,effective_routes=0.,capped=False,reachable=False)
    keep=np.flatnonzero(distances<=distances[0]*detour+1e-10)
    paths=[]
    for rank in keep:
        path=[int(target)]
        while path[-1]!=source:
            prev=int(predecessors[rank,path[-1]])
            if prev<0:raise RuntimeError('Invalid shortest-route predecessor.')
            path.append(prev)
        paths.append(path[::-1])
    costs=distances[keep]
    weights=np.exp(-(costs-costs[0])/max(temperature*costs[0],1e-10))
    weights/=weights.sum()
    entropy=float(-np.sum(weights*np.log(weights)))
    return dict(paths=paths,costs=costs.tolist(),probabilities=weights.tolist(),entropy_bits=entropy/np.log(2),
                effective_routes=float(np.exp(entropy)),capped=bool(len(costs)==limit),reachable=True,
                scope=f'at most {limit} regional simple routes, cost <= {detour} times shortest; temperature={temperature}')


def local_lens(x,index,neighbor_ids,*,compressed_ids=None,resampled_ids=None):
    """Actual neighbor offsets and PCA variance, with distances left in chord units."""
    ids=np.asarray(neighbor_ids,dtype=int)
    offsets=x[ids]-x[index]
    # Tangent-plane approximation to the normalized embedding sphere.
    tangent=offsets-(offsets@x[index])[:,None]*x[index][None,:]
    centered=tangent-tangent.mean(axis=0)
    with threadpool_limits(limits=1):
        _,singular,axes=np.linalg.svd(centered,full_matrices=False)
    energy=singular**2
    ratios=energy/max(energy.sum(),1e-15)
    projections=tangent@axes[:min(3,len(axes))].T
    projections=np.pad(projections,((0,0),(0,max(0,3-projections.shape[1]))))
    components90=int(np.searchsorted(np.cumsum(ratios),.9)+1) if energy.sum()>1e-15 else 0
    participation=float(energy.sum()**2/max(np.sum(energy**2),1e-15))
    result=dict(index=int(index),neighbor_ids=ids.tolist(),offsets=projections.tolist(),
                spectrum=ratios.tolist(),direction_scores=(centered@axes[:min(12,len(axes))].T).tolist(),components90=components90,participation_ratio=participation,
                visible_two_fraction=float(ratios[:2].sum()),radius=float(np.max(np.linalg.norm(offsets,axis=1))),
                chord_distances=np.linalg.norm(offsets,axis=1).tolist(),
                note='PCA covariance directions and distance-growth dimension are different local proxies. More directions does not establish value or travel difficulty.')
    for name,variant in [('compression',compressed_ids),('resampling',resampled_ids)]:
        if variant is not None:
            variant=np.asarray(variant,dtype=int)
            off=x[variant]-x[index]
            tan=off-(off@x[index])[:,None]*x[index][None,:]
            projected=tan@axes[:min(3,len(axes))].T
            projected=np.pad(projected,((0,0),(0,max(0,3-projected.shape[1]))))
            result[name]=dict(neighbor_ids=variant.tolist(),offsets=projected.tolist(),
                              jaccard=len(set(ids)&set(variant))/max(1,len(set(ids)|set(variant))))
    return result


def regional_ambiguity(model,atlas,limit=12):
    """Destination-conditioned route examples plus sampled-target cell summaries."""
    cells=atlas['cells'];roads=atlas['corridors']
    mapping={cell['id']:i for i,cell in enumerate(cells)}
    edges=np.asarray([(mapping[r['a']],mapping[r['b']]) for r in roads],int).reshape(-1,2)
    graph=sparse_graph(len(cells),edges,np.array([r['cost'] for r in roads]))
    _,components=connected_components(graph,directed=False)
    effective=np.ones(len(cells));examples=[];details=[]
    for comp in np.unique(components):
        inds=np.flatnonzero(components==comp)
        if len(inds)<2:continue
        # Three declared reference destinations: representative, farthest, then
        # farthest from the first two. No semantic labels select them.
        first=int(inds[np.argmax([cells[i]['n'] for i in inds])])
        dd=dijkstra(graph,directed=False,indices=first)
        second=int(inds[np.argmax(dd[inds])])
        other=dijkstra(graph,directed=False,indices=second)
        third=int(inds[np.argmax(np.minimum(dd[inds],other[inds]))])
        targets=list(dict.fromkeys([first,second,third]))
        for source in inds:
            records=[];candidates=[]
            for target in targets:
                if source==target:continue
                routes=route_set(graph,int(source),target,limit=limit)
                records.append(routes)
                details.append(dict(source_cell=cells[source]['id'],target_cell=cells[target]['id'],
                                    effective_routes=routes['effective_routes'],capped=routes['capped'],entropy_bits=routes['entropy_bits']))
                if len(routes['paths'])>1:
                    item=dict(source_cell=cells[source]['id'],target_cell=cells[target]['id'],**routes)
                    item['paths']=[[cells[i]['id'] for i in path] for path in routes['paths']]
                    candidates.append(item)
            if records:effective[source]=float(np.median([r['effective_routes'] for r in records]))
            if candidates:examples.append(max(candidates,key=lambda r:r['effective_routes']))
    return dict(cell_effective={cell['id']:float(effective[i]) for i,cell in enumerate(cells)},
                examples=examples,details=details,limit=limit,detour=1.25,
                meaning='Median entropy-equivalent count of enumerated near-shortest regional routes to up to three reference destinations. Not entropy over all paths or semantic prediction difficulty.')


def enrich_terrain(model,atlas):
    started=time.perf_counter()
    x=model['x'];p=model['profiles'];n=len(x);k=model['manifest']['signature_k']
    radius=p['neighbor_radius'];dimension=p['local_dimension']
    # Explicit near-substitute tolerance is corpus-relative, not a semantic label.
    tolerance=float(.1*np.median(radius))
    substitutes=np.mean(model['distances'][:,:k]<=tolerance,axis=1)
    variation=np.std(model['distances'][:,:k],axis=1)/np.maximum(np.mean(model['distances'][:,:k],axis=1),1e-8)
    comp_keys=sorted([key for key in p if key.startswith('pca_')],key=lambda key:int(key.split('_')[1]))
    moderate=comp_keys[len(comp_keys)//2]
    resampling=1-p['resample_jaccard'];compression=1-p[moderate]
    fog=np.maximum(resampling,compression)
    # Cliffs are abrupt relative gaps. Compare actual crossing distances with
    # very local radii, then require contrast against nearby within-cell edges.
    edges=model['stable_edges'];lengths=model['stable_lengths'];regions=model['region']
    local=np.maximum(np.median(model['distances'][:,:min(4,k)],axis=1),1e-8)
    gap=lengths/np.sqrt(local[edges[:,0]]*local[edges[:,1]])
    cliff_candidates=[];ridge_pairs=[]
    all_edges=model['edges'];freq=model['frequency'];all_lengths=model['lengths']
    endpoint_regions=regions[all_edges]
    for road in atlas['corridors']:
        a,b=road['a'],road['b']
        crossing=((endpoint_regions[:,0]==a)&(endpoint_regions[:,1]==b))|((endpoint_regions[:,0]==b)&(endpoint_regions[:,1]==a))
        vertices=np.unique(all_edges[crossing])
        near=np.isin(all_edges[:,0],vertices)|np.isin(all_edges[:,1],vertices)
        internal=near&(endpoint_regions[:,0]==endpoint_regions[:,1])
        contrast=float(np.median(all_lengths[crossing])/max(np.median(all_lengths[internal]) if internal.any() else np.median(all_lengths),1e-8))
        conductances=[]
        for j,_ in enumerate(model['manifest']['scales']):
            threshold=(len(model['manifest']['scales'])-j)/len(model['manifest']['scales'])
            keep=freq>=threshold-1e-8
            end=endpoint_regions[keep]
            va=int(np.sum(end==a));vb=int(np.sum(end==b))
            cross=int(np.sum(crossing&keep))
            conductances.append(cross/max(1,min(va,vb)))
        persistence=float(np.mean(np.asarray(conductances)<.06))
        road['boundary_contrast']=contrast;road['cut_by_scale']=conductances
        road['boundary_scale_fraction']=persistence
        reference_radius=np.median(radius[(regions==a)|(regions==b)])
        density_dip=float(np.median(radius[vertices])/max(reference_radius,1e-8))
        road['boundary_density_dip']=density_dip
        if persistence>=2/3 and contrast>1.25 and density_dip>1.5:
            ridge_pairs.append(dict(a=int(a),b=int(b),scale_fraction=persistence,contrast=contrast,density_dip=density_dip,conductance_by_scale=conductances))
        indices=np.flatnonzero(((regions[edges[:,0]]==a)&(regions[edges[:,1]]==b))|((regions[edges[:,0]]==b)&(regions[edges[:,1]]==a)))
        if len(indices) and contrast>1.8:
            ei=int(indices[np.argmax(gap[indices])])
            if gap[ei]>2.5:
                cliff_candidates.append(dict(edge=edges[ei].tolist(),relative_gap=float(gap[ei]),boundary_contrast=contrast))
    cliffs=sorted(cliff_candidates,key=lambda c:c['relative_gap'],reverse=True)[:24]
    ambiguity=regional_ambiguity(model,atlas)
    route_count=np.array([ambiguity['cell_effective'].get(int(r),1.) for r in regions])
    finite=dimension[np.isfinite(dimension)]
    dim_cap=float(max(2,np.quantile(finite,.95))) if len(finite) else 2.
    height=np.clip(np.log1p(np.nan_to_num(dimension))/np.log1p(dim_cap),0,1)
    terrain=dict(height=height,dimension_cap=dim_cap,metric_variation=variation,substitute_fraction=substitutes,
                 substitute_tolerance=tolerance,fog=fog,resampling_loss=resampling,compression_loss=compression,
                 compression_key=moderate,route_count=route_count,route_ambiguity=ambiguity,
                 cliffs=cliffs,ridges=ridge_pairs,lenses={},meanings=terrain_meanings())
    # Exact lenses at navigation representatives. Other points are explicitly
    # directed to their cell's representative, not given a fabricated spectrum.
    representative_ids=sorted({int(c['representative']) for c in atlas['cells']})
    chosen=np.asarray(representative_ids,int)
    rng=np.random.default_rng(model['manifest']['seed']+901)
    retained=np.sort(rng.choice(n,max(k+2,int(.8*n)),replace=False))
    _,sub_ids=knn(x[retained],k+1,queries=x[chosen])
    rank=int(moderate.split('_')[1]);axes=model['pca_axes'][:rank]
    view=(x-model['pca_mean'])@axes.T
    _,compressed=knn(view,k+1,queries=view[chosen])
    for j,i in enumerate(chosen):
        sub=[int(retained[u]) for u in sub_ids[j] if retained[u]!=i][:k]
        com=[int(u) for u in compressed[j] if u!=i][:k]
        terrain['lenses'][str(i)]=local_lens(x,int(i),model['neighbors'][i,:k],compressed_ids=com,resampled_ids=sub)
    corridor_lookup={(r['a'],r['b']):r for r in atlas['corridors']}
    for example in ambiguity['examples']:
        point_paths=[]
        for cells in example['paths']:
            points=[]
            for a,b in zip(cells,cells[1:]):
                road=corridor_lookup[tuple(sorted((a,b)))]
                segment=road['path'] if road['a']==a else road['path'][::-1]
                points.extend(segment if not points else segment[1:])
            point_paths.append(points)
        example['point_paths']=point_paths
    for island in atlas['islands']:
        inds=np.flatnonzero(model['component']==island['component'])
        island['field']=sample_fields(atlas['xy'][inds],dict(height=height[inds],dimension=dimension[inds],
            density=atlas['density_percentile'][inds],variation=percentile(variation)[inds],fog=fog[inds],
            ambiguity=np.log(np.maximum(route_count[inds],1))/np.log(12),substitutes=substitutes[inds]),regions[inds])
    for cell in atlas['cells']:
        inds=np.flatnonzero(regions==cell['id'])
        cell['substitute_fraction']=float(np.median(substitutes[inds]))
        cell['route_count']=float(ambiguity['cell_effective'].get(cell['id'],1.))
        cell['fog']=float(np.median(fog[inds]))
        # Plain requires ALL three measured qualities, rather than green alone.
        if cell['density_percentile']>.65 and cell['dimension_percentile']<.65 and cell['substitute_fraction']>.25:
            cell['terrain']='redundant plain'
        elif cell['dimension_percentile']>.75:cell['terrain']='many-direction terrain'
        elif cell['route_count']>3:cell['terrain']='route-ambiguous terrain'
        else:cell['terrain']='connected terrain'
    terrain["elapsed_s"]=time.perf_counter()-started
    return terrain


def sample_fields(xy,values,regions,grid=54):
    low,high=xy.min(axis=0),xy.max(axis=0)
    span=np.maximum(high-low,.05);low=low-.08*span-.02;high=high+.08*span+.02
    limits=[(low[1],high[1]),(low[0],high[0])]
    def hist(weights=None):return np.histogram2d(xy[:,1],xy[:,0],bins=grid,range=limits,weights=weights)[0]
    occupancy=gaussian_filter(hist(),1.)
    mask=occupancy>max(.005*occupancy.max(),.006)
    # Empty display bins are not measured zero-height valleys. Extend the
    # nearest supported field value there for road interpolation, while keeping
    # the surface itself masked to observed occupancy.
    nearest_supported=distance_transform_edt(~mask,return_distances=False,return_indices=True)
    fields={}
    for key,v in values.items():
        field=gaussian_filter(hist(np.nan_to_num(v)),1.)/np.maximum(occupancy,1e-10)
        field[~mask]=field[tuple(nearest_supported)][~mask]
        fields[key]=np.round(field,4).tolist()
    # Nearest displayed-region labels locate candidate seams; the seam's meaning
    # still comes from original graph cut evidence, not the display partition.
    xx=np.linspace(low[0],high[0],grid);yy=np.linspace(low[1],high[1],grid)
    coords=np.stack(np.meshgrid(xx,yy),axis=-1).reshape(-1,2)
    _,nearest=cKDTree(xy).query(coords)
    labels=np.asarray(regions)[nearest].reshape(grid,grid)
    segments=defaultdict(list)
    for row in range(grid-1):
        for col in range(grid-1):
            if not mask[row,col]:continue
            for dr,dc in [(1,0),(0,1)]:
                if not mask[row+dr,col+dc]:continue
                a,b=int(labels[row,col]),int(labels[row+dr,col+dc])
                if a==b:continue
                cx=(xx[col]+xx[col+dc])/2;cy=(yy[row]+yy[row+dr])/2
                dx=(xx[1]-xx[0])/2;dy=(yy[1]-yy[0])/2
                segment=[[cx-dx*dr,cy-dy*dc],[cx+dx*dr,cy+dy*dc]]
                segments[tuple(sorted((a,b)))].append(segment)
    return dict(bounds=[float(low[0]),float(low[1]),float(high[0]),float(high[1])],size=grid,
                valid=mask.astype(int).tolist(),values=fields,seams=[dict(a=a,b=b,segments=s) for (a,b),s in segments.items()])


def terrain_meanings():
    return [
      dict(name='Flat plain',measure='Dense + low-dimensional + many near-substitutes',visual='Low relief with aligned short neighbor links; repeated coordinates overlap.',inference='Many nearby replacements; relatively few local directions.',limit='Density alone is insufficient; near-substitutes use an explicit distance tolerance.'),
      dict(name='Mountain',measure='High local dimension or strong metric variation',visual='Measured relief and local direction spectrum; spread across many PCA axes.',inference='A two-axis map omits much of the neighborhood geometry.',limit='Height is complexity, not value or automatically expensive travel.'),
      dict(name='Cliff',measure='An unusually long crossing relative to its local spacing',visual='A break across a gap and a long, costly crossing; gap ratio is labelled.',inference='Nearby-looking locations may require a large original-space step.',limit='A cliff may still have a usable bridge; a low-dimensional region can have a cliff.'),
      dict(name='Narrow pass',measure='Low-conductance bottleneck between regions',visual='Many approach links compress into a small crossing, then fan out.',inference='Few alternative regional crossings; closure can disconnect regions.',limit='The displayed critical pass is a stricter subset of low-conductance passages.'),
      dict(name='Maze',measure='Entropy of enumerated near-shortest routes to an explicit target',visual='Actual alternative routes split, weave and rejoin.',inference='Many similarly priced routes remain plausible.',limit='Bounded regional route-set entropy, not all-path entropy; ambiguity is not necessarily failure or difficulty.'),
      dict(name='Isolated island',measure='Disconnected component or weak cross-scale attachment',visual='Separate shorelines; weak connections are dashed and identified separately.',inference='No retained route, or only condition-sensitive attachment.',limit='Water represents this graph and threshold, not absolute separation in the full space.'),
      dict(name='Ridge / boundary',measure='Low cross-neighborhood conductance across scales, crossing-distance contrast and a local density dip',visual='A continuous seam with visible crossing gates and scale evidence.',inference='Separation persists under the analyzed graph scales.',limit='Reference navigation cells define the cut; this is not proof of semantic or independently discovered cluster boundaries.'),
      dict(name='Fog / unstable ground',measure='Neighborhood changes under reference resampling or dimensional compression',visual='A disagreement field and original/changed neighbor-link ghosts.',inference='The apparent neighborhood depends on the observation condition.',limit='Compression sensitivity can be structure or noise; resampling has unavoidable missing-item loss.')]
