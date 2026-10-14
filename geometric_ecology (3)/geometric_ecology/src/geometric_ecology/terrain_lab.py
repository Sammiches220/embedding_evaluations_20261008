"""Controlled geometry demonstrations, distinct from corpus discovery.

Vector fixtures demonstrate actual neighborhood directions. Graph fixtures
isolate gaps, bottlenecks, alternatives and boundaries; they are explicitly
constructed graphs, not secretly advertised as embedding kNN discoveries.
"""
import numpy as np
from .core import normalize,knn,sparse_graph
from .terrain import route_set,local_lens,terrain_meanings


def vector_fixture(dimensions,count=2000,seed=41):
    rng=np.random.default_rng(seed)
    raw=np.zeros((count+1,32));raw[:,0]=1
    raw[1:,1:dimensions+1]=rng.normal(0,.08,size=(count,dimensions))
    # Equalize the query's kNN radius across intrinsic dimension conditions.
    for _ in range(3):
        x=normalize(raw)
        distances,_=knn(x,80+1,queries=x[:1])
        raw[1:,1:dimensions+1]*=.025/max(float(distances[0,-1]),1e-8)
    x=normalize(raw)
    _,neighbors=knn(x,81,queries=x[:1]);neighbors=neighbors[0][neighbors[0]!=0][:80]
    center=x.mean(axis=0)
    _,_,axes=np.linalg.svd(x-center,full_matrices=False)
    projected=(x-center)@axes[:2].T
    _,compressed=knn(projected,81,queries=projected[:1]);compressed=compressed[0][compressed[0]!=0][:80]
    keep=np.sort(rng.choice(np.arange(1,len(x)),int(.8*count),replace=False))
    _,sampled=knn(x[keep],80,queries=x[:1]);sampled=keep[sampled[0]]
    lens=local_lens(x,0,neighbors,compressed_ids=compressed,resampled_ids=sampled)
    lens['planted_dimensions']=dimensions
    lens['nearby_count']=80
    lens['substitute_tolerance']=.0125
    lens['near_substitute_count']=int(np.sum(np.asarray(lens['chord_distances'])<=.0125))
    return lens


def grid(rows,columns,offset=(0,0)):
    xy=np.array([[c+offset[0],r+offset[1]] for r in range(rows) for c in range(columns)],float)
    edges=[]
    for r in range(rows):
        for c in range(columns):
            i=r*columns+c
            if c+1<columns:edges.append((i,i+1))
            if r+1<rows:edges.append((i,i+columns))
    return xy,edges


def graph_fixture(kind):
    gates=[];scales=[]
    if kind in ('maze','tree'):
        xy,edges=grid(9,9)
        if kind=='tree':edges=[(u,v) for u,v in edges if v==u+1 or u%9==0]
        source,target=0,80
    else:
        rows,columns=(7,4) if kind=='ridge' else (5,5)
        right_offset=9 if kind=='cliff' else (columns+1 if kind=='ridge' else columns+3)
        left,e1=grid(rows,columns);right,e2=grid(rows,columns,(right_offset,0))
        xy=np.vstack([left,right]);offset=len(left)
        edges=e1+[(u+offset,v+offset) for u,v in e2]
        source,target=0,len(xy)-1
        if kind in ('cliff','island'):
            if kind=='cliff':
                u=(rows//2)*columns+columns-1;v=offset+(rows//2)*columns
                edges.append((u,v));gates=[(u,v)]
        elif kind=='pass':
            u=(rows//2)*columns+columns-1;v=offset+(rows//2)*columns
            start,end=xy[u],xy[v]
            extra=np.array([start+(end-start)/3,start+2*(end-start)/3])
            a,b=len(xy),len(xy)+1;xy=np.vstack([xy,extra])
            gates=[(u,a),(a,b),(b,v)];edges+=gates
        elif kind=='ridge':
            gates=[(row*columns+columns-1,offset+row*columns) for row in [1,3,5]]
            for count in [1,2,3]:
                current=edges+gates[:count]
                e=np.asarray(current,int)
                volume=np.sum(e<offset)
                other=np.sum(e>=offset)
                scales.append(dict(crossings=count,conductance=count/min(volume,other)))
            edges+=gates
    es=np.asarray(edges,int);weights=np.linalg.norm(xy[es[:,0]]-xy[es[:,1]],axis=1)
    graph=sparse_graph(len(xy),es,weights)
    routes=route_set(graph,source,target,limit=12)
    return dict(kind=kind,xy=xy,edges=es,weights=weights,source=source,target=target,
                gates=gates,scales=scales,routes=routes,
                gap_ratio=float(max(weights)/max(np.median(weights),1e-8)),
                object='Explicit controlled graph fixture; not an inferred embedding neighbor graph.')


def build_lab_data():
    vectors={str(d):vector_fixture(d) for d in [2,4,8,12]}
    graphs={name:graph_fixture(name) for name in ['cliff','pass','maze','tree','island','ridge']}
    from .mesh import neighborhood_hull
    hulls={key:neighborhood_hull(lens) for key,lens in vectors.items()}
    return dict(schema_version=2,hulls=hulls,vectors=vectors,graphs=graphs,meanings=terrain_meanings(),
                description='Controlled data to expose what each visual channel means. Vector cases have matched local kNN radius and count; graph cases isolate route structure.')
