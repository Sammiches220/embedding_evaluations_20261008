"""Optional independent check: pip install ripser, then run this file.

Ripser is used only as a validation oracle; it is not a project dependency.
"""
import numpy as np
from scipy.spatial.distance import cdist
from ripser import ripser
from embedding_feature_audit.topology import rips_h1


def ordered(values):
    values=np.asarray(values,float).reshape(-1,2)
    return values[np.lexsort((values[:,1],values[:,0]))]


rng=np.random.default_rng(17);checks=0
for n in [6,12,20,30]:
    for dimension in [2,3,5]:
        z=rng.normal(size=(n,dimension));d=cdist(z,z);distances=d[np.triu_indices(n,1)]
        for q in [.2,.6,1.]:
            cutoff=float(np.quantile(distances,q))
            # Ripser reduces float32 distances internally. Match that explicit numerical input.
            matrix=d.astype('float32').astype(float)
            oracle=ripser(matrix,distance_matrix=True,maxdim=1,thresh=cutoff,coeff=2)['dgms'][1]
            bars,_=rips_h1(matrix,cutoff)
            computed=np.column_stack([bars.birth.to_numpy(),bars.death.fillna(np.inf).to_numpy()])
            np.testing.assert_allclose(ordered(computed),ordered(oracle),rtol=1e-6,atol=1e-6)
            checks+=1
print(f'{checks} complete/cutoff H1 diagrams agree with independent Ripser calculations.')
