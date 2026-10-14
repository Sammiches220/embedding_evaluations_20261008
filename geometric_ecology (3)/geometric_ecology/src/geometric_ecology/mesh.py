"""Measured local covariance and neighborhood meshes, using arrays and dicts."""
import time
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from threadpoolctl import threadpool_limits
from .terrain import sample_fields


def neighborhood_spectra(x, neighbors, batch_size=256):
    """Tangent-neighborhood covariance spectra for every query, in batches.

    Centering and removing the sphere-normal component match local_lens.
    Gram eigenvalues avoid a full ambient-dimensional SVD at every point.
    No directions beyond the observed neighbor sample can be estimated.
    """
    n, k = neighbors.shape
    spectra = np.zeros((n, k), dtype=np.float32)
    with threadpool_limits(limits=1):
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            query = x[start:stop].astype(np.float64)
            offsets = x[neighbors[start:stop]].astype(np.float64) - query[:, None, :]
            offsets -= np.einsum('nkd,nd->nk', offsets, query)[:, :, None] * query[:, None, :]
            offsets -= offsets.mean(axis=1, keepdims=True)
            gram = offsets @ offsets.transpose(0, 2, 1)
            energy = np.maximum(np.linalg.eigvalsh(gram)[:, ::-1], 0)
            total = energy.sum(axis=1, keepdims=True)
            spectra[start:stop] = np.divide(energy, total, out=np.zeros_like(energy), where=total > 1e-15)
    valid = spectra.sum(axis=1) > .5
    effective = np.divide(1., np.sum(spectra ** 2, axis=1), out=np.zeros(n), where=valid)
    components90 = np.argmax(np.cumsum(spectra, axis=1) >= .9, axis=1) + 1
    components90[~valid] = 0
    hidden = np.clip(1 - spectra[:, :2].sum(axis=1), 0, 1)
    hidden[~valid] = 0
    return dict(spectrum=spectra, effective=effective, hidden_two=hidden,
                components90=components90, valid=valid)


def neighborhood_hull(lens):
    """Convex envelope of actual first-three-PC offsets; never add random jitter.

    Rank-two neighborhoods are triangulated planar envelopes. A line or a
    coincident neighborhood has no fabricated surface triangles.
    """
    points = np.asarray(lens['offsets'], dtype=float)
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.sum(singular > max(1e-12, singular[0] * 1e-6))) if len(singular) else 0
    triangles = []
    try:
        if rank >= 3:
            triangles = ConvexHull(points).simplices.tolist()
        elif rank == 2:
            # Stored coordinates are ordered principal directions.
            boundary = ConvexHull(points[:, :2]).vertices
            triangles = [[int(boundary[0]), int(boundary[i]), int(boundary[i + 1])]
                         for i in range(1, len(boundary) - 1)]
    except QhullError:
        # Exact duplicates/degeneracy remain visible as points and lines.
        triangles = []
    radius = float(np.max(np.linalg.norm(points, axis=1))) if len(points) else 0.
    return dict(vertices=points.tolist(), triangles=triangles, rank=rank, radius=radius,
                projected_variance=float(sum(lens['spectrum'][:3])),
                note='Actual PCA-3 neighbor envelope in a local frame; footprint normalized for legibility. Facet count is not dimension.')


def enrich_mesh(model, atlas):
    """Attach all-point spectra, interpolated color fields and local hulls."""
    start = time.perf_counter()
    k = model['manifest']['signature_k']
    stats = neighborhood_spectra(model['x'], model['neighbors'][:, :k])
    # Covariance rank has a finite sample bound; use it rather than a quantile
    # cap that would flatten the rare most-dimensional regions.
    cap = float(max(1, min(k - 1, model['x'].shape[1] - 1)))
    height = np.clip((stats['effective'] - 1) / max(1, cap - 1), 0, 1)
    xy = np.asarray(atlas['xy'])
    for island in atlas['islands']:
        inds = np.flatnonzero(model['component'] == island['component'])
        field = sample_fields(xy[inds], dict(hidden_two=stats['hidden_two'][inds],
            covariance_dimensions=stats['effective'][inds], covariance_height=height[inds]), model['region'][inds])
        island['field']['values'].update(field['values'])
    glyphs = {}
    for key, lens in atlas['terrain']['lenses'].items():
        glyphs[key] = neighborhood_hull(lens)
    return dict(schema_version=1, hidden_two=stats['hidden_two'],
        covariance_dimensions=stats['effective'], components90=stats['components90'],
        valid=stats['valid'], covariance_cap=cap, glyphs=glyphs, elapsed_s=time.perf_counter()-start,
        meaning='Color is covariance variance omitted by two axes. Local meshes are actual PCA-3 envelopes; spectra expose further hidden directions. Covariance participation ratio differs from radius-growth dimension.')
