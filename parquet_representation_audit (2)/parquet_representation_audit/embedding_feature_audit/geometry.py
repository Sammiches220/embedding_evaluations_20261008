"""A shared graph; known labels are evaluated inside the same neighborhoods."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


def make_neighbors(z, max_k, metric='cosine', batch_size=256):
    """Exclude self by row ID, including tied/duplicate embedding vectors."""
    k = min(int(max_k), len(z) - 1)
    if k < 1:
        raise ValueError('k must be positive')
    if metric=='cosine':
        from .normalization import l2_normalize
        z=l2_normalize(z)
    model = NearestNeighbors(metric=metric, algorithm='brute', n_jobs=1).fit(z)
    indices = np.empty((len(z), k), dtype=np.int64)
    distances = np.empty((len(z), k), dtype=np.float32)
    for start in range(0, len(z), batch_size):
        end = min(start + batch_size, len(z))
        d, j = model.kneighbors(z[start:end], n_neighbors=k + 1)
        for local, row in enumerate(range(start, end)):
            keep = j[local] != row
            indices[row] = j[local][keep][:k]
            distances[row] = d[local][keep][:k]
    if not np.isfinite(distances).all(): raise ValueError('Nonfinite search distances; explicitly rescale or normalize vectors')
    return indices, distances


def validate_neighbors(indices, n, max_k):
    indices = np.asarray(indices)
    if indices.ndim != 2 or len(indices) != n or indices.shape[1] < max_k:
        raise ValueError('neighbors must be N×K with K >= max requested k')
    if not np.issubdtype(indices.dtype, np.integer) or np.any((indices < 0) | (indices >= n)):
        raise ValueError('neighbors must contain integer DataFrame row POSITIONS in [0, N)')
    if np.any(indices == np.arange(n)[:, None]):
        raise ValueError('Neighbor graph contains self edges')
    if any(len(set(row)) != len(row) for row in indices):
        raise ValueError('Neighbor graph contains duplicate neighbors')
    return indices


def membership(values, names, kind):
    if kind == 'multilabel':
        return values
    return np.column_stack([values == name for name in names]).astype(np.uint8)


def purity_lift(values, valid, names, kind, neighbors, ks, baseline=None, anchor_mask=None):
    """Conditional positive-label purity. No negative-agreement inflation for tags."""
    members = membership(values, names, kind)
    counts = members[valid].sum(axis=0).astype(int)
    n_known = int(valid.sum())
    anchor_known=valid if anchor_mask is None else valid & anchor_mask
    label_rows, summaries = [], []
    for k in ks:
        idx = neighbors[:, :k]
        known = valid[idx]
        denominator = known.sum(axis=1)
        for col, name in enumerate(names):
            anchors = anchor_known & (members[:, col] == 1) & (denominator > 0)
            if not anchors.any():
                continue
            hits = (members[idx, col] * known).sum(axis=1)
            item_purity = hits[anchors] / denominator[anchors]
            prevalence = counts[col] / n_known
            base = float(baseline.get(name, prevalence)) if baseline is not None else prevalence
            finite_base = (counts[col] - 1) / (n_known - 1) if n_known > 1 else np.nan
            purity = float(item_purity.mean())
            label_rows.append(dict(k=k, label=name, count=int(counts[col]), n_anchors=int(anchors.sum()),
                prevalence=prevalence, baseline_prevalence=base, purity=purity,
                lift=purity / base if base > 0 else np.nan,
                leave_one_out_chance=finite_base,
                leave_one_out_lift=purity / finite_base if finite_base > 0 else np.nan,
                effective_neighbors=float(denominator[anchors].mean())))
        rows = [r for r in label_rows if r['k'] == k]
        if not rows:
            continue
        weights = np.array([r['n_anchors'] for r in rows])
        purity = np.array([r['purity'] for r in rows])
        lifts = np.array([r['lift'] for r in rows])
        summaries.append(dict(k=k, n_known=n_known, n_labels=len(rows),
            n_anchors=int(weights.sum()), micro_purity=float(np.average(purity, weights=weights)),
            macro_purity=float(purity.mean()), micro_lift=float(np.average(lifts, weights=weights)),
            macro_lift=float(lifts.mean()),
            chance_purity=float(np.average([r['baseline_prevalence'] for r in rows], weights=weights)),
            finite_chance_purity=float(np.average([r['leave_one_out_chance'] for r in rows], weights=weights)),
            known_neighbor_share=float(known[anchor_known].mean()) if anchor_known.any() else np.nan))
    return pd.DataFrame(summaries), pd.DataFrame(label_rows)


def continuous_geometry(values, valid, neighbors, ks, anchor_mask=None):
    y = np.asarray(values, dtype=float)
    centered = y - np.mean(y[valid])
    global_var = np.var(y[valid])
    anchor_known=valid if anchor_mask is None else valid & anchor_mask
    rows = []
    for k in ks:
        idx = neighbors[:, :k]
        edge_valid = anchor_known[:, None] & valid[idx]
        i, position = np.where(edge_valid)
        j = idx[i, position]
        if not len(i):
            continue
        safe = np.where(valid[idx], centered[idx], np.nan)
        # Rowwise variance among known neighbors (without all-NaN warnings).
        num = valid[idx].sum(axis=1)
        sums = np.nansum(safe, axis=1)
        means = np.divide(sums, num, out=np.zeros(len(y)), where=num > 0)
        sums_sq = np.nansum(safe ** 2, axis=1)
        var = np.maximum(0, np.divide(sums_sq, num, out=np.zeros(len(y)), where=num > 0) - means ** 2)
        anchors = anchor_known & (num > 0)
        denom = np.sum(centered[valid] ** 2)
        moran = valid.sum() / len(i) * np.sum(centered[i] * centered[j]) / denom if denom else np.nan
        rows.append(dict(k=k, n_known=int(valid.sum()), n_edges=len(i),
            mean_neighbor_abs_difference=float(np.abs(y[i] - y[j]).mean()),
            mean_local_variance=float(var[anchors].mean()),
            variance_ratio_to_global=float(var[anchors].mean() / global_var) if global_var else np.nan,
            morans_i=float(moran), known_neighbor_share=float(edge_valid[anchor_known].mean())))
    return pd.DataFrame(rows)


def distance_relationship(z, y, valid, metric='cosine', seed=42, cap=500, pairs=20000):
    """Quadratic distance correlation capped explicitly; sampled distance/rank test."""
    rng = np.random.default_rng(seed)
    ids = np.flatnonzero(valid)
    if len(ids) < 3:
        return {'n_distance_correlation': len(ids)}
    sub = rng.choice(ids, min(len(ids), cap), replace=False)
    a = cdist(z[sub], z[sub], metric=metric)
    b = np.abs(y[sub, None] - y[sub][None, :])
    a = a - a.mean(0)[None, :] - a.mean(1)[:, None] + a.mean()
    b = b - b.mean(0)[None, :] - b.mean(1)[:, None] + b.mean()
    denom = np.sqrt(np.mean(a*a) * np.mean(b*b))
    dcor = np.sqrt(max(0, np.mean(a*b)) / denom) if denom else np.nan
    i = rng.choice(ids, pairs); j = rng.choice(ids, pairs)
    keep = i != j; i, j = i[keep], j[keep]
    if metric == 'cosine':
        from .normalization import l2_normalize
        unit=l2_normalize(z)
        zd = np.clip(1 - np.sum(unit[i]*unit[j], axis=1),0,2)
    else:
        zd = np.linalg.norm(z[i]-z[j], axis=1)
    yd = np.abs(y[i]-y[j])
    rank = spearmanr(zd, yd).statistic if np.std(yd) > 0 and np.std(zd) > 0 else np.nan
    return dict(**({'distance_correlation':float(dcor)} if metric=='euclidean' else {'centered_distance_association':float(dcor)}), distance_statistic='classical Euclidean distance correlation' if metric=='euclidean' else 'cosine centered-distance association; no independence guarantee', sampled_distance_rows=sub.tolist(), n_distance_correlation=len(sub),
                distance_spearman=float(rank), n_sampled_pairs=len(i))
