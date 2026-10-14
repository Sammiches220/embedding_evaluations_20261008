"""Aligned embedding comparisons using plain arrays and pandas tables."""
import numpy as np
import pandas as pd


def align_embedding_rows(embeddings, ids, target_ids):
    """Require every target ID once; reorder an optional second embedding by ID."""
    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2 or not z.shape[1]:
        raise ValueError('Comparison embeddings must be a nonempty N×D numeric array')
    if ids is None:
        if len(z) != len(target_ids):
            raise ValueError('Comparison embeddings must match row order, or supply comparison_ids')
        return z
    ids = np.asarray(ids).astype(str)
    if ids.ndim != 1 or len(ids) != len(z) or len(set(ids)) != len(ids):
        raise ValueError('Comparison IDs must be unique and match embedding rows')
    rows = pd.Index(ids).get_indexer(np.asarray(target_ids).astype(str))
    if (rows < 0).any():
        raise ValueError('Comparison is missing source IDs; explicitly restrict both datasets to a common cohort first')
    return z[rows]


def anchor_scores(values, valid, names, kind, graph, k):
    """Return per-anchor score and positive-label weight; undefined anchors are NaN.

    Continuous scores average differences within each anchor, rather than across
    edges. Tag scores average agreement for the anchor's selected positive tags.
    """
    idx = graph[:, :k]
    known = valid[idx]
    count = known.sum(axis=1)
    scores = np.full(len(valid), np.nan)
    weights = np.ones(len(valid), dtype=float)
    usable = valid & (count > 0)
    if kind == 'continuous':
        differences = np.where(known, np.abs(values[:, None] - values[idx]), 0.)
        scores[usable] = differences[usable].sum(axis=1) / count[usable]
    elif kind == 'categorical':
        hits = ((values[:, None] == values[idx]) & known).sum(axis=1)
        scores[usable] = hits[usable] / count[usable]
    else:
        weights = values.sum(axis=1).astype(float)
        usable &= weights > 0
        hits = np.zeros(len(valid), dtype=float)
        for col in range(len(names)):
            anchors = np.flatnonzero(usable & (values[:, col] == 1))
            hits[anchors] += (values[idx[anchors], col] * known[anchors]).sum(axis=1)
        scores[usable] = hits[usable] / count[usable] / weights[usable]
    return scores, weights


def paired_difference(a, b, weights, groups=None, repetitions=200, seed=42):
    """Weighted paired means and bootstrap interval conditional on fixed graphs.

    Resample whole groups if given. This does not retrain embeddings/rebuild
    graphs, or estimate approximate-search error. Return None for unavailable CI.
    """
    mask = np.isfinite(a) & np.isfinite(b) & (weights > 0)
    a, b, w = a[mask], b[mask], weights[mask]
    if not len(a):
        return dict(n_paired_anchors=0, a=None, b=None, delta=None, ci_low=None, ci_high=None,
                    n_resampling_units=0)
    if groups is None:
        inverse = np.arange(len(a))
    else:
        _, inverse = np.unique(np.asarray(groups)[mask].astype(str), return_inverse=True)
    n = int(inverse.max()) + 1
    numerator = np.bincount(inverse, weights=(b-a)*w, minlength=n)
    denominator = np.bincount(inverse, weights=w, minlength=n)
    out = dict(n_paired_anchors=len(a), a=float(np.average(a, weights=w)),
               b=float(np.average(b, weights=w)), delta=float(np.average(b-a, weights=w)),
               ci_low=None, ci_high=None, n_resampling_units=n)
    if repetitions and n >= 2:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(repetitions):
            sample = rng.integers(0, n, size=n)
            draws.append(numerator[sample].sum() / denominator[sample].sum())
        out['ci_low'], out['ci_high'] = map(float, np.quantile(draws, [.025, .975]))
    return out


def compare_graphs(encoded, graph_a, graph_b, ks, groups=None, repetitions=200, seed=42):
    """Matched anchor summaries per feature and k; weights and scope are explicit."""
    rows = []
    for feature, record in encoded.items():
        y, valid, names, kind = record['values'], record['valid'], record['names'], record['kind']
        for k in ks:
            a, weights = anchor_scores(y, valid, names, kind, graph_a, k)
            b, _ = anchor_scores(y, valid, names, kind, graph_b, k)
            paired = paired_difference(a, b, weights, groups, repetitions, seed)
            rows.append(dict(feature=feature, kind=kind, k=k,
                metric='paired_anchor_mean_abs_difference' if kind == 'continuous' else 'paired_micro_purity',
                better_direction='lower' if kind == 'continuous' else 'higher',
                resampling='groups' if groups is not None else 'anchors', **paired))
    return pd.DataFrame(rows)
