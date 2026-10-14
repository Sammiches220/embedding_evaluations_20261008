"""Small, explicit geometry functions; no domain wrappers or trained detectors.

All distances are Euclidean chord lengths on unit-normalized vectors. This
preserves cosine-neighbor ordering, supplies a metric for path stretch, and
avoids interpreting squared distances as intrinsic-dimension radii.
"""
import hashlib
import heapq
import json
import time
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

EPS = 1e-8


def normalize(values):
    """Reject invalid/zero rows, rather than silently changing corpus membership."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2 or len(x) == 0:
        raise ValueError("Expected a nonempty numeric matrix of shape (items, dimensions>=2).")
    bad = ~np.isfinite(x).all(axis=1) | (np.linalg.norm(x, axis=1) <= EPS)
    if bad.any():
        raise ValueError(f"Invalid/zero embedding rows {np.flatnonzero(bad)[:20].tolist()}; filter explicitly and keep aligned IDs.")
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def knn(x, k, queries=None, exclude_self=False):
    """Exact, batched search. Self is removed by ID, including tied duplicates."""
    query = x if queries is None else queries
    extra = int(exclude_self)
    if k + extra > len(x):
        raise ValueError("Not enough reference points for requested k.")
    index = NearestNeighbors(algorithm="brute", metric="euclidean", n_jobs=1).fit(x)
    ds, ns = [], []
    with threadpool_limits(limits=1):
        for start in range(0, len(query), 256):
            d, ids = index.kneighbors(query[start:start + 256], n_neighbors=k + extra)
            if exclude_self:
                out_d, out_i = [], []
                for offset, (dd, ii) in enumerate(zip(d, ids)):
                    keep = ii != start + offset
                    out_d.append(dd[keep][:k])
                    out_i.append(ii[keep][:k])
                d, ids = np.asarray(out_d), np.asarray(out_i)
            ds.append(d)
            ns.append(ids)
    return np.maximum(np.concatenate(ds), 0), np.concatenate(ns)


def local_dimension(distances):
    """Neighbor-radius log-ratio estimate; ties/duplicates return NaN.

This is a local proxy, especially unreliable with few neighbors or curvature.
The small finite-sample correction uses (k-2), rather than (k-1).
"""
    k = distances.shape[1]
    denom = np.log(np.maximum(distances[:, -1:], EPS) / np.maximum(distances[:, :-1], EPS)).sum(axis=1)
    valid = (distances[:, 0] > EPS) & (denom > EPS) & (k >= 4)
    out = np.full(len(distances), np.nan)
    out[valid] = (k - 2) / denom[valid]
    return out


def neighbor_overlap(a, b):
    """Row-wise Jaccard; both arrays refer to the same point-identity universe."""
    return np.array([len(set(x) & set(y)) / max(1, len(set(x) | set(y))) for x, y in zip(a, b)])


def sparse_graph(n, edges, weights):
    if len(edges) == 0:
        return csr_matrix((n, n), dtype=float)
    u, v = edges.T
    # Sparse graph algorithms treat absent entries as no edge; duplicates need
    # positive machine-scale weights to remain connected.
    w = np.maximum(weights, EPS)
    return csr_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n))


def multiscale_edges(neighbors, distances, scales):
    """Mutual kNN edges. Frequency across nested k is rank support, not stability."""
    counts, lengths = Counter(), {}
    labels = []
    n = len(neighbors)
    for k in scales:
        sets = [set(row[:k]) for row in neighbors]
        current = []
        for i, row in enumerate(neighbors[:, :k]):
            for rank, j in enumerate(row):
                j = int(j)
                if i < j and i in sets[j]:
                    edge = (i, j)
                    counts[edge] += 1
                    lengths[edge] = float(distances[i, rank])
                    current.append(edge)
        es = np.asarray(current, dtype=int).reshape(-1, 2)
        labels.append(connected_components(sparse_graph(n, es, np.ones(len(es))), directed=False)[1])
    edges = np.asarray(sorted(counts), dtype=int).reshape(-1, 2)
    return edges, np.array([lengths[tuple(e)] for e in edges]), np.array([counts[tuple(e)] / len(scales) for e in edges]), np.array(labels)


def h0_filtration(n, edges, lengths):
    """Exact H0 of the sparse distance filtration, NOT full Vietoris-Rips PH.

All vertices are born at zero. Each merge kills one component. Remaining
components have infinite death, encoded as null in JSON.
"""
    parent = np.arange(n)
    size = np.ones(n, dtype=int)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    merges = []
    for ei in np.argsort(lengths, kind="stable"):
        u, v = map(int, edges[ei])
        a, b = root(u), root(v)
        if a == b:
            continue
        if size[a] < size[b]:
            a, b = b, a
        merges.append({"death": float(lengths[ei]), "left_size": int(size[a]), "right_size": int(size[b]), "edge": [u, v]})
        parent[b] = a
        size[a] += size[b]
    return {"birth": 0.0, "merges": merges, "infinite_bars": n - len(merges), "scope": "distance filtration on maximum-k mutual graph"}


def deletion_influence(graph):
    """Exact connectivity damage on a FIXED sparse graph, in O(V+E).

Fraction of remaining same-component pairs disconnected by deleting a vertex.
Neighbors are NOT refilled: this is an ablation proxy, not full leave-one-out.
"""
    n = graph.shape[0]
    disc, low, parent = np.full(n, -1), np.zeros(n, int), np.full(n, -1)
    subtree = np.ones(n, int)
    pieces = [[] for _ in range(n)]
    result = np.zeros(n)
    clock = 0
    for root in range(n):
        if disc[root] >= 0:
            continue
        disc[root] = low[root] = clock
        clock += 1
        stack = [(root, iter(graph.indices[graph.indptr[root]:graph.indptr[root + 1]]))]
        members = [root]
        while stack:
            u, it = stack[-1]
            v = next(it, None)
            if v is not None:
                v = int(v)
                if disc[v] < 0:
                    parent[v] = u
                    disc[v] = low[v] = clock
                    clock += 1
                    members.append(v)
                    stack.append((v, iter(graph.indices[graph.indptr[v]:graph.indptr[v + 1]])))
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
            else:
                stack.pop()
                p = parent[u]
                if p >= 0:
                    subtree[p] += subtree[u]
                    low[p] = min(low[p], low[u])
                    if low[u] >= disc[p]:
                        pieces[p].append(int(subtree[u]))
        total = len(members) - 1
        for u in members:
            parts = pieces[u] + [total - sum(pieces[u])]
            if total > 1:
                result[u] = (total * total - sum(s * s for s in parts)) / (total * (total - 1))
    return result


def route(graph, start, end, forbidden_edge=None):
    """Dijkstra with actual settled-node count and optional edge ablation."""
    n = graph.shape[0]
    if not (0 <= start < n and 0 <= end < n):
        raise ValueError("Route endpoint outside corpus.")
    distance, previous = np.full(n, np.inf), np.full(n, -1)
    distance[start] = 0
    queue, expanded = [(0.0, start)], 0
    while queue:
        cost, u = heapq.heappop(queue)
        if cost > distance[u]:
            continue
        expanded += 1
        if u == end:
            path = [end]
            while path[-1] != start:
                path.append(int(previous[path[-1]]))
            return {"path": path[::-1], "cost": cost, "expansions": expanded, "reachable": True}
        for p in range(graph.indptr[u], graph.indptr[u + 1]):
            v, weight = int(graph.indices[p]), graph.data[p]
            if forbidden_edge is not None and {u, v} == set(forbidden_edge):
                continue
            candidate = cost + weight
            if candidate < distance[v]:
                distance[v], previous[v] = candidate, u
                heapq.heappush(queue, (candidate, v))
    return {"path": [], "cost": None, "expansions": expanded, "reachable": False}


def sampled_betweenness(graph, sources):
    """Unweighted Brandes sample: shortest hop paths, distinct from route costs."""
    from collections import deque
    n = graph.shape[0]
    score = np.zeros(n)
    for source in sources:
        pred = [[] for _ in range(n)]
        sigma, distance = np.zeros(n), np.full(n, -1)
        sigma[source], distance[source] = 1, 0
        queue, order = deque([int(source)]), []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in graph.indices[graph.indptr[u]:graph.indptr[u + 1]]:
                if distance[v] < 0:
                    queue.append(v)
                    distance[v] = distance[u] + 1
                if distance[v] == distance[u] + 1:
                    sigma[v] += sigma[u]
                    pred[v].append(u)
        delta = np.zeros(n)
        for v in order[::-1]:
            for u in pred[v]:
                delta[u] += sigma[u] / sigma[v] * (1 + delta[v])
            if v != source:
                score[v] += delta[v]
    return score / max(1, len(sources) * (n - 1))


def region_partition(n, edges, lengths, target):
    """Graph Voronoi navigation cells, with farthest-point seeds.

Every connected component is covered. `target` budgets cells for components
with at least max(20, 2% of n) members; small components get one cell each.
Allocate additional seeds by population per seed, then use graph distance.
These are navigation bins, not semantic or persistent clusters.
    """
    graph = sparse_graph(n, edges, lengths)
    count, component = connected_components(graph, directed=False)
    members = [np.flatnonzero(component == c) for c in range(count)]
    big = [c for c in range(count) if len(members[c]) >= max(20, int(.02 * n))]
    budgets = np.ones(count, dtype=int)
    for _ in range(max(0, target - len(big))):
        eligible = [c for c in big if budgets[c] < len(members[c])]
        if not eligible: break
        c = max(eligible, key=lambda c: len(members[c]) / budgets[c])
        budgets[c] += 1
    labels = np.zeros(n, int)
    next_label = 0
    for c, inds in enumerate(members):
        if budgets[c] == 1:
            labels[inds] = next_label
            next_label += 1
            continue
        local = graph[inds][:, inds]
        seeds = [0]
        distance = dijkstra(local, directed=False, indices=0)
        while len(seeds) < budgets[c]:
            distance[seeds] = -1
            j = int(np.argmax(distance))
            seeds.append(j)
            distance = np.minimum(distance, dijkstra(local, directed=False, indices=j))
        distances = dijkstra(local, directed=False, indices=seeds)
        labels[inds] = np.argmin(distances, axis=0) + next_label
        next_label += len(seeds)
    return labels


def compression_profiles(x, neighbors, k, seed):
    """Fit corpus-only PCA; compare unrenormalized compressed Euclidean views.

The unrenormalized projection preserves distance units and isolates discarded
coordinates. Out-of-sample scoring reuses these exact transformations.
"""
    maximum = min(x.shape[1], len(x) - 1)
    ranks = sorted(set([max(2, maximum // 4), max(2, maximum // 2), max(2, int(maximum * .85))]))
    ranks = [min(maximum, r) for r in ranks]
    with threadpool_limits(limits=1):
        pca = PCA(n_components=maximum, svd_solver="full").fit(x)
    centered = x - pca.mean_
    scores, views = {}, {}
    for rank in ranks:
        y = centered @ pca.components_[:rank].T
        _, ids = knn(y, k, exclude_self=True)
        scores[f"pca_{rank}"] = neighbor_overlap(neighbors[:, :k], ids)
        views[f"pca_{rank}"] = y
    rng = np.random.default_rng(seed + 77)
    mask = np.sort(rng.choice(x.shape[1], max(1, x.shape[1] // 2), replace=False))
    _, masked = knn(x[:, mask], k, exclude_self=True)
    scores["coordinate_mask"] = neighbor_overlap(neighbors[:, :k], masked)
    return scores, pca.mean_, pca.components_, mask, views


def resampling_profiles(x, neighbors, k, repeats, seed):
    """80% reference subsamples; Jaccard against eligible original neighbors.

Queries are all original vectors, but their self identity is explicitly
excluded when retained. Compare only within the retained identity universe.
"""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        keep = np.sort(rng.choice(len(x), max(k + 2, int(.8 * len(x))), replace=False))
        _, ids = knn(x[keep], k + 1, queries=x)
        retained = set(keep)
        row_scores = []
        for i, row in enumerate(ids):
            # Limit to first k after excluding self, not an unordered truncation.
            actual = set([int(keep[j]) for j in row if keep[j] != i][:k])
            eligible = set(neighbors[i, :k]) & retained
            union = eligible | actual
            row_scores.append(len(eligible & actual) / max(1, len(union)))
        values.append(row_scores)
    return np.asarray(values)


def percentile(values):
    """Empirical midranks; undefined inputs remain undefined."""
    from scipy.stats import rankdata
    values = np.asarray(values)
    out = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if valid.any():
        out[valid] = (rankdata(values[valid], method="average") - .5) / valid.sum()
    return out


def build_map(values, ids=None, *, scales=(6, 12, 24), support=.5,
              repeats=4, landmarks=32, regions=8, seed=42, encoder="unspecified"):
    """Freeze a corpus, map it, compute signatures, and return a plain dict.

Exact backend is a feasibility reference, not an eight-million-row engine.
"""
    started = time.perf_counter()
    x = normalize(values)
    n, dim = x.shape
    scales = sorted(set(map(int, scales)))
    if n < 12 or not scales or min(scales) < 4 or max(scales) >= n - 1:
        raise ValueError("Need n>=12 and unique neighborhood scales 4 <= k < n-1.")
    if not 0 < support <= 1 or repeats < 2 or landmarks < 1 or regions < 1:
        raise ValueError("Require support in (0,1], repeats>=2, landmarks>=1, regions>=1.")
    ids = [str(i) for i in range(n)] if ids is None else list(map(str, ids))
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError("IDs must be unique and aligned one-to-one with rows.")
    k = scales[len(scales) // 2]
    # Retain one extra neighbor for out-of-sample reverse-neighbor thresholds.
    distances, neighbors = knn(x, max(scales) + 1, exclude_self=True)
    timings = {"neighbors_s": time.perf_counter() - started}
    edges, lengths, frequency, scale_components = multiscale_edges(neighbors, distances, scales)
    keep = frequency >= support
    stable_edges, stable_lengths, stable_frequency = edges[keep], lengths[keep], frequency[keep]
    graph = sparse_graph(n, stable_edges, stable_lengths / (stable_frequency + .01))
    components, component = connected_components(graph, directed=False)
    region = region_partition(n, stable_edges, stable_lengths, regions)
    # Per-point scale support averages all incident max-k mutual edges.
    sums, degree = np.zeros(n), np.zeros(n)
    for e, f in zip(edges, frequency):
        sums[e] += f
        degree[e] += 1
    scale_support = np.divide(sums, degree, out=np.zeros(n), where=degree > 0)
    compression, mean, axes, mask, _ = compression_profiles(x, neighbors, k, seed)
    sampled = resampling_profiles(x, neighbors, k, repeats, seed + 1)
    # Raw Jaccard contains unavoidable loss from missing observations. Conditional
    # recall is separately reported by validation; do not call Jaccard failure.
    perturb = np.random.default_rng(seed + 2).normal(size=x.shape)
    perturb /= np.linalg.norm(perturb, axis=1, keepdims=True)
    noisy = normalize(x + .01 * perturb)
    _, noise_neighbors = knn(noisy, k, exclude_self=True)
    perturb_stability = neighbor_overlap(neighbors[:, :k], noise_neighbors)
    hubness = np.bincount(neighbors[:, :k].ravel(), minlength=n)
    dimension = local_dimension(distances[:, :k])
    influence = deletion_influence(graph)
    betweenness = sampled_betweenness(graph, np.random.default_rng(seed).choice(n, min(24, n), replace=False))
    boundary = np.mean(region[neighbors[:, :k]] != region[:, None], axis=1)
    # Local next-hop entropy is explicitly NOT entropy of complete routes.
    choice = np.zeros(n)
    for i in range(n):
        w = graph.data[graph.indptr[i]:graph.indptr[i + 1]]
        if len(w):
            prob = 1 / np.maximum(w, EPS)
            prob /= prob.sum()
            choice[i] = np.exp(-np.sum(prob * np.log(prob)))
    profiles = {
        "cohesion_distance": distances[:, :k].mean(axis=1),
        "neighbor_radius": distances[:, k - 1],
        "inverse_radius": 1 / np.maximum(distances[:, k - 1], EPS),
        "local_dimension": dimension,
        "scale_support": scale_support,
        "region_boundary": boundary,
        "hubness": hubness,
        "resample_jaccard": sampled.mean(axis=0),
        "resample_p10": np.quantile(sampled, .1, axis=0),
        "resample_p90": np.quantile(sampled, .9, axis=0),
        "perturbation_jaccard": perturb_stability,
        "deletion_pair_loss": influence,
        "sampled_betweenness": betweenness,
        "effective_next_hops": choice,
        **compression,
    }
    # Deliberately overlapping flags, not a mutually exclusive value ranking.
    radius_pct, dim_pct = percentile(profiles["neighbor_radius"]), percentile(dimension)
    roles = []
    for i in range(n):
        tags = []
        if graph.indptr[i] == graph.indptr[i + 1]: tags.append("isolated in support graph")
        if influence[i] > 0: tags.append("fixed-graph articulation")
        if boundary[i] > .25: tags.append("cell boundary")
        if radius_pct[i] < .2: tags.append("dense neighborhood")
        if dim_pct[i] > .8: tags.append("complex local geometry")
        if perturb_stability[i] < .5: tags.append("perturbation-sensitive")
        if radius_pct[i] > .95: tags.append("sparse neighborhood")
        if not tags: tags.append("locally connected")
        roles.append(tags)
    # Cover every navigation cell, then extremes, then farthest-point coverage.
    selected = []
    for label in np.unique(region):
        members = np.flatnonzero(region == label)
        selected.append(int(members[np.argmin(profiles["cohesion_distance"][members])]))
    target = max(landmarks, len(selected))
    for key in ["deletion_pair_loss", "sampled_betweenness", "local_dimension", list(compression)[0]]:
        order = np.argsort(np.nan_to_num(profiles[key], nan=-1))[::-1]
        for i in order[:2]:
            if int(i) not in selected and len(selected) < target: selected.append(int(i))
    nearest = np.full(n, np.inf)
    for j in selected:
        nearest = np.minimum(nearest, np.linalg.norm(x - x[j], axis=1))
    while len(selected) < min(target, n):
        # Avoid repeatedly selecting a zero-distance duplicate.
        nearest[selected] = -1
        i = int(np.argmax(nearest))
        selected.append(i)
        nearest = np.minimum(nearest, np.linalg.norm(x - x[i], axis=1))
    with threadpool_limits(limits=1):
        display_pca = PCA(n_components=2, svd_solver="full").fit(x)
        xy = display_pca.transform(x)
    _, projected_neighbors = knn(xy, k, exclude_self=True)
    profiles["projection_jaccard"] = neighbor_overlap(neighbors[:, :k], projected_neighbors)
    manifest = {
        "schema_version": 1, "encoder": encoder, "n": n, "dimensions": dim,
        "metric": "unit-normalized Euclidean chord", "normalization": "L2 per row, float32",
        "backend": "exact batched brute-force", "scales": scales, "signature_k": k,
        "edge_support_threshold": support, "resamples": repeats, "seed": seed,
        "corpus_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "ids_sha256": hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest(),
        "duplicate_rows": n - len(np.unique(x, axis=0)),
        "navigation_partition": "graph Voronoi; farthest-point seeds; small components preserved",
        "requested_large_component_cells": regions,
        "components": int(components), "regions": int(region.max() + 1),
        "display_explained_variance": float(display_pca.explained_variance_ratio_.sum()),
        "timings": timings,
    }
    result = dict(x=x, ids=ids, neighbors=neighbors, distances=distances,
                  edges=edges, lengths=lengths, frequency=frequency,
                  stable_edges=stable_edges, stable_lengths=stable_lengths,
                  stable_frequency=stable_frequency, graph=graph,
                  component=component, scale_components=scale_components, region=region,
                  profiles=profiles, roles=roles, landmarks=np.array(selected), xy=xy,
                  h0=h0_filtration(n, edges, lengths), manifest=manifest,
                  pca_mean=mean, pca_axes=axes, coordinate_mask=mask,
                  display_mean=display_pca.mean_, display_axes=display_pca.components_)
    manifest["timings"]["build_total_s"] = time.perf_counter() - started
    return result


def score_queries(model, values):
    """Attach to a frozen corpus. Never pretend influence was recomputed.

Forward-neighbor attachment mass estimates ambiguity, not semantic posterior
probabilities. Quantile ranges measure reference-subsample sensitivity only.
"""
    q = normalize(values)
    x, k = model["x"], model["manifest"]["signature_k"]
    if q.shape[1] != x.shape[1]:
        raise ValueError("Query dimensions differ from frozen corpus.")
    d, ns = knn(x, k, queries=q)
    dim = local_dimension(d)
    reference_radii = model["distances"][:, k - 1]
    out = []
    rng = np.random.default_rng(model["manifest"]["seed"] + 505)
    subsets = [np.sort(rng.choice(len(x), max(k + 1, int(.8 * len(x))), replace=False)) for _ in range(model["manifest"]["resamples"])]
    query_samples = [knn(x[s], k, queries=q)[0].mean(axis=1) for s in subsets]
    compressed_neighbors = {}
    for name in model["profiles"]:
        if name.startswith("pca_"):
            rank = int(name.split("_")[1])
            y = (x - model["pca_mean"]) @ model["pca_axes"][:rank].T
            z = (q - model["pca_mean"]) @ model["pca_axes"][:rank].T
            compressed_neighbors[name] = knn(y, k, queries=z)[1]
    mask = model["coordinate_mask"]
    compressed_neighbors["coordinate_mask"] = knn(x[:, mask], k, queries=q[:, mask])[1]
    for i in range(len(q)):
        temperature = max(float(np.median(d[i])), EPS)
        p = np.exp(-(d[i] - d[i].min()) / temperature)
        p /= p.sum()
        cells = Counter()
        for j, weight in zip(ns[i], p): cells[int(model["region"][j])] += float(weight)
        probabilities = np.array(list(cells.values()))
        effective = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        comp = {int(c): float(sum(w for j, w in zip(ns[i], p) if model["component"][j] == c)) for c in set(model["component"][ns[i]])}
        compression = {name: float(neighbor_overlap(ns[i:i+1], ids[i:i+1])[0]) for name, ids in compressed_neighbors.items()}
        query_distances = np.linalg.norm(x - q[i], axis=1)
        reverse_count = int(np.sum(query_distances < reference_radii))
        # Exact insertion reverse count under strict < tie policy; not the
        # hubness of a rebuilt, enlarged corpus.
        nearest_landmarks = sorted(model["landmarks"].tolist(), key=lambda j: float(query_distances[j]))[:3]
        landmark_attachments = []
        for landmark in nearest_landmarks:
            eligible = [rank for rank, j in enumerate(ns[i]) if model["component"][j] == model["component"][landmark]]
            if not eligible:
                landmark_attachments.append({"landmark_index": int(landmark), "entry_anchor": None, "path": [], "cost_including_query_entry": None})
                continue
            rank = eligible[0]
            anchor = int(ns[i, rank])
            path = route(model["graph"], anchor, int(landmark))
            landmark_attachments.append({"landmark_index": int(landmark), "entry_anchor": anchor,
                                         "path": path["path"], "cost_including_query_entry": float(d[i, rank] + path["cost"]) if path["reachable"] else None})
        sparsity = float(np.mean(model["profiles"]["neighbor_radius"] <= d[i, k - 1]))
        out.append({
            "neighbors": [{"id": model["ids"][j], "index": int(j), "distance": float(dd)} for j, dd in zip(ns[i], d[i])],
            "cohesion_distance": float(d[i].mean()), "local_dimension": float(dim[i]) if np.isfinite(dim[i]) else None,
            "cohesion_resample_p10_p90": np.quantile(np.array(query_samples)[:, i], [.1, .9]).tolist(),
            "region_attachment_mass": dict(cells), "component_attachment_mass": comp,
            "effective_candidate_regions": effective, "reference_radius_percentile": sparsity,
            "insertion_reverse_neighbor_count": reverse_count,
            "compression": compression, "nearest_landmarks": nearest_landmarks,
            "landmark_attachments": landmark_attachments,
            "neighbor_scale_support_proxy": float(np.dot(p, model["profiles"]["scale_support"][ns[i]])),
            "neighbor_perturbation_stability_proxy": float(np.dot(p, model["profiles"]["perturbation_jaccard"][ns[i]])),
            "attachment_boundary_mass": float(1 - probabilities.max()),
            "structural_influence": None,
            "interpretation": "sparser than 95% of reference neighborhoods" if sparsity > .95 else ("ambiguous cell attachment" if effective > 1.5 else "concentrated cell attachment"),
            "limits": "No new component or persistent bridge is inferred. Ranges are perturbation sensitivity, not confidence intervals. Exact search is O(n*d).",
        })
    return out
