"""Independent controls and geometric recovery; no semantic training labels."""
import time
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, rankdata
from .core import (normalize, knn, build_map, neighbor_overlap, score_queries,
                   route, sparse_graph, multiscale_edges)


def finite_median(values):
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) else None


def auc(positive, negative):
    """Rank statistic for a planted mechanism, never a trained classifier."""
    a, b = np.asarray(positive), np.asarray(negative)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0: return None
    ranks = rankdata(np.r_[a, b])
    return float((ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def summaries(model):
    p = model["profiles"]
    return {key: finite_median(p[key]) for key in ["cohesion_distance", "local_dimension", "resample_jaccard", "perturbation_jaccard", "projection_jaccard", "deletion_pair_loss", "effective_next_hops"]} | {
        "components": model["manifest"]["components"],
        "largest_component_share": float(np.bincount(model["component"]).max() / len(model["x"])),
        "articulations": int(np.sum(p["deletion_pair_loss"] > 0)),
    }


def heldout_recovery(x, seed, scales):
    """Actual 20% held-out vectors; rebuild map on TRAIN ONLY.

Recover each held-out point's nearest reference anchor under mild isotropic
query noise. Full, unperturbed nearest anchors define the geometric target;
this is not a semantic target. Compare raw retrieval, cell-centroid retrieval,
random anchors, and frozen-PCA retrieval. Noise is .05 chord per query.
"""
    rng = np.random.default_rng(seed + 20)
    order = rng.permutation(len(x))
    test_count = max(12, int(.2 * len(x)))
    test_ids, train_ids = order[:test_count], order[test_count:]
    train, test = x[train_ids], x[test_ids]
    map_train = build_map(train, scales=scales, repeats=2, seed=seed + 20, landmarks=16)
    _, true = knn(train, 1, queries=test)
    noise = rng.normal(size=test.shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    query = normalize(test + .05 * noise)
    topk = min(10, len(train))
    distances, predicted = knn(train, topk, queries=query)
    hits = np.any(predicted == true, axis=1)
    top1 = predicted[:, 0] == true[:, 0]
    ranks = []
    for i, q in enumerate(query):
        all_d = np.linalg.norm(train - q, axis=1)
        target_d = all_d[true[i, 0]]
        ranks.append(1 + int(np.sum(all_d < target_d)))
    cells = np.unique(map_train["region"])
    centroids = np.array([train[map_train["region"] == c].mean(axis=0) for c in cells])
    _, coarse = knn(centroids, 1, queries=query)
    target_cell = map_train["region"][true[:, 0]]
    centroid_accuracy = float(np.mean(cells[coarse[:, 0]] == target_cell))
    rank = max(2, train.shape[1] // 4)
    axes, mean = map_train["pca_axes"][:rank], map_train["pca_mean"]
    _, compressed = knn((train - mean) @ axes.T, topk, queries=(query - mean) @ axes.T)
    return {
        "train_n": len(train), "heldout_n": len(test), "test_not_used_to_fit_map": True,
        "query_noise_chord": .05, "target": "unperturbed query's nearest training anchor",
        "anchor_top1": float(top1.mean()), "anchor_top10": float(hits.mean()),
        "anchor_median_rank": float(np.median(ranks)),
        "random_anchor_top10_expectation": topk / len(train),
        "cell_centroid_top1": centroid_accuracy,
        "majority_cell_baseline": float(np.bincount(target_cell).max() / len(target_cell)),
        "compressed_anchor_top10": float(np.mean(np.any(compressed == true, axis=1))),
        "limits": "Direct retrieval is the strong baseline. Cell recovery can be trivial when one cell dominates. This measures geometric anchor recovery, not usefulness to people.",
    }


def route_experiments(model, count=32):
    x, graph = model["x"], model["graph"]
    rng = np.random.default_rng(model["manifest"]["seed"] + 23)
    examples, lengths, costs, expansions, alternate, retained = [], [], [], [], [], []
    # Independently perturb geometry and rebuild graph for route survival.
    noise = rng.normal(size=x.shape)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    perturbed = normalize(x + .01 * noise)
    dd, nn = knn(perturbed, max(model["manifest"]["scales"]), exclude_self=True)
    es, ds, fr, _ = multiscale_edges(nn, dd, model["manifest"]["scales"])
    ok = fr >= model["manifest"]["edge_support_threshold"]
    perturbed_edges = {tuple(e) for e in es[ok]}
    for _ in range(count):
        a, b = map(int, rng.choice(len(x), 2, replace=False))
        r = route(graph, a, b)
        expansions.append(r["expansions"])
        if not r["reachable"]: continue
        path = r["path"]
        raw_length = sum(float(np.linalg.norm(x[u] - x[v])) for u, v in zip(path, path[1:]))
        direct = float(np.linalg.norm(x[a] - x[b]))
        stretch = raw_length / max(direct, 1e-8)
        lengths.append(stretch)
        costs.append(r["cost"] / max(direct, 1e-8))
        edge = (path[len(path) // 2 - 1], path[len(path) // 2])
        ablated = route(graph, a, b, forbidden_edge=edge)
        alternate.append(ablated["reachable"])
        path_edges = [tuple(sorted((u, v))) for u, v in zip(path, path[1:])]
        survival = float(np.mean([e in perturbed_edges for e in path_edges]))
        retained.append(survival)
        examples.append({"start": a, "end": b, "hops": len(path) - 1, "stretch": stretch,
                         "expansions": r["expansions"], "middle_edge_ablation_reachable": ablated["reachable"],
                         "perturbed_edge_retention": survival})
    return {
        "attempted": count, "reachable": len(lengths),
        "median_geometric_stretch": finite_median(lengths),
        "median_support_weighted_cost_ratio": finite_median(costs),
        "median_expansions_all_attempts": finite_median(expansions),
        "middle_edge_ablation_survival": float(np.mean(alternate)) if alternate else None,
        "mean_route_edge_retention_under_noise": float(np.mean(retained)) if retained else None,
        "examples": examples,
        "limits": "Edge retention is not whole-route survival. Alternate-route test deletes one middle edge, not every edge or branch. No global route-entropy estimate is claimed.",
    }


def evaluate(model, metadata=None, include_nulls=True):
    started = time.perf_counter()
    x, seed = model["x"], model["manifest"]["seed"]
    result = {"observed": summaries(model), "heldout": heldout_recovery(x, seed, model["manifest"]["scales"]),
              "routes": route_experiments(model), "synthetic_groups": [], "mechanism_checks": [], "nulls": {}}
    if metadata and all("demo_group" in m for m in metadata):
        labels = np.array([m["demo_group"] for m in metadata])
        for name in dict.fromkeys(labels):
            mask = labels == name
            row = {"group": name, "n": int(mask.sum())}
            row.update({key: finite_median(value[mask]) for key, value in model["profiles"].items()})
            result["synthetic_groups"].append(row)
        p = model["profiles"]
        checks = [
            ("High-dimensional patch vs basins", "local_dimension", labels == "high-dimensional patch", np.isin(labels, ["basin A", "basin B"]), False),
            ("Redundant patch vs basins", "neighbor_radius", labels == "redundant patch", np.isin(labels, ["basin A", "basin B"]), True),
            ("Connector vs basins", "sampled_betweenness", labels == "narrow connector", np.isin(labels, ["basin A", "basin B"]), False),
            ("Outliers vs basins", "neighbor_radius", labels == "random outliers", np.isin(labels, ["basin A", "basin B"]), False),
        ]
        for title, key, positive, negative, reverse in checks:
            values = -p[key] if reverse else p[key]
            result["mechanism_checks"].append({"question": title, "measurement": key, "auc": auc(values[positive], values[negative]),
                                                "interpretation": "Probability a planted positive has a larger oriented measurement than a planted negative; post-hoc synthetic diagnostic."})
    if include_nulls:
        rng = np.random.default_rng(seed + 24)
        # Same n,d; Gaussian null matches empirical mean/covariance in expectation
        # BEFORE normalization, not local density or manifold structure.
        center = x.mean(axis=0)
        u, s, vt = np.linalg.svd(x - center, full_matrices=False)
        gaussian = center + (rng.normal(size=(len(x), len(s))) * (s / np.sqrt(len(x) - 1))) @ vt
        clouds = {"uniform_sphere": normalize(rng.normal(size=x.shape)),
                  "moment_matched_gaussian": normalize(gaussian)}
        for name, values in clouds.items():
            null = build_map(values, scales=model["manifest"]["scales"], repeats=2, seed=seed, landmarks=16)
            result["nulls"][name] = summaries(null)
        result["null_notes"] = "One realization per null: descriptive contrasts, no p-values. Covariance matching is approximate after normalization. Nulls do not preserve local density."
    result["generation_difficulty"] = {
        "landmark_count": len(model["landmarks"]), "scales": len(model["manifest"]["scales"]),
        "distance_constraints_per_target_if_all_landmarks_used": len(model["landmarks"]),
        "measured_generation_success": None,
        "note": "Constraint inventory only. No generator or search-cost lower bound was implemented.",
    }
    result["elapsed_s"] = time.perf_counter() - started
    result["human_interface"] = {
        "status": "Automatic map encodings: shorelines, graph roads, passes, density contours, dimension terrain, and instability fog.",
        "principle": "The map supplies computed terrain and connectivity directly; no participant answers are required.",
        "limits": "Geometric legibility does not establish semantic or cultural value. Independent software and geometry checks are separate from the navigation interface.",
    }
    return result
