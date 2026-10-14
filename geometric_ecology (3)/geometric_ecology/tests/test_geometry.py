"""Numerical and contract tests, runnable without pytest."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import connected_components
from geometric_ecology.core import (normalize, knn, sparse_graph, h0_filtration,
                                    deletion_influence, route, build_map, score_queries)
from geometric_ecology.demo import synthetic_corpus
from geometric_ecology.io import save_model, load_model, export_explorer
from geometric_ecology.validation import heldout_recovery


class GeometryTests(unittest.TestCase):
    def test_normalization_rejects_bad_rows(self):
        for value in [np.zeros((4, 3)), np.full((4, 3), np.nan), np.ones(3), np.full((4, 3), np.inf)]:
            with self.assertRaises(ValueError): normalize(value)
        np.testing.assert_allclose(np.linalg.norm(normalize([[3, 4], [1, 0]]), axis=1), 1)

    def test_neighbors_remove_identity_with_duplicates(self):
        x = normalize([[1, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])
        d, nn = knn(x, 3, exclude_self=True)
        for i in range(len(x)):
            self.assertNotIn(i, nn[i])
            all_d = np.linalg.norm(x - x[i], axis=1)
            all_d[i] = np.inf
            np.testing.assert_allclose(d[i], np.sort(all_d)[:3], atol=1e-6)
        self.assertEqual(d[0, 0], 0)

    def test_sparse_h0_known_components(self):
        edges = np.array([[0, 1], [1, 2], [0, 2], [3, 4]])
        result = h0_filtration(6, edges, np.array([1., 2., 3., 4.]))
        self.assertEqual(result["infinite_bars"], 3)  # {0,1,2}, {3,4}, {5}
        self.assertEqual([r["death"] for r in result["merges"]], [1, 2, 4])

    def test_deletion_matches_brute_force_pair_disconnection(self):
        rng = np.random.default_rng(91)
        for n in [3, 7, 15]:
            edges = np.array([(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < .3]).reshape(-1, 2)
            g = sparse_graph(n, edges, np.ones(len(edges)))
            _, labels = connected_components(g, directed=False)
            actual = deletion_influence(g)
            for i in range(n):
                others = np.flatnonzero((labels == labels[i]) & (np.arange(n) != i))
                remaining = g[others][:, others]
                if len(others) < 2: expected = 0
                else:
                    _, sublabels = connected_components(remaining, directed=False)
                    counts = np.bincount(sublabels)
                    total = len(others)
                    expected = (total * total - np.sum(counts * counts)) / (total * (total - 1))
                self.assertAlmostEqual(actual[i], expected)

    def test_routes_handle_missing_paths_and_ablation(self):
        g = sparse_graph(5, np.array([[0, 1], [1, 2], [0, 2]]), np.array([1., 1., 3.]))
        self.assertEqual(route(g, 0, 2)["path"], [0, 1, 2])
        self.assertEqual(route(g, 0, 2, (0, 1))["path"], [0, 2])
        self.assertFalse(route(g, 0, 4)["reachable"])
        self.assertEqual(route(g, 0, 0)["path"], [0])
        with self.assertRaises(ValueError): route(g, -1, 3)

    def test_end_to_end_scoring_freezes_map_and_safe_roundtrip(self):
        x, ids, meta = synthetic_corpus(240, 16, 12)
        model = build_map(x, ids, scales=[4, 8, 12], repeats=2, landmarks=12)
        original = model["x"].copy()
        scored = score_queries(model, x[[10, 30]])
        np.testing.assert_array_equal(original, model["x"])
        self.assertEqual(scored[0]["neighbors"][0]["id"], ids[10])
        self.assertIsNone(scored[0]["structural_influence"])
        self.assertGreaterEqual(scored[0]["effective_candidate_regions"], .999999)
        self.assertGreaterEqual(scored[0]["insertion_reverse_neighbor_count"], 1)
        self.assertTrue(np.all((model["profiles"]["perturbation_jaccard"] >= 0) & (model["profiles"]["perturbation_jaccard"] <= 1)))
        with tempfile.TemporaryDirectory() as tmp:
            save_model(model, tmp)
            reloaded = load_model(tmp)
            np.testing.assert_array_equal(model["neighbors"], reloaded["neighbors"])
            self.assertEqual(model["manifest"]["corpus_sha256"], reloaded["manifest"]["corpus_sha256"])
            export_explorer(reloaded, tmp, metadata=meta)
            self.assertNotIn("__ECOLOGY_DATA__", (Path(tmp) / "explorer.html").read_text())
            with np.load(Path(tmp) / "map.npz", allow_pickle=False) as archive:
                self.assertFalse(any(archive[k].dtype.hasobject for k in archive.files))
        with self.assertRaises(ValueError): score_queries(model, np.ones((2, 17)))

    def test_heldout_recovery_has_explicit_separate_reference(self):
        x, _, _ = synthetic_corpus(240, 16, 9)
        result = heldout_recovery(x, 42, [4, 8, 12])
        self.assertEqual(result["train_n"] + result["heldout_n"], len(x))
        self.assertTrue(result["test_not_used_to_fit_map"])
        self.assertGreater(result["anchor_top10"], result["random_anchor_top10_expectation"])

if __name__ == "__main__": unittest.main()
