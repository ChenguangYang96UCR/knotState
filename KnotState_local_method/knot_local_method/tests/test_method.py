"""Mathematical invariants, sampling constraints, and evaluation isolation."""
import unittest
from unittest.mock import patch
import networkx as nx
import numpy as np
from knot_local_method import features
from knot_local_method.evaluate import select_and_test


class ExtractionTests(unittest.TestCase):
    def test_known_nullities(self):
        for graph, expected in [(nx.path_graph(3), 1), (nx.complete_graph(3), 1),
                                (nx.empty_graph(3), 3), (nx.complete_bipartite_graph(2, 3), 3)]:
            self.assertEqual(features.nullity({v:set(graph[v]) for v in graph}, list(graph)), expected)

    def test_exact_profile_and_mask(self):
        result = features.extract(nx.path_graph(3), [0, 1], max_order=3, budget=100)
        keys = {tuple(k):i for i,k in enumerate(result['keys'])}
        self.assertEqual(result['values'][0, keys[2, 2, 0]], 1)
        self.assertFalse(result['mask'][0, keys[2, 1, 0]])
        self.assertEqual(result['values'][1, keys[2, 1, 0]], 1)
        self.assertEqual(result['values'][1, keys[3, 2, 1]], 1)
        self.assertTrue(all(row[-1] for row in result['blocks']))

    def test_constraints_and_budget(self):
        G = nx.cycle_graph(12)
        H = [0, 1, 2, 3]
        original = features.nullity
        observed = []
        def record(adj, nodes):
            observed.append(list(nodes))
            return original(adj, nodes)
        with patch.object(features, 'nullity', side_effect=record):
            result = features.extract(G, H, radius=2, budget=60, max_order=4)
        self.assertLessEqual(len(observed), 60)
        cursor = 0
        adj = {v:set(G[v]) for v in G}
        for root_idx, s, a, total, count, exact in result['blocks']:
            root = result['roots'][root_idx]
            for nodes in observed[cursor:cursor+count]:
                self.assertIn(root, nodes)
                self.assertEqual(len(set(nodes)), s)
                self.assertEqual(len(set(nodes) & set(H)), a)
                self.assertTrue(set(nodes) <= features.neighborhood(adj, [root], 2))
            cursor += count

    def test_exact_relabeling(self):
        G = nx.path_graph(5)
        mapping = {i:10-i for i in G}
        a = features.extract(G, [0, 1, 2], radius=2, budget=10000)
        b = features.extract(nx.relabel_nodes(G, mapping), [10, 9, 8], radius=2, budget=10000)
        positions = {v:i for i,v in enumerate(b['roots'])}
        for i, v in enumerate(a['roots']):
            np.testing.assert_allclose(a['values'][i], b['values'][positions[mapping[v]]])

    def test_sampling_reproducibility_and_accuracy(self):
        G = nx.gnp_random_graph(14, .3, seed=7)
        kwargs = dict(radius=2, max_order=4, max_roots=1, seed=3)
        exact = features.extract(G, list(G), budget=100000, **kwargs)
        sampled = features.extract(G, list(G), budget=80, **kwargs)
        repeat = features.extract(G, list(G), budget=80, **kwargs)
        np.testing.assert_array_equal(sampled['values'], repeat['values'])
        self.assertTrue(any(not row[-1] for row in sampled['blocks']))
        # Average independent draws at the same root to reduce stochastic test noise.
        estimates = [features.extract(G, [0], radius=2, max_order=4, budget=80, seed=i)['values']
                     for i in range(30)]
        truth = features.extract(G, [0], radius=2, max_order=4, budget=100000)['values']
        np.testing.assert_allclose(np.mean(estimates, axis=0), truth, atol=.06)

    def test_target_baseline_and_empty_neighborhood(self):
        result = features.extract(nx.path_graph(3), [0, 1], mode='target', budget=100, max_order=3)
        keys = {tuple(k):i for i,k in enumerate(result['keys'])}
        self.assertEqual(result['values'][0, keys[2, 1, 0]], .5)
        self.assertEqual(result['values'][0, keys[2, 1, 2]], .5)
        empty = features.extract(nx.empty_graph(1), [0], budget=10)
        pooled, _ = features.pool(empty, 'stats')
        self.assertEqual(empty['evaluations'], 0)
        self.assertTrue(np.isfinite(pooled).all())
        self.assertFalse(empty['mask'].any())

    def test_multilabel_fusion_test_isolation(self):
        X = np.arange(48).reshape(24, 2) % 7
        y = np.asarray([[i % 2, (i // 2) % 2] for i in range(24)])
        splits = np.asarray(['train']*12 + ['val']*6 + ['test']*6)
        columns = [['mean', 2, 1, 0], ['mean', 3, 1, 1]]
        a = select_and_test(X,y,splits,columns,[2,3],[.1,1], X@X.T)
        changed = y.copy(); changed[18:] = 1-changed[18:]
        b = select_and_test(X,changed,splits,columns,[2,3],[.1,1], X@X.T)
        self.assertEqual(a[0], b[0])
        np.testing.assert_array_equal(a[3], b[3])

    def test_validation_selection_ignores_test_labels(self):
        X = np.arange(24).reshape(12, 2) % 5
        y = np.asarray([0,1]*6)
        splits = np.asarray(['train']*6 + ['val']*2 + ['test']*4)
        columns = [['mean', 2, 1, 0], ['mean', 3, 1, 1]]
        a = select_and_test(X,y,splits,columns,[2,3],[.1,1])
        changed = y.copy(); changed[8:] = 1-changed[8:]
        b = select_and_test(X,changed,splits,columns,[2,3],[.1,1])
        self.assertEqual(a[0], b[0])
        self.assertTrue(all('test_micro_f1' not in row for row in a[0]))
        np.testing.assert_array_equal(a[3], b[3])


if __name__ == '__main__':
    unittest.main()
