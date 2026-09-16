#!/usr/bin/env python3
"""Core KnotState computations.

Two representations are exposed:

1) Global order-resolved interlace state profile
   K_{s,nu}(G) = P[ nullity_F2(A[S]) = nu | |S|=s ].

2) Target-conditioned profile used in the subgraph experiments
   K^{(r)}_{s,a,nu}(G,H) = P[ nullity_F2(A[S])=nu |
       |S|=s, |S cap H|=a, S subset H union B_r(H) ].

For a simple undirected graph the adjacency matrix over F2 is alternating, so
only nullities nu with nu == s (mod 2) can occur.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import networkx as nx
import numpy as np


def allowed_nullities(s: int) -> List[int]:
    return list(range(s % 2, s + 1, 2))


def gf2_rank_bits(rows: Sequence[int]) -> int:
    basis: Dict[int, int] = {}
    rank = 0
    for row in rows:
        x = int(row)
        while x:
            p = x.bit_length() - 1
            if p in basis:
                x ^= basis[p]
            else:
                basis[p] = x
                rank += 1
                break
    return rank


def adjacency_sets(G: nx.Graph) -> Dict[int, Set[int]]:
    return {int(u): {int(v) for v in G.neighbors(u)} for u in G.nodes()}


def gf2_nullity_nodes(adj: Mapping[int, Iterable[int]], nodes: Sequence[int]) -> int:
    nodes = [int(x) for x in nodes]
    idx = {u: i for i, u in enumerate(nodes)}
    rows: List[int] = []
    for u in nodes:
        bits = 0
        for v in adj.get(u, ()):
            j = idx.get(int(v))
            if j is not None:
                bits |= 1 << j
        rows.append(bits)
    return len(nodes) - gf2_rank_bits(rows)


def r_hop(adj: Mapping[int, Iterable[int]], H: Set[int], radius: int) -> Set[int]:
    seen = set(int(x) for x in H)
    front = set(seen)
    for _ in range(int(radius)):
        nxt: Set[int] = set()
        for u in front:
            nxt.update(int(v) for v in adj.get(u, ()))
        nxt -= seen
        seen |= nxt
        front = nxt
        if not front:
            break
    return seen


def _sample_subset(rng: random.Random, population: Sequence[int], k: int):
    return rng.sample(list(population), k) if k else []


def global_knotstate(
    G: nx.Graph,
    max_order: int = 6,
    exact_cap: int = 2_000_000,
    samples: int = 100_000,
    seed: int = 0,
) -> Dict[Tuple[int, int], float]:
    """Return normalized global (s,nu) KnotState profile.

    Exact enumeration is used whenever C(n,s) <= exact_cap, otherwise uniform
    Monte Carlo sampling of s-subsets is used.
    """
    H = nx.convert_node_labels_to_integers(nx.Graph(G), ordering="sorted")
    adj = adjacency_sets(H)
    nodes = list(range(H.number_of_nodes()))
    rng = random.Random(seed)
    out: Dict[Tuple[int, int], float] = {}

    for s in range(2, max_order + 1):
        if len(nodes) < s:
            for nu in allowed_nullities(s):
                out[(s, nu)] = 0.0
            continue
        total = math.comb(len(nodes), s)
        counts = {nu: 0 for nu in allowed_nullities(s)}
        if total <= exact_cap:
            iterator = itertools.combinations(nodes, s)
            denom = total
        else:
            iterator = (_sample_subset(rng, nodes, s) for _ in range(samples))
            denom = samples
        for S in iterator:
            nu = gf2_nullity_nodes(adj, S)
            counts[nu] = counts.get(nu, 0) + 1
        for nu in allowed_nullities(s):
            out[(s, nu)] = counts.get(nu, 0) / max(denom, 1)
    return out


def global_knotstate_counts(
    G: nx.Graph,
    max_order: int = 6,
    exact_cap: int = 2_000_000,
    samples: int = 100_000,
    seed: int = 0,
) -> Tuple[Dict[Tuple[int, int], int], Dict[int, int], Dict[int, bool]]:
    """Return counts, denominators, and exactness flags for each order."""
    H = nx.convert_node_labels_to_integers(nx.Graph(G), ordering="sorted")
    adj = adjacency_sets(H)
    nodes = list(range(H.number_of_nodes()))
    rng = random.Random(seed)
    counts_out: Dict[Tuple[int, int], int] = {}
    denoms: Dict[int, int] = {}
    exact: Dict[int, bool] = {}
    for s in range(2, max_order + 1):
        total = math.comb(len(nodes), s) if len(nodes) >= s else 0
        c = {nu: 0 for nu in allowed_nullities(s)}
        if total == 0:
            denoms[s] = 0
            exact[s] = True
        elif total <= exact_cap:
            denoms[s] = total
            exact[s] = True
            for S in itertools.combinations(nodes, s):
                nu = gf2_nullity_nodes(adj, S)
                c[nu] = c.get(nu, 0) + 1
        else:
            denoms[s] = samples
            exact[s] = False
            for _ in range(samples):
                S = _sample_subset(rng, nodes, s)
                nu = gf2_nullity_nodes(adj, S)
                c[nu] = c.get(nu, 0) + 1
        for nu in allowed_nullities(s):
            counts_out[(s, nu)] = c.get(nu, 0)
    return counts_out, denoms, exact


def target_knot_keys(max_order: int = 6):
    return [
        (s, a, nu)
        for s in range(2, max_order + 1)
        for a in range(s + 1)
        for nu in allowed_nullities(s)
    ]


KNOT_KEYS_75 = target_knot_keys(6)
assert len(KNOT_KEYS_75) == 75


def target_conditioned_knotstate(
    G_or_adj,
    H_nodes: Sequence[int],
    radius: int = 1,
    max_order: int = 6,
    exact_cap: int = 2000,
    samples: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Return the target-conditioned KnotState vector.

    Coordinates are ordered by (s,a,nu), with s increasing, then a, then nu.
    For max_order=6 the dimension is 75.
    """
    adj = adjacency_sets(G_or_adj) if isinstance(G_or_adj, nx.Graph) else G_or_adj
    rng = random.Random(seed)
    H = {int(x) for x in H_nodes}
    outside = r_hop(adj, H, radius) - H
    Hin, Hout = list(H), list(outside)
    keys = target_knot_keys(max_order)
    feat = {k: 0.0 for k in keys}

    for s in range(2, max_order + 1):
        for a in range(s + 1):
            b = s - a
            if a > len(Hin) or b > len(Hout):
                continue
            total = math.comb(len(Hin), a) * math.comb(len(Hout), b)
            if total == 0:
                continue
            counts = {nu: 0 for nu in allowed_nullities(s)}
            n_eval = 0
            if total <= exact_cap:
                for I in itertools.combinations(Hin, a):
                    for O in itertools.combinations(Hout, b):
                        nu = gf2_nullity_nodes(adj, tuple(I) + tuple(O))
                        counts[nu] += 1
                        n_eval += 1
            else:
                for _ in range(samples):
                    I = _sample_subset(rng, Hin, a)
                    O = _sample_subset(rng, Hout, b)
                    nu = gf2_nullity_nodes(adj, I + O)
                    counts[nu] += 1
                    n_eval += 1
            if n_eval:
                for nu, c in counts.items():
                    feat[(s, a, nu)] = c / n_eval
    return np.asarray([feat[k] for k in keys], dtype=np.float32)


def select_target_columns(max_s: int, min_s: int = 2, keys=None) -> List[int]:
    keys = KNOT_KEYS_75 if keys is None else keys
    return [i for i, (s, a, nu) in enumerate(keys) if min_s <= s <= max_s]


def self_test() -> None:
    G = nx.complete_bipartite_graph(2, 3)
    adj = adjacency_sets(G)
    assert gf2_nullity_nodes(adj, list(G.nodes())) == 3
    x = target_conditioned_knotstate(G, [0, 1, 2], radius=1, samples=20, seed=1)
    assert x.shape == (75,)
    # K2: edge has nullity 0; nonedge has nullity 2.
    P = nx.path_graph(3)
    p = global_knotstate(P, max_order=2, exact_cap=100)
    assert abs(p[(2, 0)] - 2 / 3) < 1e-12
    assert abs(p[(2, 2)] - 1 / 3) < 1e-12


if __name__ == "__main__":
    self_test()
    print("KnotState core self-test passed")
