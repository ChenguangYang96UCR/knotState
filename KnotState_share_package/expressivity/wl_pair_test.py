#!/usr/bin/env python3
"""Exact k-WL pair equivalence tester for small/medium graphs.

This implements standard k-dimensional WL on ordered k-tuples with a shared
color dictionary for the two graphs. It is intended for spot checks and witness
audits; k=3 scales as O(n^4) per refinement round and is not optimized for a
full large BREC sweep.
"""
from __future__ import annotations
import argparse
import itertools
from collections import Counter
from pathlib import Path

import networkx as nx


def atomic_type(G: nx.Graph, tup):
    k = len(tup)
    out = []
    for i in range(k):
        for j in range(k):
            if tup[i] == tup[j]:
                out.append(0)
            elif G.has_edge(tup[i], tup[j]):
                out.append(1)
            else:
                out.append(2)
    return tuple(out)


def k_wl_equivalent(G: nx.Graph, H: nx.Graph, k: int, max_iter: int | None = None):
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    H = nx.convert_node_labels_to_integers(H, ordering="sorted")
    if G.number_of_nodes() != H.number_of_nodes():
        return False
    n = G.number_of_nodes()
    tuples = list(itertools.product(range(n), repeat=k))
    ag = [atomic_type(G, t) for t in tuples]
    ah = [atomic_type(H, t) for t in tuples]
    cmap = {s: i for i, s in enumerate(sorted(set(ag + ah)))}
    cg = [cmap[s] for s in ag]
    ch = [cmap[s] for s in ah]
    if Counter(cg) != Counter(ch):
        return False
    if max_iter is None:
        max_iter = max(2, n ** k + 2)
    index = {t: i for i, t in enumerate(tuples)}
    for _ in range(max_iter):
        sg, sh = [], []
        for idx, t in enumerate(tuples):
            mg, mh = [], []
            for coord in range(k):
                vals_g, vals_h = [], []
                for w in range(n):
                    tt = list(t); tt[coord] = w; j = index[tuple(tt)]
                    vals_g.append(cg[j]); vals_h.append(ch[j])
                mg.append(tuple(sorted(vals_g)))
                mh.append(tuple(sorted(vals_h)))
            sg.append((cg[idx], tuple(mg)))
            sh.append((ch[idx], tuple(mh)))
        cmap = {s: i for i, s in enumerate(sorted(set(sg + sh)))}
        ng = [cmap[s] for s in sg]
        nh = [cmap[s] for s in sh]
        if Counter(ng) != Counter(nh):
            return False
        if ng == cg and nh == ch:
            return True
        cg, ch = ng, nh
    return True


def load_one(path: Path):
    b = path.read_bytes().strip().splitlines()[0]
    return nx.Graph(nx.from_graph6_bytes(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=Path, required=True)
    ap.add_argument("--h", type=Path, required=True)
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()
    G, H = load_one(args.g), load_one(args.h)
    eq = k_wl_equivalent(G, H, args.k)
    print(f"{args.k}-WL equivalent: {eq}")
    print(f"{args.k}-WL separated: {not eq}")


if __name__ == "__main__":
    main()
