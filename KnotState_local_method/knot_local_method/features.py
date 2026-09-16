"""Exact and sampled GF(2) nullity profiles with explicit channel metadata."""
import itertools
import math
import random
import time
import numpy as np


def nullity(adj, nodes):
    """Compute adjacency nullity over GF(2) using integer row elimination."""
    index = {v: i for i, v in enumerate(nodes)}
    basis = {}
    for u in nodes:
        row = sum(1 << index[v] for v in adj[u] if v in index)
        while row:
            pivot = row.bit_length() - 1
            if pivot in basis:
                row ^= basis[pivot]
            else:
                basis[pivot] = row
                break
    return len(nodes) - len(basis)


def neighborhood(adj, roots, radius):
    seen = set(roots)
    frontier = set(roots)
    for _ in range(radius):
        frontier = set().union(*(adj[v] for v in frontier)) - seen if frontier else set()
        seen.update(frontier)
    return seen


def extract(G, target, radius=1, max_order=4, budget=4000, seed=0,
            max_roots=0, mode='rooted'):
    """Use at most budget nullity evaluations per target across feasible blocks.

    Small blocks are exact. Remaining budget is allocated evenly across blocks.
    Sampling is uniform within each block, with replacement across draws.
    """
    if G.is_directed() or G.is_multigraph() or any(u == v for u, v in G.edges()):
        raise ValueError('Expected a simple undirected graph without self-loops')
    if radius < 0 or max_order < 2 or budget < 1 or max_roots < 0:
        raise ValueError('Invalid extraction parameters')
    if mode not in ('rooted', 'target'):
        raise ValueError('mode must be rooted or target')
    H = set(target)
    if not H or not H <= set(G):
        raise ValueError('Target must be nonempty and present in graph')
    start = time.perf_counter()
    adj = {v: set(G[v]) for v in G}
    rng = random.Random(seed)
    roots = sorted(H)
    if mode == 'rooted' and max_roots and len(roots) > max_roots:
        roots = sorted(rng.sample(roots, max_roots))
    if mode == 'target':
        roots = [None]
    keys = [(s, a, nu) for s in range(2, max_order + 1)
            for a in range(1 if mode == 'rooted' else 0, s + 1)
            for nu in range(s % 2, s + 1, 2)]
    lookup = {key: j for j, key in enumerate(keys)}
    values = np.zeros((len(roots), len(keys)), dtype=np.float64)
    mask = np.zeros_like(values, dtype=bool)
    blocks = []
    for i, root in enumerate(roots):
        region = neighborhood(adj, H if root is None else [root], radius)
        inside = sorted((region & H) - ({root} if root is not None else set()))
        outside = sorted(region - H)
        for s in range(2, max_order + 1):
            for a in range(1 if root is not None else 0, s + 1):
                ni, no = a - int(root is not None), s - a
                if ni > len(inside) or no > len(outside):
                    continue
                total = math.comb(len(inside), ni) * math.comb(len(outside), no)
                blocks.append((i, root, s, a, inside, outside, ni, no, total))
    if len(blocks) > budget:
        raise ValueError(f'budget {budget} < {len(blocks)} feasible blocks; increase budget or reduce max_roots')
    # Allocate an equal quota, redistributing unused capacity from exact blocks.
    quotas = [0] * len(blocks)
    remaining = budget
    active = list(range(len(blocks)))
    while active and remaining:
        share = max(1, remaining // len(active))
        for j in active:
            take = min(share, blocks[j][-1] - quotas[j], remaining)
            quotas[j] += take
            remaining -= take
        active = [j for j in active if quotas[j] < blocks[j][-1]]
    records = []
    for block, count in zip(blocks, quotas):
        i, root, s, a, inside, outside, ni, no, total = block
        exact = count == total
        if exact:
            pairs = itertools.product(itertools.combinations(inside, ni),
                                      itertools.combinations(outside, no))
        else:
            pairs = ((rng.sample(inside, ni), rng.sample(outside, no)) for _ in range(count))
        for left, right in pairs:
            nodes = list(left) + list(right) + ([] if root is None else [root])
            values[i, lookup[s, a, nullity(adj, nodes)]] += 1 / count
        for nu in range(s % 2, s + 1, 2):
            mask[i, lookup[s, a, nu]] = True
        records.append((i, s, a, total, count, exact))
    return dict(values=values, mask=mask, keys=np.asarray(keys),
                roots=np.asarray([-1 if v is None else v for v in roots]),
                blocks=records, evaluations=sum(quotas), seconds=time.perf_counter()-start)


def pool(result, kind='mean'):
    """Pool feasible rows only; append coverage to distinguish absent blocks."""
    if kind not in ('mean', 'stats'):
        raise ValueError(kind)
    x, mask = result['values'], result['mask']
    count = mask.sum(axis=0)
    mean = x.sum(axis=0) / np.maximum(count, 1)
    variance = (((x - mean) ** 2) * mask).sum(axis=0) / np.maximum(count, 1)
    parts = [('mean', mean)]
    if kind == 'stats':
        parts.append(('std', np.sqrt(variance)))
    parts.append(('coverage', mask.mean(axis=0)))
    metadata = [[name, int(s), int(a), int(nu)] for name, _ in parts for s, a, nu in result['keys']]
    return np.concatenate([v for _, v in parts]), metadata
