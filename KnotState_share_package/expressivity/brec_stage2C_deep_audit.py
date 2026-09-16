#!/usr/bin/env python3
"""
Stage 2C: Deep audit of BREC KnotState wins.

What it does on the Stage-2B deep-win pairs:
  1) exact full induced k-node graphlet profile at the first separating k (k=5 or 6),
  2) exact KnotState nullity profile derived from those graphlets,
  3) ordinary 3-WL pair test,
  4) exact permutation/relabeling invariance audit of the KnotState profile.

This is a descriptor/expressivity audit, not official BREC neural RPC accuracy.
CPU only; no torch/PyG required.
"""

from __future__ import annotations
import argparse
import itertools
import math
import os
import time
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

NUM_RELABEL_BREC = 32
PAIR_STRIDE = 2 * NUM_RELABEL_BREC
FALLBACK_DEEP = [111, 125, 127, 128, 129, 130, 132, 133, 134, 135, 136, 139, 141, 142, 144]


def as_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def g6_to_graph(x):
    if isinstance(x, str):
        b = x.encode()
    elif isinstance(x, np.bytes_):
        b = bytes(x)
    elif isinstance(x, (bytes, bytearray)):
        b = bytes(x)
    else:
        b = str(x).encode()
    return nx.from_graph6_bytes(b.strip())


def locate_brec_v3(zip_path: str | None, outdir: Path) -> Path:
    candidates = [
        Path('/content/brec_knotstate/brec_v3.npy'),
        Path('/content/brec_stage2B/brec_v3.npy'),
        outdir / 'brec_v3.npy',
    ]
    for p in candidates:
        if p.exists():
            return p
    if zip_path:
        zpath = Path(zip_path)
        if not zpath.exists():
            raise FileNotFoundError(zpath)
        with zipfile.ZipFile(zpath, 'r') as z:
            names = [n for n in z.namelist() if Path(n).name == 'brec_v3.npy']
            if not names:
                raise FileNotFoundError('brec_v3.npy not found inside zip')
            target = outdir / 'brec_v3.npy'
            target.write_bytes(z.read(names[0]))
            return target
    raise FileNotFoundError('Could not find brec_v3.npy. Pass --zip /content/BREC_data_all.zip')


def locate_stage2b_csv(path: str | None) -> Path | None:
    if path:
        p = Path(path)
        return p if p.exists() else None
    candidates = [
        Path('/content/brec_stage2B/pair_results.csv'),
        Path('/content/brec_stage2B/stage2B_pairs.csv'),
        Path('/content/brec_stage2B/results.csv'),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_deep_pairs(stage2b_csv: Path | None):
    if stage2b_csv is None:
        rows = []
        for pid in FALLBACK_DEEP:
            rows.append({'pair_id': pid, 'knot_first_k': 6 if pid == 111 else 5})
        print('WARNING: Stage-2B CSV not found; using the 15 deep-win IDs from the prior run.')
        return pd.DataFrame(rows)

    df = pd.read_csv(stage2b_csv)
    if 'deep_win' in df.columns:
        sub = df[df['deep_win'].map(as_bool)].copy()
    elif 'deep_wins' in df.columns:
        sub = df[df['deep_wins'].map(as_bool)].copy()
    else:
        ids = set(FALLBACK_DEEP)
        sub = df[df['pair_id'].astype(int).isin(ids)].copy()
    if sub.empty:
        raise RuntimeError('No deep-win rows found in Stage-2B CSV')
    if 'knot_first_k' not in sub.columns:
        sub['knot_first_k'] = sub['pair_id'].map(lambda x: 6 if int(x) == 111 else 5)
    sub['pair_id'] = sub['pair_id'].astype(int)
    sub['knot_first_k'] = sub['knot_first_k'].astype(float).round().astype(int)
    sub = sub.sort_values('pair_id').reset_index(drop=True)
    return sub


def gf2_rank(A: np.ndarray) -> int:
    A = (A.astype(np.uint8) & 1).copy()
    m, n = A.shape
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if A[i, c]:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        for i in range(m):
            if i != r and A[i, c]:
                A[i] ^= A[r]
        r += 1
        if r == m:
            break
    return r


def bit_positions(k: int):
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def code_from_subset(A: np.ndarray, subset, pos) -> int:
    code = 0
    for b, (i, j) in enumerate(pos):
        if A[subset[i], subset[j]]:
            code |= (1 << b)
    return code


def build_graphlet_catalog(k: int):
    """Map every labeled k-node graph bit-code to one unlabeled atlas class."""
    atlas = [g.copy() for g in nx.graph_atlas_g() if g.number_of_nodes() == k]
    expected = {5: 34, 6: 156}.get(k)
    if expected is not None and len(atlas) != expected:
        raise RuntimeError(f'Expected {expected} unlabeled graphs on k={k}, got {len(atlas)}')
    pos = bit_positions(k)
    labeled_to_class = np.full(1 << len(pos), -1, dtype=np.int16)
    class_nullity = np.zeros(len(atlas), dtype=np.int8)

    for cid, g in enumerate(atlas):
        g = nx.convert_node_labels_to_integers(g, ordering='sorted')
        A = nx.to_numpy_array(g, nodelist=list(range(k)), dtype=np.uint8)
        class_nullity[cid] = k - gf2_rank(A)
        for perm in itertools.permutations(range(k)):
            code = 0
            for b, (i, j) in enumerate(pos):
                if A[perm[i], perm[j]]:
                    code |= (1 << b)
            prev = labeled_to_class[code]
            if prev not in (-1, cid):
                raise RuntimeError('Graphlet catalog collision')
            labeled_to_class[code] = cid
    if np.any(labeled_to_class < 0):
        miss = int(np.sum(labeled_to_class < 0))
        raise RuntimeError(f'Graphlet catalog incomplete: {miss} labeled graphs unassigned for k={k}')
    return atlas, pos, labeled_to_class, class_nullity


def graphlet_profile(A: np.ndarray, k: int, pos, labeled_to_class, nclass: int):
    counts = np.zeros(nclass, dtype=np.int64)
    n = A.shape[0]
    for subset in itertools.combinations(range(n), k):
        code = code_from_subset(A, subset, pos)
        counts[int(labeled_to_class[code])] += 1
    return counts


def nullity_profile_from_graphlets(counts: np.ndarray, class_nullity: np.ndarray, k: int):
    out = np.zeros(k + 1, dtype=np.int64)
    for cid, c in enumerate(counts):
        out[int(class_nullity[cid])] += int(c)
    return out


def exact_nullity_profile(A: np.ndarray, k: int, pos, labeled_to_class, class_nullity):
    out = np.zeros(k + 1, dtype=np.int64)
    n = A.shape[0]
    for subset in itertools.combinations(range(n), k):
        code = code_from_subset(A, subset, pos)
        cid = int(labeled_to_class[code])
        out[int(class_nullity[cid])] += 1
    return out


def relabel_invariance_audit(A: np.ndarray, k: int, base_profile: np.ndarray, repeats: int,
                             pos, labeled_to_class, class_nullity, seed: int):
    if repeats <= 0:
        return True, 0
    rng = np.random.default_rng(seed)
    failures = 0
    n = A.shape[0]
    for _ in range(repeats):
        p = rng.permutation(n)
        Ap = A[np.ix_(p, p)]
        prof = exact_nullity_profile(Ap, k, pos, labeled_to_class, class_nullity)
        if not np.array_equal(prof, base_profile):
            failures += 1
    return failures == 0, failures


def wl3_pair_separated(A: np.ndarray, B: np.ndarray, max_iter: int = 20):
    """Ordinary 3-WL, refined jointly on the pair so color IDs are comparable."""
    if A.shape[0] != B.shape[0]:
        return True, 0
    n = A.shape[0]
    I, J, K = np.indices((n, n, n))

    def init_colors(M):
        eq = (I == J).astype(np.int32) + 2 * (I == K).astype(np.int32) + 4 * (J == K).astype(np.int32)
        ed = 8 * M[I, J].astype(np.int32) + 16 * M[I, K].astype(np.int32) + 32 * M[J, K].astype(np.int32)
        return eq + ed

    CA = init_colors(A)
    CB = init_colors(B)

    def hist_diff(X, Y):
        m = int(max(X.max(initial=0), Y.max(initial=0))) + 1
        return not np.array_equal(np.bincount(X.ravel(), minlength=m), np.bincount(Y.ravel(), minlength=m))

    if hist_diff(CA, CB):
        return True, 0

    prev_num = len(np.unique(np.concatenate([CA.ravel(), CB.ravel()])))
    for it in range(1, max_iter + 1):
        # Create common IDs for coordinate-wise replacement multisets.
        line_keys = []
        refs = []
        for graph_id, C in enumerate((CA, CB)):
            # coord 0: replace first coordinate => C[:, j, k]
            for j in range(n):
                for k in range(n):
                    line_keys.append((0, tuple(sorted(C[:, j, k].tolist())))); refs.append((graph_id, 0, j, k))
            # coord 1
            for i in range(n):
                for k in range(n):
                    line_keys.append((1, tuple(sorted(C[i, :, k].tolist())))); refs.append((graph_id, 1, i, k))
            # coord 2
            for i in range(n):
                for j in range(n):
                    line_keys.append((2, tuple(sorted(C[i, j, :].tolist())))); refs.append((graph_id, 2, i, j))
        key_to_id = {key: idx for idx, key in enumerate(sorted(set(line_keys)))}
        lines = [[np.zeros((n, n), dtype=np.int32) for _ in range(3)] for _ in range(2)]
        for key, ref in zip(line_keys, refs):
            gid, coord, a, b = ref
            lines[gid][coord][a, b] = key_to_id[key]

        sigs = []
        owners = []
        for gid, C in enumerate((CA, CB)):
            L0, L1, L2 = lines[gid]
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        sigs.append((int(C[i, j, k]), int(L0[j, k]), int(L1[i, k]), int(L2[i, j])))
                        owners.append((gid, i, j, k))
        sig_to_id = {sig: idx for idx, sig in enumerate(sorted(set(sigs)))}
        NA = np.empty_like(CA)
        NB = np.empty_like(CB)
        for sig, own in zip(sigs, owners):
            gid, i, j, k = own
            (NA if gid == 0 else NB)[i, j, k] = sig_to_id[sig]

        if hist_diff(NA, NB):
            return True, it
        num = len(sig_to_id)
        CA, CB = NA, NB
        if num == prev_num:
            return False, it
        prev_num = num
    return False, max_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default='/content/BREC_data_all.zip')
    ap.add_argument('--stage2b-csv', default=None)
    ap.add_argument('--outdir', default='/content/brec_stage2C')
    ap.add_argument('--relabels', type=int, default=3,
                    help='Exact random relabel audits per graph. Use 3 for smoke, 20 for final audit.')
    ap.add_argument('--wl3-max-iter', type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    brec_path = locate_brec_v3(args.zip, outdir)
    arr = np.load(brec_path, allow_pickle=True)
    stage2b = locate_stage2b_csv(args.stage2b_csv)
    deep = load_deep_pairs(stage2b)
    print(f'Using BREC file: {brec_path}')
    print(f'Loaded {len(deep)} Stage-2B deep-win pairs: {deep.pair_id.tolist()}')

    needed_k = sorted(set(int(k) for k in deep.knot_first_k.tolist()))
    catalogs = {}
    for k in needed_k:
        print(f'Building exact unlabeled graphlet catalog for k={k} ...', flush=True)
        catalogs[k] = build_graphlet_catalog(k)
        atlas, pos, mapping, class_nullity = catalogs[k]
        print(f'  k={k}: {len(atlas)} unlabeled graphlets -> {len(np.unique(class_nullity))} nullity bins; labeled map={len(mapping)}')

    rows = []
    long_rows = []
    t0 = time.time()
    for rr, row in deep.iterrows():
        pid = int(row['pair_id'])
        k = int(row['knot_first_k'])
        base = pid * PAIR_STRIDE
        if base + 1 >= len(arr):
            raise IndexError(f'Pair {pid} out of range for brec_v3.npy length {len(arr)}')
        G = g6_to_graph(arr[base])
        H = g6_to_graph(arr[base + 1])
        G = nx.convert_node_labels_to_integers(G, ordering='sorted')
        H = nx.convert_node_labels_to_integers(H, ordering='sorted')
        AG = nx.to_numpy_array(G, dtype=np.uint8)
        AH = nx.to_numpy_array(H, dtype=np.uint8)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        if H.number_of_nodes() != n:
            raise RuntimeError(f'Pair {pid}: graph orders differ')

        atlas, pos, mapping, class_nullity = catalogs[k]
        gprof = graphlet_profile(AG, k, pos, mapping, len(atlas))
        hprof = graphlet_profile(AH, k, pos, mapping, len(atlas))
        graphlet_sep = not np.array_equal(gprof, hprof)
        knotG = nullity_profile_from_graphlets(gprof, class_nullity, k)
        knotH = nullity_profile_from_graphlets(hprof, class_nullity, k)
        knot_sep = not np.array_equal(knotG, knotH)
        if knot_sep and not graphlet_sep:
            raise RuntimeError(f'Pair {pid}: impossible audit failure: KnotState separates but full same-k graphlets do not')

        total = math.comb(n, k)
        gtv = 0.5 * np.abs(gprof / total - hprof / total).sum()
        ktv = 0.5 * np.abs(knotG / total - knotH / total).sum()
        ndiff = int(np.count_nonzero(gprof != hprof))

        wl3_sep, wl3_iter = wl3_pair_separated(AG.astype(bool), AH.astype(bool), max_iter=args.wl3_max_iter)

        okG, failG = relabel_invariance_audit(AG, k, knotG, args.relabels, pos, mapping, class_nullity, seed=10000 + pid)
        okH, failH = relabel_invariance_audit(AH, k, knotH, args.relabels, pos, mapping, class_nullity, seed=20000 + pid)
        relabel_ok = okG and okH

        dims_full = len(atlas)
        dims_knot = int(len(np.unique(class_nullity)))
        compression = dims_full / dims_knot
        deepest = bool(knot_sep and (not wl3_sep))

        rows.append({
            'pair_id': pid, 'subgroup': row.get('subgroup', 'StronglyRegular'),
            'n_g': n, 'm_g': m, 'k': k,
            'full_graphlet_dim': dims_full, 'knot_nullity_bins': dims_knot,
            'compression_ratio': compression,
            'full_graphlet_separated': graphlet_sep,
            'full_graphlet_diff_classes': ndiff,
            'full_graphlet_tv': float(gtv),
            'knot_separated': knot_sep,
            'knot_tv': float(ktv),
            'three_wl_separated': wl3_sep,
            'three_wl_first_iter': wl3_iter,
            'knot_beyond_3wl': deepest,
            'relabel_audit_ok': relabel_ok,
            'relabel_failures': failG + failH,
        })

        diff_ids = np.flatnonzero(gprof != hprof)
        for cid in diff_ids:
            long_rows.append({
                'pair_id': pid, 'k': k, 'graphlet_class_id': int(cid),
                'nullity': int(class_nullity[cid]),
                'count_G': int(gprof[cid]), 'count_H': int(hprof[cid]),
                'delta': int(gprof[cid] - hprof[cid]),
            })

        print(f"{rr+1:2d}/{len(deep)} pair={pid} k={k} | 3WL_sep={int(wl3_sep)} | graphlet_sep={int(graphlet_sep)} | knot_sep={int(knot_sep)} | beyond3WL={int(deepest)} | relabel_ok={int(relabel_ok)} | elapsed={time.time()-t0:.1f}s", flush=True)

    res = pd.DataFrame(rows)
    long_df = pd.DataFrame(long_rows)
    res.to_csv(outdir / 'deep_pair_audit.csv', index=False)
    long_df.to_csv(outdir / 'graphlet_differences_long.csv', index=False)

    summary = pd.DataFrame([{
        'deep_pairs': len(res),
        'full_same_k_graphlet_sep': int(res.full_graphlet_separated.sum()),
        'knot_sep': int(res.knot_separated.sum()),
        'three_wl_sep': int(res.three_wl_separated.sum()),
        'three_wl_fail': int((~res.three_wl_separated).sum()),
        'knot_beyond_3wl': int(res.knot_beyond_3wl.sum()),
        'relabel_audit_fail_pairs': int((~res.relabel_audit_ok).sum()),
        'k5_pairs': int((res.k == 5).sum()),
        'k6_pairs': int((res.k == 6).sum()),
        'mean_graphlet_tv': float(res.full_graphlet_tv.mean()),
        'mean_knot_tv': float(res.knot_tv.mean()),
    }])
    summary.to_csv(outdir / 'summary.csv', index=False)

    print('\n=== STAGE 2C SUMMARY ===')
    print(summary.to_string(index=False))
    print('\n=== PAIR AUDIT ===')
    cols = ['pair_id','n_g','m_g','k','full_graphlet_dim','knot_nullity_bins','compression_ratio',
            'full_graphlet_diff_classes','full_graphlet_tv','knot_tv','three_wl_separated','three_wl_first_iter','knot_beyond_3wl','relabel_audit_ok']
    print(res[cols].to_string(index=False))

    print('\nKey interpretation:')
    print('  k=5: full profile dimension=34, KnotState nullity bins=3.')
    print('  k=6: full profile dimension=156, KnotState nullity bins=4.')
    print('  Exact same-k KnotState separation must always be explained by full same-k graphlets; this script audits that.')
    print(f'Files saved under: {outdir}')


if __name__ == '__main__':
    main()
