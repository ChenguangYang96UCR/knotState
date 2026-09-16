#!/usr/bin/env python3
"""
Stage 2D: 4-WL-level audit + mechanism decomposition for the 15 BREC deep pairs.

Core tasks
----------
1) Run exact 3-FWL (Folklore WL on ordered triples). In graph-distinguishing
   power, 3-FWL is equivalent to ordinary 4-WL.
2) Recompute the exact same-k (k=5 or 6) induced graphlet profiles.
3) Decompose every graphlet-count difference by GF(2) adjacency nullity,
   showing exactly how the low-dimensional KnotState bins arise.
4) Save graphlet metadata (edges, triangles, components, exact treewidth,
   rank/nullity, graph6) and identify the dominant graphlet drivers.

This is an expressivity/descriptor audit, not official BREC neural RPC accuracy.
CPU only; no torch/PyG required.
"""

from __future__ import annotations
import argparse
import itertools
import math
import time
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

NUM_RELABEL_BREC = 32
PAIR_STRIDE = 2 * NUM_RELABEL_BREC
FALLBACK_DEEP = [111, 125, 127, 128, 129, 130, 132, 133, 134, 135, 136, 139, 141, 142, 144]


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
        Path('/content/brec_stage2C/brec_v3.npy'),
        Path('/content/brec_stage2B/brec_v3.npy'),
        Path('/content/brec_knotstate/brec_v3.npy'),
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


def locate_stage2c_csv(path: str | None) -> Path | None:
    if path:
        p = Path(path)
        return p if p.exists() else None
    for p in [
        Path('/content/brec_stage2C/deep_pair_audit.csv'),
        Path('/content/brec_stage2C/results.csv'),
    ]:
        if p.exists():
            return p
    return None


def load_deep_pairs(stage2c_csv: Path | None):
    if stage2c_csv is not None:
        df = pd.read_csv(stage2c_csv)
        if 'knot_beyond_3wl' in df.columns:
            mask = df['knot_beyond_3wl'].astype(str).str.lower().isin(['true','1','yes','y','t'])
            sub = df[mask].copy()
        else:
            sub = df[df['pair_id'].astype(int).isin(FALLBACK_DEEP)].copy()
        if not sub.empty:
            sub['pair_id'] = sub['pair_id'].astype(int)
            if 'k' not in sub.columns:
                sub['k'] = sub['pair_id'].map(lambda p: 6 if int(p) == 111 else 5)
            sub['k'] = sub['k'].astype(int)
            return sub.sort_values('pair_id').reset_index(drop=True)
    rows = [{'pair_id': p, 'k': 6 if p == 111 else 5, 'subgroup': 'StronglyRegular'} for p in FALLBACK_DEEP]
    print('WARNING: Stage-2C CSV not found; using the 15 known deep-pair IDs from the prior run.')
    return pd.DataFrame(rows)


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


def exact_treewidth_small(g: nx.Graph) -> int:
    """Exact treewidth by elimination-order brute force; safe for k<=6."""
    nodes = list(g.nodes())
    if len(nodes) <= 1:
        return 0
    best = len(nodes) - 1
    base = {v: set(g.neighbors(v)) for v in nodes}
    for order in itertools.permutations(nodes):
        adj = {v: set(nbrs) for v, nbrs in base.items()}
        width = 0
        pruned = False
        for v in order:
            nbr = list(adj.get(v, ()))
            width = max(width, len(nbr))
            if width >= best:
                pruned = True
                break
            for a_idx in range(len(nbr)):
                a = nbr[a_idx]
                for b_idx in range(a_idx + 1, len(nbr)):
                    b = nbr[b_idx]
                    adj[a].add(b)
                    adj[b].add(a)
            for u in nbr:
                adj[u].discard(v)
            adj.pop(v, None)
        if not pruned:
            best = min(best, width)
        if best == 0:
            break
    return int(best)


def build_graphlet_catalog(k: int):
    atlas = [g.copy() for g in nx.graph_atlas_g() if g.number_of_nodes() == k]
    expected = {5: 34, 6: 156}.get(k)
    if expected is not None and len(atlas) != expected:
        raise RuntimeError(f'Expected {expected} unlabeled graphs on k={k}, got {len(atlas)}')

    pos = bit_positions(k)
    labeled_to_class = np.full(1 << len(pos), -1, dtype=np.int16)
    metadata = []

    for cid, g0 in enumerate(atlas):
        g = nx.convert_node_labels_to_integers(g0, ordering='sorted')
        A = nx.to_numpy_array(g, nodelist=list(range(k)), dtype=np.uint8)
        rank = gf2_rank(A)
        nullity = k - rank
        tri = int(sum(nx.triangles(g).values()) // 3)
        comps = int(nx.number_connected_components(g))
        tw = exact_treewidth_small(g)
        degseq = ','.join(map(str, sorted((d for _, d in g.degree()), reverse=True)))
        g6 = nx.to_graph6_bytes(g, header=False).strip().decode()
        metadata.append({
            'k': k,
            'graphlet_class_id': cid,
            'graph6': g6,
            'edges': int(g.number_of_edges()),
            'triangles': tri,
            'components': comps,
            'connected': bool(nx.is_connected(g)),
            'treewidth_exact': tw,
            'gf2_rank': int(rank),
            'gf2_nullity': int(nullity),
            'degree_sequence': degseq,
        })

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
        raise RuntimeError(f'Graphlet catalog incomplete for k={k}: {(labeled_to_class < 0).sum()} codes missing')

    meta = pd.DataFrame(metadata)
    class_nullity = meta.gf2_nullity.to_numpy(dtype=np.int8)
    return atlas, pos, labeled_to_class, class_nullity, meta


def graphlet_profile(A: np.ndarray, k: int, pos, labeled_to_class, nclass: int):
    counts = np.zeros(nclass, dtype=np.int64)
    for subset in itertools.combinations(range(A.shape[0]), k):
        code = code_from_subset(A, subset, pos)
        counts[int(labeled_to_class[code])] += 1
    return counts


def nullity_profile_from_graphlets(counts: np.ndarray, class_nullity: np.ndarray, k: int):
    out = np.zeros(k + 1, dtype=np.int64)
    for cid, c in enumerate(counts):
        out[int(class_nullity[cid])] += int(c)
    return out


def atomic_triple_colors(A: np.ndarray):
    n = A.shape[0]
    I, J, K = np.indices((n, n, n))
    eq = (I == J).astype(np.int32) + 2 * (I == K).astype(np.int32) + 4 * (J == K).astype(np.int32)
    ed = 8 * A[I, J].astype(np.int32) + 16 * A[I, K].astype(np.int32) + 32 * A[J, K].astype(np.int32)
    return eq + ed


def color_hist_diff(CA: np.ndarray, CB: np.ndarray) -> bool:
    m = int(max(int(CA.max()), int(CB.max()))) + 1
    return not np.array_equal(np.bincount(CA.ravel(), minlength=m), np.bincount(CB.ravel(), minlength=m))


def fwl3_pair_separated(A: np.ndarray, B: np.ndarray, max_iter: int = 10):
    """
    Exact 3-FWL pair test, refined jointly so color IDs are comparable.

    For tuple (i,j,k), the FWL neighborhood is the multiset over w of the
    *joint* replacement-color vector
      (c(w,j,k), c(i,w,k), c(i,j,w)).
    This is different from ordinary 3-WL, which keeps three separate
    coordinate-wise multisets.
    """
    if A.shape[0] != B.shape[0]:
        return True, 0, 0
    n = A.shape[0]
    CA = atomic_triple_colors(A)
    CB = atomic_triple_colors(B)
    if color_hist_diff(CA, CB):
        return True, 0, int(len(np.unique(np.concatenate([CA.ravel(), CB.ravel()]))))

    prev_num = int(len(np.unique(np.concatenate([CA.ravel(), CB.ravel()]))))
    for it in range(1, max_iter + 1):
        sig_to_id = {}
        next_id = 0
        next_arrays = []
        for C in (CA, CB):
            N = np.empty_like(C, dtype=np.int32)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        neigh = [(int(C[w, j, k]), int(C[i, w, k]), int(C[i, j, w])) for w in range(n)]
                        neigh.sort()
                        sig = (int(C[i, j, k]), tuple(neigh))
                        cid = sig_to_id.get(sig)
                        if cid is None:
                            cid = next_id
                            sig_to_id[sig] = cid
                            next_id += 1
                        N[i, j, k] = cid
            next_arrays.append(N)
        NA, NB = next_arrays
        if color_hist_diff(NA, NB):
            return True, it, next_id
        CA, CB = NA, NB
        if next_id == prev_num:
            return False, it, next_id
        prev_num = next_id
    return False, max_iter, prev_num


def fmt_bin_delta(delta_vec: np.ndarray):
    return ';'.join(f'nu{nu}:{int(v):+d}' for nu, v in enumerate(delta_vec) if int(v) != 0) or 'all_zero'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', default='/content/BREC_data_all.zip')
    ap.add_argument('--stage2c-csv', default=None)
    ap.add_argument('--outdir', default='/content/brec_stage2D')
    ap.add_argument('--fwl-max-iter', type=int, default=10)
    ap.add_argument('--pair-limit', type=int, default=0, help='0=all 15; use 3 for smoke test')
    ap.add_argument('--top-drivers', type=int, default=8)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    brec_path = locate_brec_v3(args.zip, outdir)
    arr = np.load(brec_path, allow_pickle=True)
    stage2c = locate_stage2c_csv(args.stage2c_csv)
    deep = load_deep_pairs(stage2c)
    if args.pair_limit > 0:
        deep = deep.iloc[:args.pair_limit].copy()

    print(f'Using BREC file: {brec_path}')
    print(f'Loaded {len(deep)} Stage-2D pairs: {deep.pair_id.tolist()}')
    print('3-FWL is used here; in graph-distinguishing power it corresponds to ordinary 4-WL.')

    catalogs = {}
    all_meta = []
    for k in sorted(set(deep.k.astype(int).tolist())):
        print(f'Building exact k={k} graphlet catalog + metadata ...', flush=True)
        cat = build_graphlet_catalog(k)
        catalogs[k] = cat
        all_meta.append(cat[-1])
        print(f'  k={k}: {len(cat[0])} unlabeled graphlets, nullity bins={sorted(cat[-1].gf2_nullity.unique().tolist())}')
    meta_all = pd.concat(all_meta, ignore_index=True)
    meta_all.to_csv(outdir / 'graphlet_catalog_metadata.csv', index=False)

    rows = []
    long_rows = []
    bin_rows = []
    t0 = time.time()

    for rr, row in deep.iterrows():
        pid = int(row['pair_id'])
        k = int(row['k'])
        base = pid * PAIR_STRIDE
        if base + 1 >= len(arr):
            raise IndexError(f'Pair {pid} out of range for brec_v3.npy length={len(arr)}')
        G = nx.convert_node_labels_to_integers(g6_to_graph(arr[base]), ordering='sorted')
        H = nx.convert_node_labels_to_integers(g6_to_graph(arr[base + 1]), ordering='sorted')
        AG = nx.to_numpy_array(G, dtype=np.uint8)
        AH = nx.to_numpy_array(H, dtype=np.uint8)
        n = G.number_of_nodes()
        m = G.number_of_edges()

        atlas, pos, mapping, class_nullity, meta = catalogs[k]
        gprof = graphlet_profile(AG, k, pos, mapping, len(atlas))
        hprof = graphlet_profile(AH, k, pos, mapping, len(atlas))
        delta = gprof - hprof
        knotG = nullity_profile_from_graphlets(gprof, class_nullity, k)
        knotH = nullity_profile_from_graphlets(hprof, class_nullity, k)
        knot_delta = knotG - knotH
        knot_sep = not np.array_equal(knotG, knotH)
        if not knot_sep:
            print(f'WARNING pair {pid}: KnotState no longer separates at k={k}')

        total = math.comb(n, k)
        knot_tv = 0.5 * np.abs(knot_delta / total).sum()
        graphlet_tv = 0.5 * np.abs(delta / total).sum()
        dom_nu = int(np.argmax(np.abs(knot_delta)))

        fwl_sep, fwl_iter, fwl_colors = fwl3_pair_separated(AG.astype(bool), AH.astype(bool), max_iter=args.fwl_max_iter)
        beyond4wl_level = bool(knot_sep and (not fwl_sep))

        # Per-nullity mechanism decomposition: signed total, absolute mass, cancellation.
        for nu in sorted(set(int(x) for x in class_nullity.tolist())):
            ids = np.flatnonzero(class_nullity == nu)
            signed = int(delta[ids].sum())
            abs_mass = int(np.abs(delta[ids]).sum())
            cancel_ratio = (abs(signed) / abs_mass) if abs_mass > 0 else 0.0
            bin_rows.append({
                'pair_id': pid, 'k': k, 'nullity': nu,
                'signed_graphlet_delta_sum': signed,
                'knot_bin_delta': int(knot_delta[nu]),
                'absolute_graphlet_delta_mass': abs_mass,
                'cancellation_ratio': float(cancel_ratio),
                'normalized_bin_delta': float(signed / total),
            })
            if signed != int(knot_delta[nu]):
                raise RuntimeError(f'Pair {pid} nu={nu}: graphlet->nullity aggregation mismatch')

        # Per-graphlet driver rows.
        diff_ids = np.flatnonzero(delta != 0)
        for cid in diff_ids:
            md = meta.iloc[int(cid)]
            long_rows.append({
                'pair_id': pid, 'n_g': n, 'm_g': m, 'k': k,
                'graphlet_class_id': int(cid),
                'graph6': md.graph6,
                'gf2_rank': int(md.gf2_rank),
                'gf2_nullity': int(md.gf2_nullity),
                'edges': int(md.edges),
                'triangles': int(md.triangles),
                'components': int(md.components),
                'treewidth_exact': int(md.treewidth_exact),
                'degree_sequence': md.degree_sequence,
                'count_G': int(gprof[cid]),
                'count_H': int(hprof[cid]),
                'delta': int(delta[cid]),
                'abs_delta': int(abs(delta[cid])),
                'normalized_delta': float(delta[cid] / total),
                'abs_normalized_delta': float(abs(delta[cid]) / total),
            })

        top_ids = sorted(diff_ids.tolist(), key=lambda c: abs(int(delta[c])), reverse=True)[:args.top_drivers]
        top_text = ','.join(f"g{c}(nu={int(class_nullity[c])},d={int(delta[c]):+d})" for c in top_ids)

        rows.append({
            'pair_id': pid,
            'subgroup': row.get('subgroup', 'StronglyRegular'),
            'n_g': n, 'm_g': m, 'k': k,
            'full_graphlet_dim': len(atlas),
            'knot_nullity_bins': int(len(np.unique(class_nullity))),
            'compression_ratio': float(len(atlas) / len(np.unique(class_nullity))),
            'graphlet_tv': float(graphlet_tv),
            'knot_tv': float(knot_tv),
            'dominant_nullity': dom_nu,
            'knot_delta_by_nullity': fmt_bin_delta(knot_delta),
            'changed_graphlet_classes': int(len(diff_ids)),
            'top_graphlet_drivers': top_text,
            'three_fwl_separated': bool(fwl_sep),
            'three_fwl_first_iter': int(fwl_iter),
            'three_fwl_final_colors': int(fwl_colors),
            'knot_beyond_3fwl_4wl_level': beyond4wl_level,
        })

        print(
            f"{rr+1:2d}/{len(deep)} pair={pid} k={k} | 3FWL_sep={int(fwl_sep)} "
            f"| Knot_sep={int(knot_sep)} | beyond_4WL_level={int(beyond4wl_level)} "
            f"| dom_nu={dom_nu} | elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    res = pd.DataFrame(rows)
    long_df = pd.DataFrame(long_rows)
    bins_df = pd.DataFrame(bin_rows)
    res.to_csv(outdir / 'pair_audit.csv', index=False)
    long_df.to_csv(outdir / 'graphlet_driver_long.csv', index=False)
    bins_df.to_csv(outdir / 'nullity_bin_mechanism.csv', index=False)

    # Global driver summary, separately by k.
    global_driver_frames = []
    if not long_df.empty:
        for k, sub in long_df.groupby('k'):
            agg = sub.groupby(['k','graphlet_class_id','graph6','gf2_rank','gf2_nullity','edges','triangles','components','treewidth_exact','degree_sequence']).agg(
                pairs_changed=('pair_id','nunique'),
                total_abs_delta=('abs_delta','sum'),
                mean_abs_freq_delta=('abs_normalized_delta','mean'),
                signed_delta_sum=('delta','sum'),
            ).reset_index().sort_values(['pairs_changed','total_abs_delta'], ascending=[False,False])
            agg.to_csv(outdir / f'global_graphlet_drivers_k{k}.csv', index=False)
            global_driver_frames.append(agg)
    if global_driver_frames:
        pd.concat(global_driver_frames, ignore_index=True).to_csv(outdir / 'global_graphlet_drivers_all.csv', index=False)

    summary = pd.DataFrame([{
        'pairs': len(res),
        'three_fwl_sep': int(res.three_fwl_separated.sum()),
        'three_fwl_fail': int((~res.three_fwl_separated).sum()),
        'knot_beyond_3fwl_4wl_level': int(res.knot_beyond_3fwl_4wl_level.sum()),
        'k5_pairs': int((res.k == 5).sum()),
        'k6_pairs': int((res.k == 6).sum()),
        'mean_graphlet_tv': float(res.graphlet_tv.mean()),
        'mean_knot_tv': float(res.knot_tv.mean()),
        'mean_compression_ratio': float(res.compression_ratio.mean()),
    }])
    summary.to_csv(outdir / 'summary.csv', index=False)

    print('\n=== STAGE 2D SUMMARY ===')
    print(summary.to_string(index=False))
    print('\n=== PAIR AUDIT ===')
    cols = ['pair_id','n_g','m_g','k','compression_ratio','dominant_nullity','knot_delta_by_nullity',
            'changed_graphlet_classes','three_fwl_separated','three_fwl_first_iter','knot_beyond_3fwl_4wl_level']
    print(res[cols].to_string(index=False))

    print('\n=== NULLITY MECHANISM (first rows) ===')
    print(bins_df.head(30).to_string(index=False))

    if not long_df.empty and 5 in long_df.k.unique():
        top5 = pd.read_csv(outdir / 'global_graphlet_drivers_k5.csv').head(12)
        print('\n=== GLOBAL k=5 GRAPHLET DRIVERS (top 12) ===')
        print(top5[['graphlet_class_id','gf2_nullity','edges','triangles','treewidth_exact','pairs_changed','total_abs_delta','mean_abs_freq_delta']].to_string(index=False))

    print('\nInterpretation guardrails:')
    print('  * 3-FWL is used as the 4-WL-level test (known equivalent graph-distinguishing power).')
    print('  * KnotState is an algebraic compression of same-k graphlet counts; it cannot contain more information than the full same-k profile.')
    print('  * The mechanism CSV shows exactly which graphlet differences survive aggregation into each GF(2)-nullity bin.')
    print(f'Files saved under: {outdir}')


if __name__ == '__main__':
    main()
