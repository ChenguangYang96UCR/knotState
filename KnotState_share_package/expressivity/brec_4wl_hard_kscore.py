#!/usr/bin/env python3
"""Screen the final 20 official BREC CFI pairs with global KnotState.

By the ordering used in the BREC CFI block, local indices 80..99 are the 20
pairs designated as 4-WL-hard in the benchmark construction. This script is a
KnotState screening tool. Monte Carlo separation is NOT an exact certificate;
positive candidates should be rerun with a larger budget or exact enumeration
when feasible.
"""
from __future__ import annotations
import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def as_bytes(z):
    if isinstance(z, bytes): return z.strip()
    if isinstance(z, np.bytes_): return bytes(z).strip()
    if isinstance(z, str): return z.encode().strip()
    return str(z).encode().strip()


def graph_to_bitrows(G):
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    n = G.number_of_nodes()
    rows = [0] * n
    for u, v in G.edges():
        rows[int(u)] |= 1 << int(v)
        rows[int(v)] |= 1 << int(u)
    return n, rows


def gf2_nullity_subset(bitrows, subset):
    subset = [int(x) for x in subset]
    s = len(subset)
    rows = [0] * s
    for i, u in enumerate(subset):
        ru = int(bitrows[int(u)])
        mask = 0
        for j, v in enumerate(subset):
            if (ru >> int(v)) & 1:
                mask |= 1 << j
        rows[i] = mask
    rank = 0
    for col in range(s - 1, -1, -1):
        pivot = next((i for i in range(rank, s) if (rows[i] >> col) & 1), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pr = rows[rank]
        for i in range(s):
            if i != rank and ((rows[i] >> col) & 1):
                rows[i] ^= pr
        rank += 1
        if rank == s:
            break
    return s - rank


def sample_profile(n, rows, s, samples, seed):
    rng = np.random.default_rng(seed)
    allowed = list(range(s % 2, s + 1, 2))
    c = {nu: 0 for nu in allowed}
    for _ in range(samples):
        subset = rng.choice(n, size=s, replace=False)
        nu = gf2_nullity_subset(rows, subset)
        c[nu] = c.get(nu, 0) + 1
    return c


def one_pair(local_id, pair, samples, repeats, seed):
    A = nx.from_graph6_bytes(as_bytes(pair[0])); B = nx.from_graph6_bytes(as_bytes(pair[1]))
    nA, rA = graph_to_bitrows(A); nB, rB = graph_to_bitrows(B)
    if nA != nB:
        raise RuntimeError(f"pair {local_id}: n mismatch")
    rows = []
    for s in range(2, 7):
        for rep in range(repeats):
            ca = sample_profile(nA, rA, s, samples, seed + local_id*100003 + s*1009 + rep*17)
            cb = sample_profile(nB, rB, s, samples, seed + local_id*100003 + s*1009 + rep*17 + 99991)
            for nu in sorted(set(ca) | set(cb)):
                pA, pB = ca.get(nu,0)/samples, cb.get(nu,0)/samples
                se = math.sqrt(max(pA*(1-pA),0)/samples + max(pB*(1-pB),0)/samples)
                z = abs(pA-pB)/se if se > 0 else (math.inf if pA != pB else 0.0)
                rows.append({"local_id":local_id,"brec_pair":260+local_id,"s":s,"nu":nu,"rep":rep,
                             "p_A":pA,"p_B":pB,"abs_diff":abs(pA-pB),"z":z})
    df = pd.DataFrame(rows)
    agg = df.groupby(["s","nu"], as_index=False).agg(abs_diff=("abs_diff","mean"), z=("z","mean"))
    best = agg.sort_values(["z","abs_diff"], ascending=False).iloc[0]
    strong = bool(best.z >= 6.0 and best.abs_diff > 0)
    return {"brec_pair":260+local_id,"cfi_local":local_id,"n":nA,"m_A":A.number_of_edges(),"m_B":B.number_of_edges(),
            "best_s":int(best.s),"best_nullity":int(best.nu),"best_abs_diff":float(best.abs_diff),"best_z":float(best.z),
            "mc_6sigma_candidate":strong}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfi", type=Path, required=True, help="official BREC customize/Data/raw/cfi.npy")
    ap.add_argument("--samples", type=int, default=100000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--out", type=Path, default=Path("brec_4wl_hard_kscore"))
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    x = np.load(args.cfi, allow_pickle=True)
    if x.ndim != 2 or x.shape[0] < 100 or x.shape[1] < 2:
        raise RuntimeError(f"unexpected cfi.npy shape {x.shape}")
    tasks = [(i, x[i]) for i in range(80,100)]
    summaries, details = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(one_pair,i,p,args.samples,args.repeats,args.seed): i for i,p in tasks}
        for done, fut in enumerate(as_completed(futs), 1):
            s, d = fut.result(); summaries.append(s); details.extend(d)
            print(f"completed {done}/20 | BREC {s['brec_pair']} | candidate={s['mc_6sigma_candidate']} | best=(s={s['best_s']},nu={s['best_nullity']},z={s['best_z']:.2f})", flush=True)
    sdf = pd.DataFrame(summaries).sort_values("brec_pair")
    pd.DataFrame(details).to_csv(args.out/"channel_details.csv", index=False)
    sdf.to_csv(args.out/"pair_summary.csv", index=False)
    print("\n=== BREC 4-WL-HARD KNOTSTATE SCREEN ===")
    print(sdf.to_string(index=False))
    print("Strong MC candidates:", int(sdf.mc_6sigma_candidate.sum()), "/ 20")


if __name__ == "__main__":
    main()
