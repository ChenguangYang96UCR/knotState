#!/usr/bin/env python3
"""
BREC KnotState Expressivity Scan (Colab-friendly, CPU only)

Goal
----
For the 400 official BREC graph pairs, compare several permutation-invariant
structural tests:
  1) 1-WL color refinement (pairwise, exact modulo deterministic integer colors)
  2) adjacency spectrum
  3) normalized Laplacian spectrum
  4) basic structural signature: degree sequence + triangles + 4-cycles + components
  5) KnotState induced-subset GF(2) rank/nullity distribution

KnotState for subset size k is
  p_G(k, nu) = #{S subset V: |S|=k, nullity_F2(A[S])=nu} / C(n,k).

For each k, this script computes the distribution exactly when C(n,k) is below
--exact-limit. Otherwise it estimates the distribution by uniform Monte Carlo.
A sampled pair is declared separated only with conservative statistical evidence:
  max_z >= --z-threshold AND max_abs_bin_difference >= --effect-min.
Any exact k-bin difference is accepted as exact separation.

This is an expressivity diagnostic, NOT the official BREC RPC neural evaluation.
It is designed to answer whether the KnotState structural descriptor itself
separates graph pairs that common baselines fail to separate.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd


BREC_URLS = [
    "https://github.com/GraphPKU/BREC/raw/refs/heads/Release/BREC_data_all.zip",
    "https://raw.githubusercontent.com/GraphPKU/BREC/Release/BREC_data_all.zip",
]

PAIR_NUM = 400
NUM_RELABEL = 32
PAIR_BLOCK = NUM_RELABEL * 2  # x0,y0,x1,y1,...,x31,y31
CATEGORIES = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}


def category_of(pair_id: int) -> str:
    for name, (lo, hi) in CATEGORIES.items():
        if lo <= pair_id < hi:
            return name
    return "Unknown"


def download_brec(root: Path, user_zip: str | None = None) -> Path:
    """Return path to brec_v3.npy, downloading/unzipping official data if needed."""
    root.mkdir(parents=True, exist_ok=True)
    found = list(root.rglob("brec_v3.npy"))
    if found:
        return found[0]

    zip_path = root / "BREC_data_all.zip"
    if user_zip:
        src = Path(user_zip)
        if not src.exists():
            raise FileNotFoundError(src)
        if src.resolve() != zip_path.resolve():
            shutil.copy2(src, zip_path)
    elif not zip_path.exists():
        last_err = None
        for url in BREC_URLS:
            try:
                print(f"Downloading official BREC data from:\n  {url}", flush=True)
                urllib.request.urlretrieve(url, zip_path)
                if zip_path.stat().st_size < 100_000:
                    raise RuntimeError("downloaded file is unexpectedly small")
                break
            except Exception as e:
                last_err = e
                if zip_path.exists():
                    zip_path.unlink()
        else:
            raise RuntimeError(
                "Automatic BREC download failed. Download BREC_data_all.zip from "
                "https://github.com/GraphPKU/BREC/tree/Release and upload it to Colab, "
                "then rerun with --zip /content/BREC_data_all.zip"
            ) from last_err

    print(f"Unzipping {zip_path} ...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    found = list(root.rglob("brec_v3.npy"))
    if not found:
        raise FileNotFoundError("brec_v3.npy was not found after extracting BREC_data_all.zip")
    return found[0]


def as_graph6_bytes(x) -> bytes:
    if isinstance(x, bytes):
        return x.strip()
    if isinstance(x, np.bytes_):
        return bytes(x).strip()
    if isinstance(x, str):
        return x.encode().strip()
    # Some object arrays may wrap a scalar.
    if isinstance(x, np.ndarray) and x.shape == ():
        return as_graph6_bytes(x.item())
    raise TypeError(f"Unsupported graph6 object type: {type(x)}")


def load_original_pairs(npy_path: Path, pair_limit: int):
    arr = np.load(npy_path, allow_pickle=True)
    expected = PAIR_NUM * PAIR_BLOCK
    if arr.size < expected:
        raise ValueError(
            f"BREC array has {arr.size} entries; expected at least {expected}. "
            "This does not look like official brec_v3.npy."
        )
    pairs = []
    for pid in range(pair_limit):
        base = pid * PAIR_BLOCK
        gx = nx.from_graph6_bytes(as_graph6_bytes(arr[base]))
        gy = nx.from_graph6_bytes(as_graph6_bytes(arr[base + 1]))
        gx = nx.Graph(gx)
        gy = nx.Graph(gy)
        gx.remove_edges_from(nx.selfloop_edges(gx))
        gy.remove_edges_from(nx.selfloop_edges(gy))
        pairs.append((gx, gy))
    return pairs


# ---------- Exact baseline invariants ----------

def wl_pair_separated(g: nx.Graph, h: nx.Graph) -> bool:
    """1-WL comparison by refining the disjoint union with a shared color dictionary."""
    n1, n2 = g.number_of_nodes(), h.number_of_nodes()
    g_nodes = list(g.nodes())
    h_nodes = list(h.nodes())
    gi = {v: i for i, v in enumerate(g_nodes)}
    hi = {v: n1 + i for i, v in enumerate(h_nodes)}
    neigh = [[] for _ in range(n1 + n2)]
    for u, v in g.edges():
        a, b = gi[u], gi[v]
        neigh[a].append(b); neigh[b].append(a)
    for u, v in h.edges():
        a, b = hi[u], hi[v]
        neigh[a].append(b); neigh[b].append(a)

    colors = np.zeros(n1 + n2, dtype=np.int64)
    prev_pair_hist = None
    for _ in range(max(n1 + n2, 1) + 2):
        sigs = [(int(colors[i]), tuple(sorted(int(colors[j]) for j in neigh[i])))
                for i in range(n1 + n2)]
        uniq = {s: idx for idx, s in enumerate(sorted(set(sigs)))}
        new_colors = np.array([uniq[s] for s in sigs], dtype=np.int64)
        hg = tuple(sorted(Counter(new_colors[:n1]).items()))
        hh = tuple(sorted(Counter(new_colors[n1:]).items()))
        if hg != hh:
            return True
        pair_hist = (hg, hh)
        if np.array_equal(new_colors, colors) or pair_hist == prev_pair_hist:
            return False
        prev_pair_hist = pair_hist
        colors = new_colors
    return False


def adjacency_spectrum(g: nx.Graph) -> np.ndarray:
    A = nx.to_numpy_array(g, dtype=np.float64)
    return np.linalg.eigvalsh(A)


def norm_laplacian_spectrum(g: nx.Graph) -> np.ndarray:
    A = nx.to_numpy_array(g, dtype=np.float64)
    d = A.sum(axis=1)
    inv = np.zeros_like(d)
    nz = d > 0
    inv[nz] = 1.0 / np.sqrt(d[nz])
    L = np.eye(len(d), dtype=np.float64) - (inv[:, None] * A * inv[None, :])
    # For isolated vertices, standard normalized Laplacian convention has diagonal 0.
    L[~nz, ~nz] = 0.0
    return np.linalg.eigvalsh(L)


def spectra_different(a: np.ndarray, b: np.ndarray, tol: float = 1e-7) -> bool:
    if a.shape != b.shape:
        return True
    return not np.allclose(a, b, atol=tol, rtol=0.0)


def count_four_cycles(g: nx.Graph) -> int:
    A = nx.to_numpy_array(g, dtype=np.int64)
    A2 = A @ A
    iu = np.triu_indices_from(A2, k=1)
    cn = A2[iu]
    # Each 4-cycle has two opposite vertex pairs.
    return int(np.sum(cn * (cn - 1) // 2) // 2)


def basic_signature(g: nx.Graph):
    deg = tuple(sorted(d for _, d in g.degree()))
    tri = int(sum(nx.triangles(g).values()) // 3)
    c4 = count_four_cycles(g)
    comps = tuple(sorted((len(c) for c in nx.connected_components(g)), reverse=True))
    return (g.number_of_nodes(), g.number_of_edges(), deg, tri, c4, comps)


# ---------- GF(2) KnotState ----------

def adjacency_bits(g: nx.Graph):
    nodes = list(g.nodes())
    pos = {v: i for i, v in enumerate(nodes)}
    bits = [0] * len(nodes)
    for v in nodes:
        i = pos[v]
        b = 0
        for u in g.neighbors(v):
            b |= 1 << pos[u]
        bits[i] = b
    return bits


def gf2_rank_mask(adj_bits, mask: int) -> int:
    """Rank over GF(2) of the adjacency principal submatrix indexed by mask."""
    basis = {}
    rank = 0
    mm = mask
    while mm:
        lsb = mm & -mm
        i = lsb.bit_length() - 1
        x = adj_bits[i] & mask
        while x:
            p = x.bit_length() - 1
            b = basis.get(p)
            if b is None:
                basis[p] = x
                rank += 1
                break
            x ^= b
        mm ^= lsb
    return rank


def combo_mask(combo) -> int:
    m = 0
    for v in combo:
        m |= 1 << int(v)
    return m


def nullity_counts_exact(adj_bits, n: int, k: int) -> np.ndarray:
    counts = np.zeros(k + 1, dtype=np.int64)
    for comb in itertools.combinations(range(n), k):
        mask = combo_mask(comb)
        r = gf2_rank_mask(adj_bits, mask)
        counts[k - r] += 1
    return counts


def nullity_counts_sampled(adj_bits, n: int, k: int, samples: int, seed: int) -> np.ndarray:
    counts = np.zeros(k + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        inds = rng.choice(n, size=k, replace=False)
        mask = combo_mask(inds)
        r = gf2_rank_mask(adj_bits, mask)
        counts[k - r] += 1
    return counts


def graph_seed(g: nx.Graph) -> int:
    # Used only to decorrelate MC streams, not as an invariant value.
    # n/m/degree multiset are permutation-invariant.
    payload = f"{g.number_of_nodes()}|{g.number_of_edges()}|{tuple(sorted(d for _,d in g.degree()))}"
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def compute_knot_signature(
    g: nx.Graph,
    max_k: int,
    samples: int,
    repeats: int,
    exact_limit: int,
    base_seed: int,
):
    n = g.number_of_nodes()
    bits = adjacency_bits(g)
    out = {}
    for k in range(2, min(max_k, n) + 1):
        total_comb = math.comb(n, k)
        if total_comb <= exact_limit:
            c = nullity_counts_exact(bits, n, k)
            out[k] = {
                "exact": True,
                "counts": c.astype(np.int64),
                "total": int(c.sum()),
                "probs": c.astype(np.float64) / max(int(c.sum()), 1),
            }
        else:
            all_counts = np.zeros(k + 1, dtype=np.int64)
            # Same sample count for all graphs; streams are reproducible.
            gs = graph_seed(g)
            for r in range(repeats):
                seed = (base_seed + 1000003 * k + 9176 * r + gs) % (2**32 - 1)
                all_counts += nullity_counts_sampled(bits, n, k, samples, seed)
            total = int(all_counts.sum())
            out[k] = {
                "exact": False,
                "counts": all_counts,
                "total": total,
                "probs": all_counts.astype(np.float64) / max(total, 1),
            }
    return out


def compare_knot_signatures(sig_g, sig_h, z_threshold: float, effect_min: float):
    first_k = None
    evidence = None
    max_z = 0.0
    max_diff = 0.0
    max_tv = 0.0
    details = []

    for k in sorted(set(sig_g) & set(sig_h)):
        a, b = sig_g[k], sig_h[k]
        pa, pb = a["probs"], b["probs"]
        L = max(len(pa), len(pb))
        pa = np.pad(pa, (0, L-len(pa)))
        pb = np.pad(pb, (0, L-len(pb)))
        diff = np.abs(pa - pb)
        tv = 0.5 * float(diff.sum())
        local_max_diff = float(diff.max(initial=0.0))
        max_diff = max(max_diff, local_max_diff)
        max_tv = max(max_tv, tv)

        if a["exact"] and b["exact"]:
            separated = bool(np.any(a["counts"] != b["counts"]))
            z = float("inf") if separated else 0.0
            ev = "exact"
        else:
            na, nb = max(a["total"], 1), max(b["total"], 1)
            # Conservative binwise normal approximation to two multinomial proportions.
            se = np.sqrt(pa*(1-pa)/na + pb*(1-pb)/nb + 1e-18)
            zvec = diff / se
            z = float(np.max(zvec, initial=0.0))
            separated = bool(z >= z_threshold and local_max_diff >= effect_min)
            ev = "sampled"
        if math.isfinite(z):
            max_z = max(max_z, z)
        elif separated:
            max_z = float("inf")
        details.append({"k": k, "exact": a["exact"] and b["exact"], "tv": tv,
                        "max_abs_diff": local_max_diff, "max_z": z, "separated": separated})
        if separated and first_k is None:
            first_k = k
            evidence = ev

    return {
        "separated": first_k is not None,
        "first_k": first_k,
        "evidence": evidence,
        "max_z": max_z,
        "max_abs_diff": max_diff,
        "max_tv": max_tv,
        "details": details,
    }


def cache_key(g: nx.Graph, args) -> str:
    # Graph6 bytes here are only a cache identifier; not used as the descriptor.
    g6 = nx.to_graph6_bytes(g, header=False).strip()
    h = hashlib.sha256(g6).hexdigest()
    return f"{h}|k{args.max_k}|s{args.samples}|r{args.repeats}|e{args.exact_limit}|seed{args.seed}"


def load_cache(path: Path):
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_cache(path: Path, cache):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def get_sig(g, args, cache):
    key = cache_key(g, args)
    if key not in cache:
        cache[key] = compute_knot_signature(
            g, args.max_k, args.samples, args.repeats, args.exact_limit, args.seed
        )
    return cache[key]


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    def agg_block(sub):
        n = len(sub)
        return pd.Series({
            "pairs": n,
            "wl_sep": int(sub.wl_separated.sum()),
            "adj_spectrum_sep": int(sub.adj_spectrum_separated.sum()),
            "lap_spectrum_sep": int(sub.lap_spectrum_separated.sum()),
            "basic_sep": int(sub.basic_stats_separated.sum()),
            "knot_sep": int(sub.knot_separated.sum()),
            "knot_sep_rate": float(sub.knot_separated.mean()) if n else np.nan,
            "knot_beyond_1wl": int((sub.knot_separated & ~sub.wl_separated).sum()),
            "hard_wins": int(sub.hard_win.sum()),
            "hard_win_rate": float(sub.hard_win.mean()) if n else np.nan,
        })
    pieces = []
    overall = agg_block(df); overall["category"] = "ALL"; pieces.append(overall)
    for cat in CATEGORIES:
        sub = df[df.category == cat]
        if len(sub):
            s = agg_block(sub); s["category"] = cat; pieces.append(s)
    out = pd.DataFrame(pieces)
    cols = ["category"] + [c for c in out.columns if c != "category"]
    return out[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/brec_knotstate", help="output/data directory")
    ap.add_argument("--zip", default=None, help="optional uploaded BREC_data_all.zip path")
    ap.add_argument("--pair-limit", type=int, default=400, help="1..400; use 60 for smoke test")
    ap.add_argument("--max-k", type=int, default=8, help="largest induced subset size")
    ap.add_argument("--samples", type=int, default=1000, help="MC samples per k per repeat when not exact")
    ap.add_argument("--repeats", type=int, default=3, help="MC repeats")
    ap.add_argument("--exact-limit", type=int, default=200000,
                    help="enumerate all k-subsets when C(n,k) <= this")
    ap.add_argument("--z-threshold", type=float, default=5.0)
    ap.add_argument("--effect-min", type=float, default=0.01,
                    help="minimum absolute bin-probability difference for sampled separation")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    args.pair_limit = max(1, min(PAIR_NUM, args.pair_limit))

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    npy = download_brec(root, args.zip)
    print(f"Using BREC file: {npy}", flush=True)
    pairs = load_original_pairs(npy, args.pair_limit)
    print(f"Loaded {len(pairs)} original BREC pairs.", flush=True)

    cache_path = root / "knotstate_cache.pkl"
    cache = load_cache(cache_path)
    rows = []
    t0 = time.time()

    for pid, (g, h) in enumerate(pairs):
        wl_sep = wl_pair_separated(g, h)
        adj_sep = spectra_different(adjacency_spectrum(g), adjacency_spectrum(h))
        lap_sep = spectra_different(norm_laplacian_spectrum(g), norm_laplacian_spectrum(h))
        basic_sep = basic_signature(g) != basic_signature(h)

        sg = get_sig(g, args, cache)
        sh = get_sig(h, args, cache)
        kc = compare_knot_signatures(sg, sh, args.z_threshold, args.effect_min)

        hard_win = bool(
            kc["separated"] and
            (not wl_sep) and
            (not adj_sep) and
            (not lap_sep) and
            (not basic_sep)
        )
        rows.append({
            "pair_id": pid,
            "category": category_of(pid),
            "n_g": g.number_of_nodes(), "n_h": h.number_of_nodes(),
            "m_g": g.number_of_edges(), "m_h": h.number_of_edges(),
            "wl_separated": wl_sep,
            "adj_spectrum_separated": adj_sep,
            "lap_spectrum_separated": lap_sep,
            "basic_stats_separated": basic_sep,
            "knot_separated": kc["separated"],
            "knot_first_k": kc["first_k"],
            "knot_evidence": kc["evidence"],
            "knot_max_z": kc["max_z"],
            "knot_max_abs_diff": kc["max_abs_diff"],
            "knot_max_tv": kc["max_tv"],
            "hard_win": hard_win,
        })

        if (pid + 1) % 10 == 0 or pid == len(pairs) - 1:
            df_now = pd.DataFrame(rows)
            df_now.to_csv(root / "pair_results.csv", index=False)
            save_cache(cache_path, cache)
            print(
                f"{pid+1:3d}/{len(pairs)} | knot_sep={int(df_now.knot_separated.sum())} "
                f"| hard_wins={int(df_now.hard_win.sum())} | elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(root / "pair_results.csv", index=False)
    summary = make_summary(df)
    summary.to_csv(root / "summary.csv", index=False)

    examples = df[df.hard_win].copy()
    examples = examples.sort_values(["knot_evidence", "knot_first_k", "knot_max_tv"],
                                    ascending=[True, True, False])
    examples.head(50).to_csv(root / "hard_examples.csv", index=False)

    # Also expose the less strict set: Knot separates while 1-WL does not.
    beyond = df[df.knot_separated & ~df.wl_separated].copy()
    beyond.to_csv(root / "knot_beyond_1wl.csv", index=False)

    report = []
    report.append("# BREC KnotState Expressivity Scan\n")
    report.append(f"Pairs analyzed: {len(df)}")
    report.append(f"KnotState separated: {int(df.knot_separated.sum())}/{len(df)}")
    report.append(f"KnotState beyond 1-WL: {int((df.knot_separated & ~df.wl_separated).sum())}/{len(df)}")
    report.append(
        "Hard wins (Knot separates while 1-WL, adjacency spectrum, normalized-Laplacian "
        f"spectrum, and basic degree/triangle/4-cycle stats all fail): {int(df.hard_win.sum())}/{len(df)}"
    )
    report.append("\nImportant: sampled KnotState separations are statistical, not exact proofs. "
                  "Exact-evidence rows arise only from fully enumerated subset sizes.")
    (root / "report.md").write_text("\n\n".join(report))

    print("\n=== BREC EXPRESSIVITY SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== KEY COUNTS ===")
    print("KnotState separated:", int(df.knot_separated.sum()), "/", len(df))
    print("Knot beyond 1-WL:", int((df.knot_separated & ~df.wl_separated).sum()), "/", len(df))
    print("Hard wins:", int(df.hard_win.sum()), "/", len(df))
    if len(examples):
        print("\nFirst hard examples:")
        print(examples.head(15)[[
            "pair_id", "category", "n_g", "m_g", "knot_first_k", "knot_evidence",
            "knot_max_abs_diff", "knot_max_tv"
        ]].to_string(index=False))
    else:
        print("\nNo hard examples under the current conservative thresholds.")
    print(f"\nFiles saved under: {root}")
    print("NOTE: This is a descriptor expressivity scan, not official BREC RPC neural-model accuracy.")


if __name__ == "__main__":
    main()
