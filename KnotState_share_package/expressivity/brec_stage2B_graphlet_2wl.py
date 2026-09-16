#!/usr/bin/env python3
"""
BREC Stage 2B: full 4-node graphlets + 2-WL vs KnotState (Colab-friendly, CPU only)

Default scope: the 100 BREC Regular pairs, pair ids 60..159.
This script compares:
  - 1-WL
  - 2-WL (2-dimensional Weisfeiler-Lehman on ordered pairs)
  - adjacency spectrum
  - normalized Laplacian spectrum
  - basic stats: degree sequence + triangles + 4-cycles + components
  - full induced 4-node graphlet counts (all 11 unlabeled simple graphs on 4 vertices)
  - KnotState GF(2) rank/nullity signature

The key outputs are:
  knot_beyond_graphlet4:
      KnotState separates, but full 4-node graphlet counts do not.
  knot_beyond_2wl:
      KnotState separates, but 2-WL does not.
  deep_win:
      KnotState separates while 1-WL, 2-WL, both spectra, basic stats,
      and full induced 4-node graphlet counts all fail.

Important mathematical audit:
At subset size k=4, the KnotState nullity histogram is a grouping of the 11
induced 4-node graphlet counts according to GF(2) adjacency nullity. Therefore,
an exact k=4 KnotState separation MUST imply a full 4-graphlet separation.
The script checks this and raises a warning if the cached results violate it.

This is a descriptor expressivity scan, not official BREC RPC neural accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pickle
import shutil
import time
import urllib.request
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


BREC_URLS = [
    "https://github.com/GraphPKU/BREC/raw/refs/heads/Release/BREC_data_all.zip",
    "https://raw.githubusercontent.com/GraphPKU/BREC/Release/BREC_data_all.zip",
]
PAIR_NUM = 400
NUM_RELABEL = 32
PAIR_BLOCK = NUM_RELABEL * 2

# Official BREC v3 indexing used by the release code.
# Regular 60..109 are simple regular, 110..159 are strongly regular.
def subgroup_of(pid: int) -> str:
    if 60 <= pid < 110:
        return "SimpleRegular"
    if 110 <= pid < 160:
        return "StronglyRegular"
    return "Other"


def download_brec(root: Path, user_zip: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    # Reuse the Stage 2A extraction if it exists in the same Colab runtime.
    prior = Path("/content/brec_knotstate/brec_v3.npy")
    if prior.exists():
        return prior

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
                    raise RuntimeError("downloaded BREC zip is unexpectedly small")
                break
            except Exception as e:
                last_err = e
                if zip_path.exists():
                    zip_path.unlink()
        else:
            raise RuntimeError(
                "Automatic BREC download failed. Upload BREC_data_all.zip and rerun "
                "with --zip /content/BREC_data_all.zip"
            ) from last_err

    print(f"Unzipping {zip_path} ...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    found = list(root.rglob("brec_v3.npy"))
    if not found:
        raise FileNotFoundError("brec_v3.npy not found after extracting BREC_data_all.zip")
    return found[0]


def as_graph6_bytes(x) -> bytes:
    if isinstance(x, bytes):
        return x.strip()
    if isinstance(x, np.bytes_):
        return bytes(x).strip()
    if isinstance(x, str):
        return x.encode().strip()
    if isinstance(x, np.ndarray) and x.shape == ():
        return as_graph6_bytes(x.item())
    raise TypeError(f"Unsupported graph6 object type: {type(x)}")


def load_pairs(npy_path: Path, start: int, end: int):
    arr = np.load(npy_path, allow_pickle=True)
    expected = PAIR_NUM * PAIR_BLOCK
    if arr.size < expected:
        raise ValueError(f"BREC array has {arr.size} entries; expected at least {expected}")
    out = []
    for pid in range(start, end):
        base = pid * PAIR_BLOCK
        g = nx.Graph(nx.from_graph6_bytes(as_graph6_bytes(arr[base])))
        h = nx.Graph(nx.from_graph6_bytes(as_graph6_bytes(arr[base + 1])))
        g.remove_edges_from(nx.selfloop_edges(g))
        h.remove_edges_from(nx.selfloop_edges(h))
        # Force compact integer labels for fast bit operations.
        g = nx.convert_node_labels_to_integers(g, ordering="sorted")
        h = nx.convert_node_labels_to_integers(h, ordering="sorted")
        out.append((pid, g, h))
    return out


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------

def one_wl_pair_separated(g: nx.Graph, h: nx.Graph) -> bool:
    if g.number_of_nodes() != h.number_of_nodes():
        return True
    n = g.number_of_nodes()
    ng = [sorted(g.neighbors(i)) for i in range(n)]
    nh = [sorted(h.neighbors(i)) for i in range(n)]
    cg = np.zeros(n, dtype=np.int64)
    ch = np.zeros(n, dtype=np.int64)

    for _ in range(2 * n + 5):
        sg = [(int(cg[i]), tuple(sorted(int(cg[j]) for j in ng[i]))) for i in range(n)]
        sh = [(int(ch[i]), tuple(sorted(int(ch[j]) for j in nh[i]))) for i in range(n)]
        all_sigs = sorted(set(sg + sh))
        cmap = {s: i for i, s in enumerate(all_sigs)}
        new_g = np.array([cmap[s] for s in sg], dtype=np.int64)
        new_h = np.array([cmap[s] for s in sh], dtype=np.int64)
        if Counter(new_g.tolist()) != Counter(new_h.tolist()):
            return True
        if np.array_equal(new_g, cg) and np.array_equal(new_h, ch):
            return False
        cg, ch = new_g, new_h
    return False


def init_2wl_colors(g: nx.Graph) -> np.ndarray:
    n = g.number_of_nodes()
    A = nx.to_numpy_array(g, nodelist=range(n), dtype=np.int8)
    c = np.full((n, n), 2, dtype=np.int64)  # nonedge / unequal
    c[A > 0] = 1                            # edge
    np.fill_diagonal(c, 0)                  # equal tuple
    return c


def two_wl_pair_separated(g: nx.Graph, h: nx.Graph, max_iter: int | None = None) -> bool:
    """Exact 2-WL comparison on ordered vertex pairs with shared color dictionaries."""
    if g.number_of_nodes() != h.number_of_nodes():
        return True
    n = g.number_of_nodes()
    cg = init_2wl_colors(g)
    ch = init_2wl_colors(h)
    if Counter(cg.ravel().tolist()) != Counter(ch.ravel().tolist()):
        return True

    if max_iter is None:
        max_iter = n * n + 5

    for _ in range(max_iter):
        sig_g = []
        sig_h = []
        # Signature of ordered pair (i,j): old color plus multiset over z of
        # (color(i,z), color(z,j)).
        for i in range(n):
            for j in range(n):
                mg = tuple(sorted((int(cg[i, z]), int(cg[z, j])) for z in range(n)))
                mh = tuple(sorted((int(ch[i, z]), int(ch[z, j])) for z in range(n)))
                sig_g.append((int(cg[i, j]), mg))
                sig_h.append((int(ch[i, j]), mh))

        all_sigs = sorted(set(sig_g + sig_h))
        cmap = {s: idx for idx, s in enumerate(all_sigs)}
        ng = np.array([cmap[s] for s in sig_g], dtype=np.int64).reshape(n, n)
        nh = np.array([cmap[s] for s in sig_h], dtype=np.int64).reshape(n, n)

        if Counter(ng.ravel().tolist()) != Counter(nh.ravel().tolist()):
            return True
        if np.array_equal(ng, cg) and np.array_equal(nh, ch):
            return False
        cg, ch = ng, nh

    return False


def adjacency_spectrum(g: nx.Graph) -> np.ndarray:
    return np.linalg.eigvalsh(nx.to_numpy_array(g, dtype=np.float64))


def norm_laplacian_spectrum(g: nx.Graph) -> np.ndarray:
    A = nx.to_numpy_array(g, dtype=np.float64)
    d = A.sum(axis=1)
    inv = np.zeros_like(d)
    nz = d > 0
    inv[nz] = 1.0 / np.sqrt(d[nz])
    L = np.eye(len(d), dtype=np.float64) - inv[:, None] * A * inv[None, :]
    L[~nz, ~nz] = 0.0
    return np.linalg.eigvalsh(L)


def spectra_different(a: np.ndarray, b: np.ndarray, tol: float = 1e-7) -> bool:
    return a.shape != b.shape or not np.allclose(a, b, atol=tol, rtol=0.0)


def count_four_cycles(g: nx.Graph) -> int:
    A = nx.to_numpy_array(g, dtype=np.int64)
    A2 = A @ A
    iu = np.triu_indices_from(A2, k=1)
    cn = A2[iu]
    return int(np.sum(cn * (cn - 1) // 2) // 2)


def basic_signature(g: nx.Graph):
    deg = tuple(sorted(d for _, d in g.degree()))
    tri = int(sum(nx.triangles(g).values()) // 3)
    c4 = count_four_cycles(g)
    comps = tuple(sorted((len(c) for c in nx.connected_components(g)), reverse=True))
    return (g.number_of_nodes(), g.number_of_edges(), deg, tri, c4, comps)


# -----------------------------------------------------------------------------
# Full induced 4-node graphlet counts
# -----------------------------------------------------------------------------

EDGE4 = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
PERMS4 = list(itertools.permutations(range(4)))


def permute_pattern4(pattern: int, perm) -> int:
    out = 0
    for bit, (i, j) in enumerate(EDGE4):
        oi, oj = perm[i], perm[j]
        if oi > oj:
            oi, oj = oj, oi
        old_bit = EDGE4.index((oi, oj))
        if (pattern >> old_bit) & 1:
            out |= 1 << bit
    return out


def build_graphlet4_classes():
    canon = {}
    for p in range(64):
        canon[p] = min(permute_pattern4(p, perm) for perm in PERMS4)
    reps = sorted(set(canon.values()))
    if len(reps) != 11:
        raise RuntimeError(f"Expected 11 unlabeled graphs on 4 vertices, got {len(reps)}")
    rep_to_id = {r: i for i, r in enumerate(reps)}
    pattern_to_id = {p: rep_to_id[c] for p, c in canon.items()}
    return reps, pattern_to_id


GRAPHLET4_REPS, GRAPHLET4_CLASS = build_graphlet4_classes()


def graphlet4_pattern_from_vertices(adj_bits, a, b, c, d) -> int:
    v = (a, b, c, d)
    p = 0
    for bit, (i, j) in enumerate(EDGE4):
        if (adj_bits[v[i]] >> v[j]) & 1:
            p |= 1 << bit
    return p


def adjacency_bits(g: nx.Graph):
    n = g.number_of_nodes()
    bits = [0] * n
    for u, v in g.edges():
        bits[u] |= 1 << v
        bits[v] |= 1 << u
    return bits


def graphlet4_counts(g: nx.Graph) -> np.ndarray:
    n = g.number_of_nodes()
    bits = adjacency_bits(g)
    counts = np.zeros(11, dtype=np.int64)
    for a, b, c, d in itertools.combinations(range(n), 4):
        p = graphlet4_pattern_from_vertices(bits, a, b, c, d)
        counts[GRAPHLET4_CLASS[p]] += 1
    return counts


def gf2_rank_rows(rows) -> int:
    basis = {}
    rank = 0
    for x in rows:
        x = int(x)
        while x:
            pivot = x.bit_length() - 1
            if pivot in basis:
                x ^= basis[pivot]
            else:
                basis[pivot] = x
                rank += 1
                break
    return rank


def graphlet4_class_metadata() -> pd.DataFrame:
    rows = []
    for cid, p in enumerate(GRAPHLET4_REPS):
        A = np.zeros((4, 4), dtype=np.int8)
        for bit, (i, j) in enumerate(EDGE4):
            if (p >> bit) & 1:
                A[i, j] = A[j, i] = 1
        row_bits = []
        for i in range(4):
            r = 0
            for j in range(4):
                if A[i, j]:
                    r |= 1 << j
            row_bits.append(r)
        rank = gf2_rank_rows(row_bits)
        g = nx.from_numpy_array(A)
        rows.append({
            "graphlet4_id": cid,
            "canonical_6bit_pattern": p,
            "edges": int(g.number_of_edges()),
            "degree_sequence": "-".join(map(str, sorted(d for _, d in g.degree()))),
            "triangles": int(sum(nx.triangles(g).values()) // 3),
            "components": nx.number_connected_components(g),
            "gf2_rank": rank,
            "gf2_nullity": 4 - rank,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# KnotState, compatible in spirit with Stage 2A
# -----------------------------------------------------------------------------

def gf2_rank_mask(adj_bits, mask: int) -> int:
    basis = {}
    rank = 0
    mm = mask
    while mm:
        lsb = mm & -mm
        i = lsb.bit_length() - 1
        x = adj_bits[i] & mask
        while x:
            p = x.bit_length() - 1
            if p in basis:
                x ^= basis[p]
            else:
                basis[p] = x
                rank += 1
                break
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
        counts[k - gf2_rank_mask(adj_bits, mask)] += 1
    return counts


def nullity_counts_sampled(adj_bits, n: int, k: int, samples: int, seed: int) -> np.ndarray:
    counts = np.zeros(k + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        inds = rng.choice(n, size=k, replace=False)
        mask = combo_mask(inds)
        counts[k - gf2_rank_mask(adj_bits, mask)] += 1
    return counts


def graph_seed(g: nx.Graph) -> int:
    payload = f"{g.number_of_nodes()}|{g.number_of_edges()}|{tuple(sorted(d for _, d in g.degree()))}"
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def compute_knot_signature(g, max_k, samples, repeats, exact_limit, seed):
    n = g.number_of_nodes()
    bits = adjacency_bits(g)
    out = {}
    for k in range(2, min(max_k, n) + 1):
        total = math.comb(n, k)
        if total <= exact_limit:
            c = nullity_counts_exact(bits, n, k)
            out[k] = {"exact": True, "counts": c, "total": int(c.sum()),
                      "probs": c.astype(float) / max(int(c.sum()), 1)}
        else:
            c = np.zeros(k + 1, dtype=np.int64)
            gs = graph_seed(g)
            for r in range(repeats):
                rs = (seed + 1000003 * k + 9176 * r + gs) % (2**32 - 1)
                c += nullity_counts_sampled(bits, n, k, samples, rs)
            out[k] = {"exact": False, "counts": c, "total": int(c.sum()),
                      "probs": c.astype(float) / max(int(c.sum()), 1)}
    return out


def compare_knot(sig_g, sig_h, z_threshold, effect_min):
    first_k = None
    evidence = None
    max_tv = 0.0
    max_diff = 0.0
    for k in sorted(set(sig_g) & set(sig_h)):
        a, b = sig_g[k], sig_h[k]
        pa, pb = a["probs"], b["probs"]
        L = max(len(pa), len(pb))
        pa = np.pad(pa, (0, L - len(pa)))
        pb = np.pad(pb, (0, L - len(pb)))
        diff = np.abs(pa - pb)
        max_diff = max(max_diff, float(diff.max(initial=0.0)))
        max_tv = max(max_tv, 0.5 * float(diff.sum()))

        if a["exact"] and b["exact"]:
            separated = bool(np.any(a["counts"] != b["counts"]))
            ev = "exact"
        else:
            na, nb = max(a["total"], 1), max(b["total"], 1)
            se = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb + 1e-18)
            z = float(np.max(diff / se, initial=0.0))
            separated = bool(z >= z_threshold and float(diff.max(initial=0.0)) >= effect_min)
            ev = "sampled"
        if separated and first_k is None:
            first_k = k
            evidence = ev
    return {
        "knot_separated": first_k is not None,
        "knot_first_k": first_k,
        "knot_evidence": evidence,
        "knot_max_tv": max_tv,
        "knot_max_abs_diff": max_diff,
    }


def knot_cache_key(g: nx.Graph, args) -> str:
    g6 = nx.to_graph6_bytes(g, header=False).strip()
    h = hashlib.sha256(g6).hexdigest()
    return f"{h}|k{args.max_k}|s{args.samples}|r{args.repeats}|e{args.exact_limit}|seed{args.seed}"


def load_pickle(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def save_pickle(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_prior_knot_csv(path: Path, pids):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        need = set(pids)
        have = set(int(x) for x in df["pair_id"].tolist())
        if not need.issubset(have):
            return None
        keep = df[df.pair_id.isin(sorted(need))].copy()
        cols = ["pair_id", "knot_separated", "knot_first_k", "knot_evidence",
                "knot_max_tv", "knot_max_abs_diff"]
        return keep[cols].set_index("pair_id").to_dict(orient="index")
    except Exception:
        return None


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    def block(name, sub):
        n = len(sub)
        return {
            "group": name,
            "pairs": n,
            "1wl_sep": int(sub.one_wl_separated.sum()),
            "2wl_sep": int(sub.two_wl_separated.sum()),
            "adj_spectrum_sep": int(sub.adj_spectrum_separated.sum()),
            "lap_spectrum_sep": int(sub.lap_spectrum_separated.sum()),
            "basic_sep": int(sub.basic_stats_separated.sum()),
            "graphlet4_sep": int(sub.graphlet4_separated.sum()),
            "knot_sep": int(sub.knot_separated.sum()),
            "knot_beyond_2wl": int((sub.knot_separated & ~sub.two_wl_separated).sum()),
            "knot_beyond_graphlet4": int((sub.knot_separated & ~sub.graphlet4_separated).sum()),
            "knot_beyond_2wl_and_graphlet4": int(
                (sub.knot_separated & ~sub.two_wl_separated & ~sub.graphlet4_separated).sum()
            ),
            "deep_wins": int(sub.deep_win.sum()),
            "deep_win_rate": float(sub.deep_win.mean()) if n else np.nan,
        }

    rows = [block("Regular_ALL", df)]
    for grp in ["SimpleRegular", "StronglyRegular"]:
        sub = df[df.subgroup == grp]
        if len(sub):
            rows.append(block(grp, sub))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/content/brec_stage2B")
    ap.add_argument("--zip", default=None, help="optional /content/BREC_data_all.zip")
    ap.add_argument("--start", type=int, default=60)
    ap.add_argument("--end", type=int, default=160)
    ap.add_argument("--prior-knot-csv", default="/content/brec_knotstate/pair_results.csv")
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--exact-limit", type=int, default=200000)
    ap.add_argument("--z-threshold", type=float, default=5.0)
    ap.add_argument("--effect-min", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if not (0 <= args.start < args.end <= PAIR_NUM):
        raise ValueError("Require 0 <= start < end <= 400")

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(vars(args), indent=2))

    npy = download_brec(root, args.zip)
    print(f"Using BREC file: {npy}", flush=True)
    pairs = load_pairs(npy, args.start, args.end)
    pids = [pid for pid, _, _ in pairs]
    print(f"Loaded {len(pairs)} BREC pairs: ids {args.start}..{args.end-1}", flush=True)

    # Save the 11 graphlet classes and their GF(2) nullities.
    graphlet4_class_metadata().to_csv(root / "graphlet4_class_map.csv", index=False)

    prior_knot = load_prior_knot_csv(Path(args.prior_knot_csv), pids)
    if prior_knot is not None:
        print(f"Reusing KnotState results from {args.prior_knot_csv}", flush=True)
    else:
        print("No complete prior KnotState CSV found; KnotState will be recomputed.", flush=True)

    knot_cache_path = root / "knot_cache.pkl"
    knot_cache = load_pickle(knot_cache_path)
    rows = []
    t0 = time.time()

    for idx, (pid, g, h) in enumerate(pairs, start=1):
        one_sep = one_wl_pair_separated(g, h)
        two_sep = two_wl_pair_separated(g, h)
        adj_sep = spectra_different(adjacency_spectrum(g), adjacency_spectrum(h))
        lap_sep = spectra_different(norm_laplacian_spectrum(g), norm_laplacian_spectrum(h))
        basic_sep = basic_signature(g) != basic_signature(h)

        g4g = graphlet4_counts(g)
        g4h = graphlet4_counts(h)
        graphlet4_sep = bool(np.any(g4g != g4h))
        graphlet4_l1 = int(np.abs(g4g - g4h).sum())

        if prior_knot is not None:
            kc = prior_knot[pid]
        else:
            kg = knot_cache_key(g, args)
            kh = knot_cache_key(h, args)
            if kg not in knot_cache:
                knot_cache[kg] = compute_knot_signature(
                    g, args.max_k, args.samples, args.repeats, args.exact_limit, args.seed
                )
            if kh not in knot_cache:
                knot_cache[kh] = compute_knot_signature(
                    h, args.max_k, args.samples, args.repeats, args.exact_limit, args.seed
                )
            kc = compare_knot(
                knot_cache[kg], knot_cache[kh], args.z_threshold, args.effect_min
            )

        knot_sep = bool(kc["knot_separated"])
        knot_first_k = None if pd.isna(kc["knot_first_k"]) else int(kc["knot_first_k"])
        knot_evidence = kc["knot_evidence"]

        # Mathematical consistency audit.
        if knot_sep and knot_first_k == 4 and str(knot_evidence) == "exact" and not graphlet4_sep:
            warnings.warn(
                f"Pair {pid}: exact k=4 KnotState separates but full 4-graphlet counts do not. "
                "This should be impossible; check cached data / implementation."
            )

        deep_win = bool(
            knot_sep
            and not one_sep
            and not two_sep
            and not adj_sep
            and not lap_sep
            and not basic_sep
            and not graphlet4_sep
        )

        row = {
            "pair_id": pid,
            "subgroup": subgroup_of(pid),
            "n_g": g.number_of_nodes(),
            "n_h": h.number_of_nodes(),
            "m_g": g.number_of_edges(),
            "m_h": h.number_of_edges(),
            "one_wl_separated": one_sep,
            "two_wl_separated": two_sep,
            "adj_spectrum_separated": adj_sep,
            "lap_spectrum_separated": lap_sep,
            "basic_stats_separated": basic_sep,
            "graphlet4_separated": graphlet4_sep,
            "graphlet4_l1_count_diff": graphlet4_l1,
            "knot_separated": knot_sep,
            "knot_first_k": knot_first_k,
            "knot_evidence": knot_evidence,
            "knot_max_tv": float(kc.get("knot_max_tv", np.nan)),
            "knot_max_abs_diff": float(kc.get("knot_max_abs_diff", np.nan)),
            "knot_beyond_2wl": bool(knot_sep and not two_sep),
            "knot_beyond_graphlet4": bool(knot_sep and not graphlet4_sep),
            "knot_beyond_2wl_and_graphlet4": bool(knot_sep and not two_sep and not graphlet4_sep),
            "deep_win": deep_win,
        }
        rows.append(row)

        if idx % 10 == 0 or idx == len(pairs):
            df_now = pd.DataFrame(rows)
            df_now.to_csv(root / "pair_results.csv", index=False)
            if prior_knot is None:
                save_pickle(knot_cache_path, knot_cache)
            print(
                f"{idx:3d}/{len(pairs)} | 2WL_sep={int(df_now.two_wl_separated.sum())} "
                f"| g4_sep={int(df_now.graphlet4_separated.sum())} "
                f"| knot_sep={int(df_now.knot_separated.sum())} "
                f"| deep_wins={int(df_now.deep_win.sum())} "
                f"| elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(root / "pair_results.csv", index=False)
    summary = make_summary(df)
    summary.to_csv(root / "summary.csv", index=False)

    knot_beyond_g4 = df[df.knot_beyond_graphlet4].copy()
    knot_beyond_g4.to_csv(root / "knot_beyond_graphlet4.csv", index=False)

    knot_beyond_2 = df[df.knot_beyond_2wl].copy()
    knot_beyond_2.to_csv(root / "knot_beyond_2wl.csv", index=False)

    deep = df[df.deep_win].copy()
    deep.to_csv(root / "deep_wins.csv", index=False)

    # Extra audit: how many exact k=4 Knot wins are also graphlet-separated?
    exact_k4 = df[
        df.knot_separated
        & (df.knot_first_k == 4)
        & (df.knot_evidence.astype(str) == "exact")
    ]
    audit_ok = int(exact_k4.graphlet4_separated.sum())

    print("\n=== STAGE 2B SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== KEY COUNTS ===")
    print("Regular pairs:", len(df))
    print("2-WL separated:", int(df.two_wl_separated.sum()), "/", len(df))
    print("Full induced 4-graphlet separated:", int(df.graphlet4_separated.sum()), "/", len(df))
    print("KnotState separated:", int(df.knot_separated.sum()), "/", len(df))
    print("Knot beyond 2-WL:", int(df.knot_beyond_2wl.sum()), "/", len(df))
    print("Knot beyond full 4-graphlets:", int(df.knot_beyond_graphlet4.sum()), "/", len(df))
    print(
        "Knot beyond BOTH 2-WL and full 4-graphlets:",
        int(df.knot_beyond_2wl_and_graphlet4.sum()), "/", len(df)
    )
    print("Deep wins (also spectra/basic fail):", int(df.deep_win.sum()), "/", len(df))
    print(
        f"Audit exact k=4 Knot separations explained by full graphlets: "
        f"{audit_ok}/{len(exact_k4)}"
    )

    if len(knot_beyond_g4):
        print("\nFirst KnotState wins beyond full 4-graphlets:")
        print(knot_beyond_g4.head(20)[[
            "pair_id", "subgroup", "n_g", "m_g", "two_wl_separated",
            "knot_first_k", "knot_evidence", "knot_max_tv", "deep_win"
        ]].to_string(index=False))
    else:
        print("\nNo KnotState wins beyond full 4-node graphlets in this scan.")

    if len(deep):
        print("\nFirst DEEP WINS:")
        print(deep.head(20)[[
            "pair_id", "subgroup", "n_g", "m_g", "knot_first_k",
            "knot_evidence", "knot_max_tv"
        ]].to_string(index=False))

    report = [
        "# BREC Stage 2B: Graphlets + 2-WL vs KnotState",
        "",
        f"Pairs: {len(df)} (default Regular ids 60..159)",
        f"2-WL separated: {int(df.two_wl_separated.sum())}/{len(df)}",
        f"Full induced 4-graphlet separated: {int(df.graphlet4_separated.sum())}/{len(df)}",
        f"KnotState separated: {int(df.knot_separated.sum())}/{len(df)}",
        f"Knot beyond 2-WL: {int(df.knot_beyond_2wl.sum())}/{len(df)}",
        f"Knot beyond full 4-graphlets: {int(df.knot_beyond_graphlet4.sum())}/{len(df)}",
        f"Knot beyond both 2-WL and full 4-graphlets: {int(df.knot_beyond_2wl_and_graphlet4.sum())}/{len(df)}",
        f"Deep wins: {int(df.deep_win.sum())}/{len(df)}",
        "",
        "Audit fact: exact KnotState separation first appearing at k=4 must imply a full induced 4-graphlet separation, because the k=4 nullity histogram is a coarsening of the 11 graphlet counts by GF(2) adjacency nullity.",
    ]
    (root / "report.md").write_text("\n".join(report))

    print(f"\nFiles saved under: {root}")
    print("NOTE: This is an expressivity descriptor scan, not official BREC RPC neural accuracy.")


if __name__ == "__main__":
    main()
