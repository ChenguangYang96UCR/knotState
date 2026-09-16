#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""KnotState-SG sweep for SubGNN/WLKS-style subgraph benchmarks.

AutoDL-oriented script. It:
- loads SubGNN-format data (edge_list.txt + subgraphs.pth)
- optionally downloads the four real-world SubGNN datasets
- optionally generates Density / Component synthetic datasets
- computes 75D target-conditioned KnotState with bitwise GF(2) rank
- sweeps radius, sampling budget, state order and Internal/Boundary/Context channels
- evaluates KnotState, simple structural features, and simple+KnotState
- supports single-label and multi-label tasks
- caches features and appends results so reruns resume safely
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import multiprocessing as mp
import os
import random
import re
import time
import warnings
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import SVC

REAL_DATASETS = ["ppi_bp", "hpo_neuro", "hpo_metab", "em_user"]
ALIASES = {
    "ppi_bp": ["ppi_bp", "ppi-bp", "ppibp"],
    "hpo_neuro": ["hpo_neuro", "hpo-neuro", "hponeuro"],
    "hpo_metab": ["hpo_metab", "hpo-metab", "hpometab"],
    "em_user": ["em_user", "em-user", "emuser"],
    "density": ["density"],
    "component": ["component", "components", "cc"],
    "coreness": ["coreness", "core"],
    "cut_ratio": ["cut_ratio", "cutratio", "cut-ratio"],
}
SUBGNN_DROPBOX_URL = "https://www.dropbox.com/sh/zv7gw2bqzqev9yn/AACR9iR4Ok7f9x1fIAiVCdj3a?dl=1"
PUBLISHED_WLKS = {
    "ppi_bp": 0.648, "hpo_neuro": 0.653, "hpo_metab": 0.579,
    "em_user": 0.918, "density": 0.960, "cut_ratio": 0.600,
    "coreness": 0.913, "component": 1.000,
}


def log(x):
    print(x, flush=True)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def norm_name(s: str):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def str_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


# -------------------- KnotState core --------------------

def allowed_nullities(s: int):
    return list(range(s % 2, s + 1, 2))


KNOT_KEYS = [
    (s, a, nu)
    for s in range(2, 7)
    for a in range(s + 1)
    for nu in allowed_nullities(s)
]
assert len(KNOT_KEYS) == 75


def build_adj_sets(G: nx.Graph):
    return {int(u): {int(v) for v in G.neighbors(u)} for u in G.nodes()}


def gf2_rank_bits(rows: Sequence[int]):
    basis = {}
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


def gf2_nullity_fast(adj, nodes):
    nodes = list(nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    rows = []
    for u in nodes:
        bits = 0
        for v in adj.get(u, ()):
            j = idx.get(v)
            if j is not None:
                bits |= 1 << j
        rows.append(bits)
    return len(nodes) - gf2_rank_bits(rows)


def r_hop(adj, H: Set[int], radius: int):
    seen = set(H)
    front = set(H)
    for _ in range(radius):
        nxt = set()
        for u in front:
            nxt.update(adj.get(u, ()))
        nxt -= seen
        seen |= nxt
        front = nxt
        if not front:
            break
    return seen


def knotstate_one(adj, H_nodes, radius=1, max_exact=2000, samples=500, seed=42):
    rng = random.Random(seed)
    H = {int(x) for x in H_nodes}
    outside = r_hop(adj, H, radius) - H
    Hin, Hout = list(H), list(outside)
    feat = {k: 0.0 for k in KNOT_KEYS}

    for s in range(2, 7):
        for a in range(s + 1):
            b = s - a
            if a > len(Hin) or b > len(Hout):
                continue
            total = math.comb(len(Hin), a) * math.comb(len(Hout), b)
            if total == 0:
                continue
            counts = {nu: 0 for nu in allowed_nullities(s)}
            n_eval = 0
            if total <= max_exact:
                for I in itertools.combinations(Hin, a):
                    for O in itertools.combinations(Hout, b):
                        nu = gf2_nullity_fast(adj, I + O)
                        counts[nu] += 1
                        n_eval += 1
            else:
                for _ in range(samples):
                    I = rng.sample(Hin, a) if a else []
                    O = rng.sample(Hout, b) if b else []
                    nu = gf2_nullity_fast(adj, I + O)
                    counts[nu] += 1
                    n_eval += 1
            if n_eval:
                for nu in counts:
                    feat[(s, a, nu)] = counts[nu] / n_eval
    return np.asarray([feat[k] for k in KNOT_KEYS], dtype=np.float32)


_W_ADJ = None
_W_SUB = None
_W_R = None
_W_EXACT = None
_W_SAMPLES = None
_W_SEED = None


def _worker_init(adj, subgraphs, radius, max_exact, samples, feature_seed):
    global _W_ADJ, _W_SUB, _W_R, _W_EXACT, _W_SAMPLES, _W_SEED
    _W_ADJ, _W_SUB = adj, subgraphs
    _W_R, _W_EXACT, _W_SAMPLES, _W_SEED = radius, max_exact, samples, feature_seed


def _worker(i):
    x = knotstate_one(
        _W_ADJ, _W_SUB[i], _W_R, _W_EXACT, _W_SAMPLES,
        _W_SEED * 1000003 + i,
    )
    return i, x


def compute_knot_matrix(adj, subgraphs, radius, max_exact, samples, feature_seed, jobs):
    n = len(subgraphs)
    X = np.zeros((n, 75), dtype=np.float32)
    if jobs <= 1:
        for i in range(n):
            X[i] = knotstate_one(adj, subgraphs[i], radius, max_exact, samples, feature_seed * 1000003 + i)
            if (i + 1) % 100 == 0 or i + 1 == n:
                log(f"    {i+1}/{n}")
        return X
    ctx = mp.get_context("fork")
    chunksize = max(1, n // (jobs * 8))
    with ctx.Pool(jobs, initializer=_worker_init,
                  initargs=(adj, subgraphs, radius, max_exact, samples, feature_seed)) as pool:
        done = 0
        for i, x in pool.imap_unordered(_worker, range(n), chunksize=chunksize):
            X[i] = x
            done += 1
            if done % 100 == 0 or done == n:
                log(f"    {done}/{n}")
    return X


# -------------------- data --------------------

def read_edges(path: Path):
    G = nx.Graph()
    with path.open() as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            try:
                u, v = int(p[0]), int(p[1])
            except ValueError:
                continue
            if u == v:
                G.add_node(u)
            else:
                G.add_edge(u, v)
    return G


def read_subgraphs(path: Path):
    subgraphs, raw_labels, splits = [], [], []
    multilabel = False
    with path.open() as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            nodes = [int(x) for x in p[0].split("-") if x]
            labs = [x for x in p[1].split("-") if x]
            split = p[2].strip().lower()
            if not nodes or not labs or split not in {"train", "val", "test"}:
                continue
            multilabel |= len(labs) > 1
            subgraphs.append(nodes)
            raw_labels.append(labs)
            splits.append(split)
    mapping = {}
    for labs in raw_labels:
        for lab in labs:
            if lab not in mapping:
                mapping[lab] = len(mapping)
    encoded = [[mapping[x] for x in labs] for labs in raw_labels]
    if multilabel:
        mlb = MultiLabelBinarizer(classes=list(range(len(mapping))))
        Y = mlb.fit_transform(encoded).astype(np.int8)
    else:
        Y = np.asarray([x[0] for x in encoded], dtype=np.int64)
    return subgraphs, Y, np.asarray(splits), multilabel, mapping


def locate_dataset(data_root: Path, name: str):
    aliases = [norm_name(x) for x in ALIASES[name]]
    candidates = []
    if data_root.exists():
        for p in data_root.rglob("*"):
            if p.is_dir() and (p / "edge_list.txt").exists() and (p / "subgraphs.pth").exists():
                candidates.append(p)
    exact = [p for p in candidates if norm_name(p.name) in aliases]
    if exact:
        return sorted(exact, key=lambda p: len(str(p)))[0]
    partial = [p for p in candidates if any(a in norm_name(str(p)) for a in aliases)]
    return sorted(partial, key=lambda p: len(str(p)))[0] if partial else None


def download_real(data_root: Path):
    import requests
    ensure_dir(data_root)
    if all(locate_dataset(data_root, d) for d in REAL_DATASETS):
        log("Real-world datasets already present.")
        return
    zpath = data_root / "SubGNN_real_world.zip"
    log("Downloading public SubGNN real-world bundle...")
    with requests.get(SUBGNN_DROPBOX_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with zpath.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    if not zipfile.is_zipfile(zpath):
        raise RuntimeError("Dropbox response was not a ZIP. Download the SubGNN dataset bundle manually and extract under --data-root.")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(data_root)
    log("Download/extraction done.")


# -------------------- lightweight synthetic generators --------------------

def quantile_labels(values, n_bins=3):
    values = np.asarray(values)
    sv = sorted(values.tolist())
    pos = (len(sv) / float(n_bins)) * np.arange(1, n_bins + 1)
    bins = np.asarray([sv[int(p) - 1] for p in pos])
    bins = np.unique(bins)[:-1]
    return np.digitize(values, bins=bins)


def make_split(n, seed=42):
    rng = random.Random(seed)
    idx = list(range(n))
    tr = set(rng.sample(idx, int(n * 0.8)))
    rest = [i for i in idx if i not in tr]
    va = set(rng.sample(rest, len(rest) // 2))
    return ["train" if i in tr else ("val" if i in va else "test") for i in idx]


def write_dataset(out, G, subs, labels, splits):
    ensure_dir(out)
    nx.write_edgelist(G, out / "edge_list.txt", data=False)
    with (out / "subgraphs.pth").open("w") as f:
        for H, y, sp in zip(subs, labels, splits):
            f.write(f"{'-'.join(map(str, H))}\t{y}\t{sp}\t\n")


def bfs_target(G, start, size=20, depth=3):
    nodes = [start] + [v for _, v in nx.bfs_edges(G, start, depth_limit=depth)]
    if len(nodes) < size:
        seen = set(nodes)
        for v in G.nodes():
            if v not in seen:
                nodes.append(v)
                if len(nodes) >= size:
                    break
    return nodes[:size]


def generate_density(out: Path, seed=42):
    log("Generating Density...")
    rng = random.Random(seed)
    G = nx.barabasi_albert_graph(5000, 5, seed=seed)
    subs = [bfs_target(G, rng.choice(list(G.nodes()))) for _ in range(250)]
    for H in subs:
        desired = rng.choice([0.05, 0.25, 0.45])
        for _ in range(100):
            cur = nx.density(G.subgraph(H))
            if abs(cur - desired) < 0.01:
                break
            if cur > desired:
                edges = list(G.subgraph(H).edges())
                if not edges:
                    break
                G.remove_edge(*rng.choice(edges))
            else:
                G.add_edge(*rng.sample(H, 2))
    values = [nx.density(G.subgraph(H)) for H in subs]
    y = quantile_labels(values, 3)
    write_dataset(out, G, subs, y.tolist(), make_split(len(subs), seed))


def generate_component(out: Path, seed=42):
    log("Generating Component...")
    rng = random.Random(seed)
    G = nx.barabasi_albert_graph(1000, 5, seed=seed)
    subs, labels = [], []
    cc_range = [1, 1, 1, 1, 5, 6, 7, 8, 9, 10]
    for i in range(250):
        ncc = rng.choice(cc_range)
        H = []
        for c in range(ncc):
            local = nx.extended_barabasi_albert_graph(15, 5, rng.choice([0.1, 0.5, 0.9]), 0.0, seed=seed + i * 31 + c)
            offset = G.number_of_nodes()
            local = nx.relabel_nodes(local, {u: offset + j for j, u in enumerate(local.nodes())})
            base_root = rng.choice(list(G.nodes()))
            comp_root = rng.choice(list(local.nodes()))
            G = nx.compose(G, local)
            G.add_edge(base_root, comp_root)
            H.extend(local.nodes())
        subs.append(list(H))
        cc = nx.number_connected_components(G.subgraph(H))
        labels.append(0 if cc < 5 else 1)
    write_dataset(out, G, subs, labels, make_split(len(subs), seed))


def ensure_synthetic(root: Path, name: str):
    out = root / name
    if (out / "edge_list.txt").exists() and (out / "subgraphs.pth").exists():
        return out
    if name == "density":
        generate_density(out)
    elif name == "component":
        generate_component(out)
    else:
        raise RuntimeError(f"No auto-generator for {name}")
    return out


# -------------------- simple baseline --------------------

def simple_features(G, subgraphs):
    rows = []
    N = G.number_of_nodes()
    for H0 in subgraphs:
        H = set(H0)
        SG = G.subgraph(H)
        n = len(H)
        m = SG.number_of_edges()
        gdeg = np.asarray([G.degree(v) for v in H], float)
        ideg = np.asarray([SG.degree(v) for v in H], float)
        boundary = sum(1 for u in H for v in G.neighbors(u) if v not in H)
        rows.append([
            n, m, nx.density(SG) if n > 1 else 0.0,
            boundary, boundary / (n * max(1, N - n)),
            gdeg.mean() if len(gdeg) else 0.0,
            gdeg.std() if len(gdeg) else 0.0,
            ideg.mean() if len(ideg) else 0.0,
            ideg.std() if len(ideg) else 0.0,
            nx.number_connected_components(SG) if n else 0,
            nx.average_clustering(SG) if n > 1 else 0.0,
        ])
    return np.asarray(rows, dtype=np.float32)


def select_cols(channel, min_s, max_s):
    cols = []
    for j, (s, a, nu) in enumerate(KNOT_KEYS):
        if not (min_s <= s <= max_s):
            continue
        ok = (
            (channel == "internal" and a == s) or
            (channel == "boundary" and 0 < a < s) or
            (channel == "context" and a == 0) or
            (channel == "internal_boundary" and a > 0) or
            (channel == "full")
        )
        if ok:
            cols.append(j)
    return cols


# -------------------- classifier --------------------

def split_idx(splits):
    return tuple(np.where(splits == x)[0] for x in ["train", "val", "test"])


def clf_grid(profile):
    Cs = [0.1, 1, 10] if profile == "quick" else [0.01, 0.1, 1, 10, 100]
    gammas = ["scale"] if profile != "full" else ["scale", 0.01, 0.1, 1.0]
    for C in Cs:
        yield "linear", C, "na"
    for C in Cs:
        for g in gammas:
            yield "rbf", C, g


def make_model(multilabel, kernel, C, gamma):
    kw = {"C": C, "kernel": kernel}
    if kernel == "rbf":
        kw["gamma"] = gamma
    base = SVC(**kw)
    est = OneVsRestClassifier(base, n_jobs=1) if multilabel else base
    return make_pipeline(StandardScaler(), est)


def metrics(y, p):
    return (
        float(f1_score(y, p, average="micro", zero_division=0)),
        float(f1_score(y, p, average="macro", zero_division=0)),
        float(accuracy_score(y, p)),
    )


def tune_test(X, Y, splits, multilabel, profile):
    tr, va, te = split_idx(splits)
    best = None
    for kernel, C, gamma in clf_grid(profile):
        try:
            model = make_model(multilabel, kernel, C, gamma)
            model.fit(X[tr], Y[tr])
            vmicro, vmacro, vacc = metrics(Y[va], model.predict(X[va]))
        except Exception as e:
            warnings.warn(str(e))
            continue
        cur = (vmicro, vmacro)
        if best is None or cur > best[0]:
            best = (cur, model, kernel, C, gamma, vmicro, vmacro, vacc)
    if best is None:
        raise RuntimeError("all classifier configurations failed")
    _, model, kernel, C, gamma, vmicro, vmacro, vacc = best
    tmicro, tmacro, tacc = metrics(Y[te], model.predict(X[te]))
    return dict(kernel=kernel, C=C, gamma=gamma,
                val_micro_f1=vmicro, val_macro_f1=vmacro, val_accuracy=vacc,
                test_micro_f1=tmicro, test_macro_f1=tmacro, test_accuracy=tacc)


# -------------------- persistence --------------------

COLUMNS = [
    "dataset", "n_graph_nodes", "n_graph_edges", "n_subgraphs", "multilabel", "n_classes",
    "train_n", "val_n", "test_n", "radius", "samples", "feature_seed", "max_exact",
    "representation", "channel", "min_s", "max_s", "n_features", "kernel", "C", "gamma",
    "val_micro_f1", "val_macro_f1", "val_accuracy", "test_micro_f1", "test_macro_f1", "test_accuracy",
    "feature_seconds", "published_wlks_micro_f1",
]


def key_from_row(r):
    return (str(r["dataset"]), int(r["radius"]), int(r["samples"]), int(r["feature_seed"]),
            int(r["max_exact"]), str(r["representation"]), str(r["channel"]), int(r["min_s"]), int(r["max_s"]))


def load_done(path):
    if not path.exists():
        return set()
    return {key_from_row(r) for _, r in pd.read_csv(path).iterrows()}


def append_row(path, row):
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def summarize(raw, out):
    if not raw.exists():
        return
    df = pd.read_csv(raw)
    if df.empty:
        return
    rows = []
    for _, g in df.groupby(["dataset", "representation"], dropna=False):
        rows.append(g.sort_values(["val_micro_f1", "val_macro_f1"], ascending=False).iloc[0])
    pd.DataFrame(rows).to_csv(out / "best_by_dataset_representation.csv", index=False)
    knot = df[df.representation == "knot"]
    if not knot.empty:
        rows = []
        for _, g in knot.groupby(["dataset", "channel"]):
            rows.append(g.sort_values(["val_micro_f1", "val_macro_f1"], ascending=False).iloc[0])
        pd.DataFrame(rows).to_csv(out / "best_knot_by_channel.csv", index=False)
    top = df.sort_values(["dataset", "test_micro_f1"], ascending=[True, False]).groupby("dataset").head(20)
    top.to_csv(out / "top20_for_analysis_ONLY.csv", index=False)


def profile_cfg(profile):
    if profile == "quick":
        return dict(radii=[1], samples=[200, 500], seeds=[0],
                    channels=["internal", "boundary", "full"], orders=[(2, 6), (3, 6)])
    if profile == "balanced":
        return dict(radii=[1, 2], samples=[500, 1000], seeds=[0],
                    channels=["internal", "boundary", "context", "internal_boundary", "full"],
                    orders=[(2, 4), (3, 4), (2, 5), (3, 5), (2, 6), (3, 6)])
    return dict(radii=[1, 2], samples=[200, 500, 1000], seeds=[0, 1],
                channels=["internal", "boundary", "context", "internal_boundary", "full"],
                orders=[(2, 4), (3, 4), (2, 5), (3, 5), (2, 6), (3, 6), (4, 6), (5, 6)])


def run_dataset(name, ddir, out, args, cfg, raw, done):
    log("\n" + "=" * 72 + f"\n{name}  {ddir}\n" + "=" * 72)
    G = read_edges(ddir / "edge_list.txt")
    subs, Y, splits, multilabel, mapping = read_subgraphs(ddir / "subgraphs.pth")
    for H in subs:
        G.add_nodes_from(H)
    tr, va, te = split_idx(splits)
    nclasses = Y.shape[1] if multilabel else len(np.unique(Y))
    log(f"graph={G.number_of_nodes()}/{G.number_of_edges()} targets={len(subs)} split={len(tr)}/{len(va)}/{len(te)} classes={nclasses} multilabel={multilabel}")

    dout = ensure_dir(out / name)
    fdir = ensure_dir(dout / "features")
    simple_path = dout / "simple_features.npy"
    Xs = np.load(simple_path) if simple_path.exists() and not args.force_features else simple_features(G, subs)
    np.save(simple_path, Xs)

    skey = (name, 0, 0, 0, args.max_exact, "simple", "simple", 0, 0)
    if skey not in done or args.force:
        r = tune_test(Xs, Y, splits, multilabel, args.profile)
        row = dict(dataset=name, n_graph_nodes=G.number_of_nodes(), n_graph_edges=G.number_of_edges(),
                   n_subgraphs=len(subs), multilabel=multilabel, n_classes=nclasses,
                   train_n=len(tr), val_n=len(va), test_n=len(te), radius=0, samples=0,
                   feature_seed=0, max_exact=args.max_exact, representation="simple", channel="simple",
                   min_s=0, max_s=0, n_features=Xs.shape[1], feature_seconds=0.0,
                   published_wlks_micro_f1=PUBLISHED_WLKS.get(name, np.nan), **r)
        append_row(raw, row); done.add(skey)
        log(f"simple test={r['test_micro_f1']:.4f}")

    adj = build_adj_sets(G)
    del G

    for radius in cfg["radii"]:
        for samples in cfg["samples"]:
            for fseed in cfg["seeds"]:
                cache = fdir / f"knot_r{radius}_m{samples}_seed{fseed}_exact{args.max_exact}.npy"
                t0 = time.time()
                if cache.exists() and not args.force_features:
                    Xk = np.load(cache); sec = 0.0
                    log(f"cache {cache.name}")
                else:
                    log(f"features r={radius} samples={samples} seed={fseed}")
                    Xk = compute_knot_matrix(adj, subs, radius, args.max_exact, samples, fseed, args.jobs)
                    sec = time.time() - t0
                    np.save(cache, Xk)
                    log(f"  saved {Xk.shape} in {sec:.1f}s")

                for channel in cfg["channels"]:
                    for mins, maxs in cfg["orders"]:
                        cols = select_cols(channel, mins, maxs)
                        if not cols:
                            continue
                        for rep in ["knot", "simple+knot"]:
                            key = (name, radius, samples, fseed, args.max_exact, rep, channel, mins, maxs)
                            if key in done and not args.force:
                                continue
                            X = Xk[:, cols] if rep == "knot" else np.concatenate([Xs, Xk[:, cols]], axis=1)
                            try:
                                r = tune_test(X, Y, splits, multilabel, args.profile)
                            except Exception as e:
                                log(f"FAILED {rep} {channel} {mins}-{maxs}: {e}")
                                continue
                            row = dict(dataset=name, n_graph_nodes=len(adj),
                                       n_graph_edges=sum(map(len, adj.values())) // 2,
                                       n_subgraphs=len(subs), multilabel=multilabel, n_classes=nclasses,
                                       train_n=len(tr), val_n=len(va), test_n=len(te), radius=radius,
                                       samples=samples, feature_seed=fseed, max_exact=args.max_exact,
                                       representation=rep, channel=channel, min_s=mins, max_s=maxs,
                                       n_features=X.shape[1], feature_seconds=sec,
                                       published_wlks_micro_f1=PUBLISHED_WLKS.get(name, np.nan), **r)
                            append_row(raw, row); done.add(key)
                            log(f"{rep:11s} {channel:17s} s={mins}-{maxs} val={r['val_micro_f1']:.4f} test={r['test_micro_f1']:.4f}")
                summarize(raw, out)


def self_test():
    G = nx.complete_bipartite_graph(2, 3)
    adj = build_adj_sets(G)
    assert gf2_nullity_fast(adj, list(G.nodes())) == 3
    x = knotstate_one(adj, [0, 1, 2], radius=1, samples=30, seed=1)
    assert x.shape == (75,)
    log("SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/SubGNN_data"))
    ap.add_argument("--out-root", type=Path, default=Path("/root/autodl-tmp/KnotStateSweep"))
    ap.add_argument("--datasets", default="ppi_bp,hpo_neuro,hpo_metab,em_user,density,component")
    ap.add_argument("--profile", choices=["quick", "balanced", "full"], default="balanced")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--max-exact", type=int, default=2000)
    ap.add_argument("--download-real", action="store_true")
    ap.add_argument("--generate-synthetic", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-features", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--radii", default="")
    ap.add_argument("--samples", default="")
    ap.add_argument("--feature-seeds", default="")
    args = ap.parse_args()

    if args.self_test:
        self_test(); return

    ensure_dir(args.data_root); ensure_dir(args.out_root)
    if args.download_real:
        try:
            download_real(args.data_root)
        except Exception as e:
            log(f"WARNING download failed: {e}")

    cfg = profile_cfg(args.profile)
    if args.radii: cfg["radii"] = int_list(args.radii)
    if args.samples: cfg["samples"] = int_list(args.samples)
    if args.feature_seeds: cfg["seeds"] = int_list(args.feature_seeds)

    pd.DataFrame([{"dataset": k, "published_wlks_micro_f1": v} for k, v in PUBLISHED_WLKS.items()]).to_csv(args.out_root / "published_wlks_reference.csv", index=False)
    raw = args.out_root / "raw_results.csv"
    done = load_done(raw)

    log(f"profile={args.profile} jobs={args.jobs} radii={cfg['radii']} samples={cfg['samples']} seeds={cfg['seeds']}")
    for name in str_list(args.datasets):
        if name not in ALIASES:
            log(f"SKIP unknown {name}"); continue
        ddir = locate_dataset(args.data_root, name)
        if ddir is None and name in {"density", "component"} and args.generate_synthetic:
            ddir = args.data_root / name
            if name == "density": generate_density(ddir)
            else: generate_component(ddir)
        if ddir is None:
            log(f"SKIP {name}: missing edge_list.txt + subgraphs.pth under {args.data_root}")
            continue
        try:
            run_dataset(name, ddir, args.out_root, args, cfg, raw, done)
        except Exception as e:
            import traceback
            log(f"ERROR {name}: {e}"); traceback.print_exc()
    summarize(raw, args.out_root)
    log(f"DONE. Results: {raw}")


if __name__ == "__main__":
    main()
