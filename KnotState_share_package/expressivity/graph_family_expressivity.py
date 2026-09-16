#!/usr/bin/env python3
"""Order-wise KnotState distinguishability for a graph family.

Input: graph6 strings, one graph per line. This is convenient for SR25 and
strongly-regular graph collections. Exact counts are used when feasible;
otherwise Monte Carlo counts are reported as approximate and SHOULD NOT be
presented as exact certificates.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.knotstate_core import allowed_nullities, global_knotstate_counts


def read_g6(path: Path):
    graphs = []
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(b"#"):
                continue
            graphs.append(nx.Graph(nx.from_graph6_bytes(line)))
    return graphs


def signature(counts, denoms, upto):
    sig = []
    for s in range(2, upto + 1):
        sig.append(("denom", s, int(denoms[s])))
        for nu in allowed_nullities(s):
            sig.append((s, nu, int(counts[(s, nu)])))
    return tuple(sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=Path, required=True, help="graph6 file; one graph per line")
    ap.add_argument("--max-order", type=int, default=6)
    ap.add_argument("--exact-cap", type=int, default=2_000_000)
    ap.add_argument("--samples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("graph_family_results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    graphs = read_g6(args.graphs)
    if not graphs:
        raise SystemExit("No graph6 graphs were read")

    records = []
    all_data = []
    for i, G in enumerate(graphs):
        c, d, ex = global_knotstate_counts(
            G, max_order=args.max_order, exact_cap=args.exact_cap,
            samples=args.samples, seed=args.seed + i * 1009,
        )
        all_data.append((c, d, ex))
        records.append({
            "graph_id": i, "n": G.number_of_nodes(), "m": G.number_of_edges(),
            "all_orders_exact": all(ex.values()),
        })
        if (i + 1) % 50 == 0 or i + 1 == len(graphs):
            print(f"computed {i+1}/{len(graphs)}", flush=True)

    rows = []
    N = len(graphs)
    total_pairs = N * (N - 1) // 2
    for upto in range(2, args.max_order + 1):
        sigs = [signature(c, d, upto) for c, d, ex in all_data]
        groups = defaultdict(int)
        for s in sigs:
            groups[s] += 1
        unresolved = sum(v * (v - 1) // 2 for v in groups.values())
        sep = total_pairs - unresolved
        exact_upto = all(all(ex.get(s, False) for s in range(2, upto + 1)) for _, _, ex in all_data)
        rows.append({
            "max_order": upto,
            "unique_signatures": len(groups),
            "separated_pairs": sep,
            "total_pairs": total_pairs,
            "separation_rate": sep / total_pairs if total_pairs else 1.0,
            "exact_upto_order": exact_upto,
        })

    pd.DataFrame(records).to_csv(args.out / "graph_metadata.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "order_summary.csv", index=False)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print("\n=== FAMILY EXPRESSIVITY ===")
    print(summary.to_string(index=False))
    if not bool(summary.iloc[-1]["exact_upto_order"]):
        print("\nWARNING: at least one order used Monte Carlo. Approximate signatures are screening evidence, not exact certificates.")


if __name__ == "__main__":
    main()
