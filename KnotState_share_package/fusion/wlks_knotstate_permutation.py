#!/usr/bin/env python3
"""Matched within-split permutation validation for WLKS + KnotState fusion.

For each permutation the correspondence between target subgraphs and KnotState
rows is shuffled independently inside train/validation/test. The full
validation-based model-selection procedure is rerun. This estimates whether the
observed fusion gain is larger than what can arise from an unrelated auxiliary
feature matrix with the same marginal distribution and split sizes.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from wlks_knotstate_fusion import (
    parse_named_paths,load_array,load_splits,parse_num_list,
    baseline_search,search_fusion
)


def shuffle_within_splits(X, splits, rng):
    Xs=np.asarray(X).copy()
    for idx in splits.values():
        idx=np.asarray(idx,dtype=int)
        perm=rng.permutation(idx)
        Xs[idx]=X[perm]
    return Xs


def empirical_p(null, observed):
    null=np.asarray(null,dtype=float)
    return float((1+np.sum(null >= observed))/(len(null)+1))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wlks", action="append", required=True)
    ap.add_argument("--knot", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--orders", default="3,4,6")
    ap.add_argument("--lambdas", default="0,0.01,0.03,0.05,0.1,0.3,0.5,0.7,1")
    ap.add_argument("--C-grid", default="0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12")
    ap.add_argument("--norms", default="none,diag")
    ap.add_argument("--perms", type=int, default=99)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--fixed-order", type=int, default=0, help="e.g. 4 for a matched-K4 null; 0 tunes all --orders")
    ap.add_argument("--out", type=Path, default=Path("WLKS_KNOT_PERMUTATION"))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    wlks={name:np.asarray(load_array(p),dtype=float) for name,p in parse_named_paths(args.wlks).items()}
    X=np.asarray(load_array(args.knot),dtype=float); y=np.asarray(load_array(args.labels)); splits=load_splits(args.splits,len(y))
    Cs=parse_num_list(args.C_grid,float); lambdas=parse_num_list(args.lambdas,float)
    orders=[args.fixed_order] if args.fixed_order else parse_num_list(args.orders,int)
    norms=[x.strip() for x in args.norms.split(",") if x.strip()]

    _,bbest=baseline_search(wlks,y,splits,Cs,norms,False)
    _,real=search_fusion(wlks,X,y,splits,orders,lambdas,Cs,norms,False)
    base_test=float(bbest.test_micro_f1); real_gain=float(real.test_micro_f1-base_test)

    rng=np.random.default_rng(args.seed); rows=[]
    for b in range(args.perms):
        Xs=shuffle_within_splits(X,splits,rng)
        _,best=search_fusion(wlks,Xs,y,splits,orders,lambdas,Cs,norms,False)
        gain=float(best.test_micro_f1-base_test)
        rows.append({"perm":b,"gain":gain,"test_micro_f1":float(best.test_micro_f1),
                     "val_micro_f1":float(best.val_micro_f1),"lambda":float(best["lambda"]),
                     "max_order":int(best.max_order),"C":float(best.C),"norm":str(best.norm),"wlks":str(best.wlks)})
        if (b+1)%10==0 or b+1==args.perms:
            print(f"permutation {b+1}/{args.perms}",flush=True)
    df=pd.DataFrame(rows); df.to_csv(args.out/"null_runs.csv",index=False)
    null=df.gain.values
    summary={
        "baseline_test_micro_f1":base_test,
        "real_fusion_test_micro_f1":float(real.test_micro_f1),
        "real_gain":real_gain,
        "null_gain_mean":float(np.mean(null)),
        "null_gain_std":float(np.std(null,ddof=1)) if len(null)>1 else float("nan"),
        "real_minus_null_mean":float(real_gain-np.mean(null)),
        "empirical_p_upper":empirical_p(null,real_gain),
        "percentile":float(100*np.mean(null < real_gain)),
        "null_lambda_zero_fraction":float(np.mean(df["lambda"].values==0)),
        "permutations":args.perms,
        "fixed_order":args.fixed_order or None,
        "real_selected":real.to_dict(),
        "baseline_selected":bbest.to_dict(),
    }
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,default=float))
    print("\n=== MATCHED PERMUTATION RESULT ===")
    for k,v in summary.items():
        if k not in {"real_selected","baseline_selected"}: print(f"{k}: {v}")
    print("Saved:",args.out)

if __name__=="__main__": main()
