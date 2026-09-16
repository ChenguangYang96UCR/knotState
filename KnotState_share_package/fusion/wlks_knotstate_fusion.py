#!/usr/bin/env python3
"""Validation-selected WLKS + KnotState kernel fusion.

This script deliberately does NOT bundle the external WLKS implementation.
Instead it consumes one or more precomputed WLKS Gram matrices and combines
them with a KnotState linear kernel:

    K_fusion = (1-lambda) K_WLKS + lambda K_KS.

Model selection (WLKS kernel/layer, KnotState max order, normalization, C,
lambda) is performed on the validation split only. The test split is evaluated
once for the selected configuration.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.knotstate_core import KNOT_KEYS_75, select_target_columns


def parse_num_list(s, cast=float):
    return [cast(x) for x in str(s).split(",") if str(x).strip()]


def parse_named_paths(items: List[str]) -> Dict[str, Path]:
    out = {}
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item; name = Path(path).stem
        out[name] = Path(path)
    return out


def load_array(path: Path):
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        if "K" in z: return z["K"]
        if "arr_0" in z: return z["arr_0"]
        if len(z.files) == 1: return z[z.files[0]]
        raise ValueError(f"{path}: NPZ has multiple arrays; expected key K or one array")
    if path.suffix == ".csv":
        return pd.read_csv(path, header=None).values
    raise ValueError(f"Unsupported array format: {path}")


def load_splits(path: Path, n: int):
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        if all(k in z for k in ["train","val","test"]):
            return {k: np.asarray(z[k], dtype=int) for k in ["train","val","test"]}
    a = load_array(path)
    if a.ndim == 1 and len(a) == n:
        if a.dtype.kind in "OUS":
            x = np.asarray([str(v).lower() for v in a])
            return {k: np.where(x == k)[0] for k in ["train","val","test"]}
        # numeric convention: 0 train, 1 val, 2 test
        return {"train":np.where(a==0)[0],"val":np.where(a==1)[0],"test":np.where(a==2)[0]}
    raise ValueError("splits must be an NPZ with train/val/test indices or a length-n vector of split labels")


def diag_normalize(K: np.ndarray, eps: float = 1e-12):
    d = np.sqrt(np.maximum(np.diag(K), eps))
    return K / (d[:,None] * d[None,:])


def center_by_train(K: np.ndarray, train_idx: np.ndarray):
    # Center in the feature space using training-set means; applies to full Gram.
    tr = np.asarray(train_idx, dtype=int)
    row_mean = K[:, tr].mean(axis=1, keepdims=True)
    col_mean = K[tr, :].mean(axis=0, keepdims=True)
    grand = K[np.ix_(tr,tr)].mean()
    return K - row_mean - col_mean + grand


def prepare_kernel(K, norm, train_idx):
    K = np.asarray(K, dtype=float)
    if norm == "none": return K
    if norm == "diag": return diag_normalize(K)
    if norm == "center": return center_by_train(K, train_idx)
    if norm == "center_diag": return diag_normalize(center_by_train(K, train_idx))
    raise ValueError(norm)


def knot_kernel(X, max_order, norm, train_idx):
    X = np.asarray(X, dtype=float)
    if X.shape[1] == 75:
        cols = select_target_columns(max_order, min_s=2, keys=KNOT_KEYS_75)
        X = X[:, cols]
    elif max_order not in (0, -1):
        # For non-75D user features, max_order is metadata only.
        pass
    K = X @ X.T
    return prepare_kernel(K, norm, train_idx)


def micro_f1(y_true, y_pred):
    if np.asarray(y_true).ndim == 1:
        return float(f1_score(y_true, y_pred, average="micro"))
    return float(f1_score(y_true, y_pred, average="micro", zero_division=0))


def fit_predict_precomputed(K, y, train_idx, eval_idx, C):
    tr, ev = np.asarray(train_idx), np.asarray(eval_idx)
    Ktr = K[np.ix_(tr,tr)]
    Kev = K[np.ix_(ev,tr)]
    ytr = y[tr]
    if np.asarray(y).ndim == 1:
        model = SVC(C=C, kernel="precomputed")
    else:
        model = OneVsRestClassifier(SVC(C=C, kernel="precomputed"))
    model.fit(Ktr, ytr)
    return model.predict(Kev)


def evaluate(K, y, splits, C, refit_train_val=False):
    tr, va, te = splits["train"], splits["val"], splits["test"]
    pv = fit_predict_precomputed(K, y, tr, va, C)
    val = micro_f1(y[va], pv)
    fit_idx = np.concatenate([tr,va]) if refit_train_val else tr
    pt = fit_predict_precomputed(K, y, fit_idx, te, C)
    test = micro_f1(y[te], pt)
    return val, test


def search_fusion(wlks: Dict[str,np.ndarray], Xk, y, splits, orders, lambdas, Cs, norms,
                  refit_train_val=False):
    rows = []
    tr = splits["train"]
    for wl_name, raw_kw in wlks.items():
        for norm in norms:
            Kw = prepare_kernel(raw_kw, norm, tr)
            for order in orders:
                Kk = knot_kernel(Xk, order, norm, tr)
                for lam in lambdas:
                    K = (1.0-lam)*Kw + lam*Kk
                    for C in Cs:
                        val, test = evaluate(K,y,splits,C,refit_train_val=refit_train_val)
                        rows.append({"wlks":wl_name,"norm":norm,"max_order":order,"lambda":lam,"C":C,
                                     "val_micro_f1":val,"test_micro_f1":test})
    df = pd.DataFrame(rows)
    # deterministic tie-break: higher val, then lower lambda, lower order, lower C, lexicographic kernel
    best = df.sort_values(["val_micro_f1","lambda","max_order","C","wlks"],
                          ascending=[False,True,True,True,True]).iloc[0]
    return df, best


def baseline_search(wlks, y, splits, Cs, norms, refit_train_val=False):
    rows=[]; tr=splits["train"]
    for name,Kraw in wlks.items():
        for norm in norms:
            K=prepare_kernel(Kraw,norm,tr)
            for C in Cs:
                val,test=evaluate(K,y,splits,C,refit_train_val=refit_train_val)
                rows.append({"wlks":name,"norm":norm,"C":C,"val_micro_f1":val,"test_micro_f1":test})
    df=pd.DataFrame(rows)
    best=df.sort_values(["val_micro_f1","C","wlks"],ascending=[False,True,True]).iloc[0]
    return df,best


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wlks", action="append", required=True, help="name=/path/kernel.npy; repeat for WL layers")
    ap.add_argument("--knot", type=Path, required=True, help="n x 75 target-conditioned KnotState features")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--orders", default="3,4,6")
    ap.add_argument("--lambdas", default="0,0.01,0.03,0.05,0.1,0.3,0.5,0.7,1")
    ap.add_argument("--C-grid", default="0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12")
    ap.add_argument("--norms", default="none,diag")
    ap.add_argument("--refit-train-val", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("WLKS_KNOT_FUSION"))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)

    paths=parse_named_paths(args.wlks)
    wlks={name:np.asarray(load_array(p),dtype=float) for name,p in paths.items()}
    Xk=np.asarray(load_array(args.knot),dtype=float)
    y=np.asarray(load_array(args.labels))
    n=len(y); splits=load_splits(args.splits,n)
    for name,K in wlks.items():
        if K.shape != (n,n): raise ValueError(f"WLKS {name} shape {K.shape}, expected {(n,n)}")
    if Xk.shape[0] != n: raise ValueError(f"KnotState rows {Xk.shape[0]} != labels {n}")

    orders=parse_num_list(args.orders,int); lambdas=parse_num_list(args.lambdas,float); Cs=parse_num_list(args.C_grid,float)
    norms=[x.strip() for x in args.norms.split(",") if x.strip()]

    bdf, bbest=baseline_search(wlks,y,splits,Cs,norms,args.refit_train_val)
    fdf, fbest=search_fusion(wlks,Xk,y,splits,orders,lambdas,Cs,norms,args.refit_train_val)
    bdf.to_csv(args.out/"baseline_search.csv",index=False); fdf.to_csv(args.out/"fusion_search.csv",index=False)
    summary={
        "n":n,
        "split_sizes":{k:int(len(v)) for k,v in splits.items()},
        "baseline":bbest.to_dict(),
        "fusion":fbest.to_dict(),
        "gain_vs_selected_baseline":float(fbest.test_micro_f1-bbest.test_micro_f1),
        "note":"All model/hyperparameter selection uses validation only; test is read after candidate selection."
    }
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2,default=float))
    print("\n=== WLKS BASELINE ==="); print(bbest.to_string())
    print("\n=== WLKS + KNOTSTATE FUSION ==="); print(fbest.to_string())
    print(f"\nTest gain vs selected WLKS baseline: {summary['gain_vs_selected_baseline']:+.4f}")
    print("Saved:",args.out)

if __name__=="__main__": main()
