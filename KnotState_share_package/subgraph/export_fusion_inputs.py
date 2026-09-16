#!/usr/bin/env python3
"""Export labels/splits from SubGNN-style subgraphs.pth for the fusion scripts.

The historical datasets use a text format despite the .pth suffix:
  nodes-separated-by-dashes<TAB>label(s)<TAB>train|val|test
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def read_subgraphs(path: Path):
    labels, splits = [], []
    multi = False
    with path.open() as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3: continue
            labs = [x for x in p[1].split("-") if x]
            split = p[2].strip().lower()
            if not labs or split not in {"train","val","test"}: continue
            multi |= len(labs) > 1
            labels.append(labs); splits.append(split)
    uniq = sorted({x for row in labels for x in row})
    lut = {x:i for i,x in enumerate(uniq)}
    if multi:
        Y = np.zeros((len(labels), len(uniq)), dtype=np.int8)
        for i,row in enumerate(labels):
            for x in row: Y[i,lut[x]] = 1
    else:
        Y = np.asarray([lut[row[0]] for row in labels], dtype=np.int64)
    return Y, np.asarray(splits), uniq


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--subgraphs",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    Y,splits,names=read_subgraphs(args.subgraphs)
    np.save(args.out/"labels.npy",Y); np.save(args.out/"splits.npy",splits)
    (args.out/"label_names.txt").write_text("\n".join(names))
    print("labels",Y.shape,"splits",{s:int((splits==s).sum()) for s in ["train","val","test"]})

if __name__=="__main__": main()
