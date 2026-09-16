#!/usr/bin/env python3
from pathlib import Path
import sys, tempfile
import numpy as np
import networkx as nx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.knotstate_core import self_test, target_conditioned_knotstate

self_test()
G=nx.cycle_graph(6)
x=target_conditioned_knotstate(G,[0,1,2],radius=1,samples=50,seed=7)
assert x.shape==(75,) and np.isfinite(x).all()
print("All package core smoke tests passed.")
