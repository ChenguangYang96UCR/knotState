# Data formats

## graph6 graph family

`graph_family_expressivity.py` expects one graph6 string per line.

Example:

```text
Dhc
D~{
...
```

## BREC

The original BREC expressivity scripts expect the official BREC release (`BREC_data_all.zip` / `brec_v3.npy`).

The 4-WL-hard CFI screening script expects the official CFI array:

```text
customize/Data/raw/cfi.npy
```

with shape `(100,2)` in the release used in our experiments.

## SubGNN-style real subgraph datasets

`edge_list.txt`:

```text
u v
u v
...
```

`subgraphs.pth` (text despite the suffix):

```text
node-node-node<TAB>label-or-labels<TAB>train|val|test
```

## KnotState feature matrix

For the real subgraph experiments, the standard cache is `n x 75` in NumPy `.npy` format. The 75 columns follow `KNOT_KEYS_75` in `core/knotstate_core.py`:

```text
for s=2..6:
  for a=0..s:
    for admissible nullity nu:
      (s,a,nu)
```

## WLKS kernel

The clean fusion code expects a full square Gram matrix `K` of shape `n x n` in `.npy`, `.npz`, or headerless `.csv` form.

If `.npz` is used, a key named `K` is preferred.

## Labels

- single-label: `labels.npy` of shape `(n,)`
- multi-label: binary indicator `labels.npy` of shape `(n,c)`

## Splits

Either:

- `splits.npy` of length `n` containing strings `train`, `val`, `test`; or
- `splits.npz` containing integer arrays `train`, `val`, `test`.
