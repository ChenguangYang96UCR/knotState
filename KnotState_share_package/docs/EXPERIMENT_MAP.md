# Experiment-to-code map

## 1. Expressivity headline

### SR25 / SRG family scans
Use:

- `core/knotstate_core.py`
- `expressivity/graph_family_expressivity.py`

These scripts compute order-resolved GF(2) adjacency-nullity profiles and count unique cumulative signatures / separated graph pairs.

### BREC
Primary historical descriptor scan:

- `expressivity/brec_knotstate_expressivity.py`

Additional controls/audits:

- `expressivity/brec_stage2B_graphlet_2wl.py`
- `expressivity/brec_stage2C_deep_audit.py`
- `expressivity/brec_stage2D_fwl_mechanism.py`

The stage-2 scripts are useful when discussing *why* a KnotState witness appears, and whether it is already implied by same-order graphlets or lower-dimensional WL controls.

### 4-WL-hard CFI screening

- `expressivity/brec_4wl_hard_kscore.py`

This is a screening script for the last 20 official CFI pairs. It uses Monte Carlo unless separately upgraded to exact enumeration. A positive `6-sigma` candidate is a reason to run a high-budget/exact certificate, not itself a theorem.

## 2. Target-conditioned KnotState

- `subgraph/knotstate_subgraph_sweep.py`

This is the original real-subgraph feature generator. Its core 75 channels are indexed by `(s,a,nu)` with `2 <= s <= 6`.

A cleaner minimal implementation is also available in:

- `core/knotstate_core.py::target_conditioned_knotstate`

## 3. WLKS fusion

The exact external WLKS kernel-generation code is not bundled because it is a separate project/repository. Use that implementation to produce one or more Gram matrices, then run:

- `fusion/wlks_knotstate_fusion.py`

The fusion is

```text
K_fusion = (1-lambda) K_WLKS + lambda K_KnotState.
```

Selection of WLKS kernel/layer, KnotState order, normalization, SVM C and lambda is performed using the validation split only.

## 4. Reliability / permutation analysis

- `fusion/wlks_knotstate_permutation.py`

Rows of KnotState are permuted **within train/validation/test separately**, and the entire validation-selection procedure is rerun. This is the appropriate code to share when collaborators ask how the reported fusion gain was validated statistically.

## 5. Why some files look historical

Several `brec_stage2*.py` files are preserved because they are the actual scripts used during the development/audit sequence. The cleaner `core`, `graph_family_expressivity`, and `fusion` scripts were added to make the package easier for collaborators to read and reuse.
