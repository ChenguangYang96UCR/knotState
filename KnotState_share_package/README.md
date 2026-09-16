# KnotState — expressivity and WLKS-fusion code package

This package collects the code that is useful for discussing the current KnotState project with collaborators. It intentionally separates three layers:

1. **Core representation** — GF(2) adjacency-nullity states on induced node sets.
2. **Expressivity audits** — graph-pair/family distinguishability, BREC audits, graphlet/WL controls.
3. **Subgraph learning and WLKS fusion** — generation of the 75-dimensional target-conditioned KnotState representation and validation-selected fusion with precomputed WLKS kernels.

## Mathematical object

For a simple graph `G` and an induced vertex set `S`, KnotState records the nullity over `GF(2)` of the principal adjacency submatrix `A_G[S]`.

Global profile:

```text
K_{s,nu}(G) = P[ nullity_GF2(A_G[S]) = nu | |S| = s ].
```

Target-conditioned profile used for subgraph tasks:

```text
K^{(r)}_{s,a,nu}(G,H)
  = P[ nullity_GF2(A_G[S]) = nu
       | |S|=s, |S ∩ H|=a, S ⊂ H ∪ B_r(H) ].
```

For `2 <= s <= 6`, the target-conditioned representation has 75 channels `(s,a,nu)`.

## Repository layout

```text
core/
  knotstate_core.py                 Clean reference implementation of global and target-conditioned KnotState.

expressivity/
  graph_family_expressivity.py      Generic SR25/SRG-style family scan from graph6 files.
  wl_pair_test.py                   Exact standard k-WL pair test for small/medium witness graphs.
  brec_knotstate_expressivity.py    Original BREC descriptor-level expressivity scan.
  brec_stage2B_graphlet_2wl.py      Full 4-graphlet + 2-WL controls.
  brec_stage2C_deep_audit.py        Exact deep-pair graphlet/nullity + 3-WL/relabeling audit.
  brec_stage2D_fwl_mechanism.py     3-FWL (= 4-WL level) mechanism audit on deep pairs.
  brec_4wl_hard_kscore.py           Screening script for the final 20 BREC CFI 4-WL-hard pairs.

subgraph/
  knotstate_subgraph_sweep.py       Original 75D target-conditioned KnotState feature/evaluation pipeline.
  export_fusion_inputs.py           Export labels and train/val/test split arrays for fusion.

fusion/
  wlks_knotstate_fusion.py          Clean, shareable validation-selected kernel fusion implementation.
  wlks_knotstate_permutation.py     Within-split matched permutation validation for fusion gain.

docs/
  EXPERIMENT_MAP.md                 Which script corresponds to which reported result.
  DATA_FORMATS.md                   Required input formats.
  REFERENCE_RESULTS.md              Headline numbers from the current experiments.
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py
```

No GPU is required for the expressivity or kernel-fusion scripts.

## Quick expressivity examples

### BREC smoke test

The BREC script can download the official `BREC_data_all.zip` automatically when networking is available, or use a local zip via `--zip`.

```bash
python expressivity/brec_knotstate_expressivity.py \
  --root ./runs/brec \
  --pair-limit 60 \
  --max-k 6 \
  --samples 2000 \
  --repeats 2
```

### SR25 / strongly-regular family

Provide a graph6 file with one graph per line:

```bash
python expressivity/graph_family_expressivity.py \
  --graphs /path/to/SR25.g6 \
  --max-order 6 \
  --exact-cap 2000000 \
  --out ./runs/sr25
```

If an order switches to Monte Carlo because `C(n,s) > exact_cap`, the output explicitly marks it as approximate. Do **not** use an MC-only difference as an exact mathematical certificate.

### Pairwise k-WL spot check

```bash
python expressivity/wl_pair_test.py --g graph_A.g6 --h graph_B.g6 -k 3
```

The implementation is exact but not optimized for a full large BREC sweep.

## Target-conditioned feature generation

The historical real-world subgraph data use the SubGNN text format:

```text
edge_list.txt
subgraphs.pth
```

Despite the `.pth` suffix, `subgraphs.pth` in these datasets is a text file.

Generate/cached KnotState features with:

```bash
python subgraph/knotstate_subgraph_sweep.py \
  --data-root /path/to/SubGNN_data \
  --out-root ./runs/knotstate_features \
  --datasets ppi_bp,hpo_neuro,hpo_metab,em_user \
  --profile balanced \
  --jobs 12
```

## WLKS + KnotState fusion

> **Provenance note.** The raw final server-side fusion script was not present in the accessible project-file snapshot, so `fusion/*.py` are clean collaborator-facing implementations of the documented validation-only fusion and matched-permutation protocol. See `docs/SOURCE_PROVENANCE.md`.

The external WLKS implementation is **not redistributed here**. Produce the WLKS Gram matrix/matrices using the official WLKS code, then pass the resulting full `n x n` kernels to the fusion script.

First export labels/splits:

```bash
python subgraph/export_fusion_inputs.py \
  --subgraphs /path/to/dataset/raw/subgraphs.pth \
  --out ./fusion_inputs
```

Then run validation-selected fusion:

```bash
python fusion/wlks_knotstate_fusion.py \
  --wlks wl2=/path/to/WLKS_layer2.npy \
  --wlks wl3=/path/to/WLKS_layer3.npy \
  --knot /path/to/knot_r1_m2048_seed2_exact2000.npy \
  --labels ./fusion_inputs/labels.npy \
  --splits ./fusion_inputs/splits.npy \
  --orders 3,4,6 \
  --lambdas 0,0.01,0.03,0.05,0.1,0.3,0.5,0.7,1 \
  --out ./runs/fusion
```

The test split is not used for hyperparameter selection. The search table and the selected configuration are saved to disk.

### Matched permutation test

```bash
python fusion/wlks_knotstate_permutation.py \
  --wlks wl2=/path/to/WLKS_layer2.npy \
  --knot /path/to/knot_features.npy \
  --labels ./fusion_inputs/labels.npy \
  --splits ./fusion_inputs/splits.npy \
  --orders 3,4,6 \
  --perms 99 \
  --out ./runs/permutation_global
```

For a matched fixed-order control, for example `K_{<=4}`:

```bash
python fusion/wlks_knotstate_permutation.py \
  --wlks wl2=/path/to/WLKS_layer2.npy \
  --knot /path/to/knot_features.npy \
  --labels ./fusion_inputs/labels.npy \
  --splits ./fusion_inputs/splits.npy \
  --fixed-order 4 \
  --perms 99 \
  --out ./runs/permutation_k4
```

## Important claim discipline

- The GF(2) adjacency-nullity primitive is related to the interlace-polynomial literature; the novelty here is the order-resolved / target-conditioned localized representation and the study of its WL complementarity.
- Current evidence supports **non-nested / complementary distinguishing power**, not universal domination of WL.
- Exact enumeration and Monte Carlo screening are separated in the code and should be described differently in a paper.
- The CFI-hard family is a current limitation, not something to hide.
- Fusion gains must be reported with validation-only model selection and reliability/statistical assessment.

See `docs/REFERENCE_RESULTS.md` for the current headline numbers and `docs/EXPERIMENT_MAP.md` for provenance.
