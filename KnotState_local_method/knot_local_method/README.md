# Local KnotState: controlled first-stage experiments

This implementation tests whether node-centered structural distributions improve prediction before introducing GNNs, attention, low-rank projections, or learned sampling. Those extensions are not implemented here. The original `KnotState_share_package` remains independent.

## Step 1: Set up the environment

The project directory is:

```text
/Users/chenguangyang/Desktop/ucr_work/knotState/KnotState_local_method
```

Run the following commands from that directory. The Python package is the nested `knot_local_method/` directory.

```bash
cd /Users/chenguangyang/Desktop/ucr_work/knotState/KnotState_local_method
python -m pip install -r knot_local_method/requirements.txt
python -m unittest discover -s knot_local_method/tests -v
```

Use the same Python interpreter for dependency installation and execution.

## Step 2: Extract rooted features

`features.py` provides `extract(..., mode="rooted")`. For each selected center v in target H, it selects vertex sets S subject to:

- S contains v and lies inside the closed r-hop neighborhood of v.
- |S|=s and |S intersect H|=a, where a>=1.
- All induced edges are retained; disconnected induced subgraphs are allowed.
- Sampling is uniform within each feasible group.

The descriptor records adjacency nullity over GF(2), computed as s minus the rank obtained by XOR elimination. Each (v,s,a) histogram is normalized separately.

The result includes values, validity masks, channel keys, root IDs, candidate counts, evaluation counts, exactness flags, and elapsed time. An invalid channel means that the selection is impossible, rather than that a particular nullity has zero probability.

`mode="target"` computes the original target-conditioned distribution over the r-hop neighborhood of H and allows a=0. Its mathematical descriptor matches the original formulation, but its sampling-budget policy differs; it does not directly reproduce historical results.

## Step 3: Control the computation budget

`--budget` caps nullity evaluations per target and per extraction method. Feasible groups receive approximately equal allocations. Small groups are enumerated exactly, and unused capacity is redistributed. Larger groups are sampled independently with replacement across draws; nodes never repeat within a draw.

`--max-roots 16` uniformly selects at most 16 target nodes as centers. Use 0 for all target nodes. Rooted mean and rooted mean/std features reuse the same extracted local matrix. A budget smaller than the number of feasible groups raises an error instead of silently dropping groups.

Equal budget caps do not guarantee equal actual computation: small candidate spaces finish exactly. Compare actual evaluation counts and runtime in the manifest. Exact features are invariant under corresponding node relabeling; finite Monte Carlo estimates need not be identical after relabeling.

## Step 4: Export a real dataset

Input formats:

- `edge_list.txt`: two integer node IDs per line.
- `subgraphs.pth`: text despite its extension, with `node-node<TAB>label-label<TAB>train|val|test` on each line.

Parsing is strict; malformed samples are not silently skipped. Labels use one shared vocabulary. Self-loop input lines preserve the node but not the loop, so the loaded graph is simple and undirected. Node IDs should be nonnegative integers, consistent with the hyphen-delimited format.

```bash
python -m knot_local_method.export \
  --edges /path/to/raw/edge_list.txt \
  --targets /path/to/raw/subgraphs.pth \
  --radius 1 --max-order 4 --budget 4000 --max-roots 16 --seed 0 \
  --out work/local_seed0
```

Outputs:

- `features.npz`: target, rooted_mean, rooted_stats, labels, splits, and sample_ids.
- `local.npz`: unpooled local values, masks, roots, offsets, and channel keys. Rows for target i are `[offsets[i]:offsets[i+1]]`.
- `manifest.json`: input SHA256 hashes, configuration, label vocabulary, target nodes, explicit column metadata, and per-group computation records.

Exports recompute features and overwrite files in the selected output directory. Use separate directories for different seeds. Cache loading does not enable pickle.

## Step 5: Pool features and evaluate classifiers

The representations are:

- `target`: target-conditioned probability vector, without coverage columns.
- `rooted_mean`: channel means over feasible centers plus the fraction of feasible centers (coverage).
- `rooted_stats`: those means, population standard deviations over feasible centers, and coverage.

Coverage contains feasibility information, not just nullity information. If improvements appear, add an ablation removing coverage. These pooled representations do not preserve node adjacency and do not establish the benefit of node-level GNN interactions.

```bash
python -m knot_local_method.evaluate \
  --cache work/local_seed0 --orders 3,4 --Cs 0.1,1,10 \
  --out work/eval_seed0
```

The classifier search uses StandardScaler followed by a linear or RBF SVM. Scaling is fitted on training examples only. RBF uses gamma=scale, and multilabel tasks use OneVsRest. Explicit column metadata determines the selected orders; there is no hardcoded 75-dimensional assumption.

Candidate configurations are evaluated on validation micro-F1 only. Each predefined method evaluates its selected configuration on test once. The model remains fitted on training examples only. For feature SVM ties, lower order, lower C, and then linear kernel take priority. For fusion ties, lower order, lower lambda, and then lower C take priority.

Outputs are `validation_search.json` (without test scores), `summary.json`, and `predictions.npz` containing test sample IDs, labels, and predictions. Use validation for continued development rather than repeatedly adapting the method to test results.

## Step 6: Optionally combine with WLKS

Supply an external NPZ containing `K` (n by n) and `sample_ids` aligned with exported target rows. IDs must reflect the actual kernel row ordering; assigning arange to an incorrectly ordered matrix does not establish alignment. The implementation checks dimensions, symmetry, finite entries, nonnegative diagonal, and matching IDs. It neither computes WLKS nor performs an expensive PSD eigendecomposition check.

```bash
python -m knot_local_method.evaluate \
  --cache work/local_seed0 --wlks /path/to/aligned_wlks.npz \
  --out work/fusion_seed0
```

Both kernels are diagonally normalized before mixing:

```text
K_KS = normalize(Z @ Z.T)
K_fusion = (1-lambda) * normalize(K_WLKS) + lambda * K_KS
```

The lambda grid is 0, 0.03, 0.1, 0.5, and 1. A WLKS-only result is also reported. This version accepts one predetermined WLKS matrix per run rather than tuning external layers. Fusion does not apply StandardScaler, so interpret it separately from standalone feature SVMs.

## Step 7: Run the end-to-end smoke example

```bash
python -m knot_local_method.demo --out work/demo
python -m knot_local_method.export --edges work/demo/edge_list.txt --targets work/demo/subgraphs.pth --budget 200 --max-roots 4 --out work/demo_cache
python -m knot_local_method.evaluate --cache work/demo_cache --out work/demo_eval
```

The demo separates paths and complete graphs. It verifies execution, not scientific effectiveness. The optional fusion interface was also checked with a synthetic Gram matrix, not a real WLKS result.

For real experiments, keep splits fixed and repeat extraction with seeds 0, 1, and 2. Report micro-F1, seed variation, evaluation counts, and runtime. Three seeds on one split are not three independent datasets. Complete the four real-dataset comparisons before broadening the radius/order search.

## File responsibilities

| File | Responsibility |
|---|---|
| features.py | GF(2) nullity, rooted sampling, budget allocation, pooling |
| export.py | Dataset parsing, aligned caches, computation records |
| evaluate.py | Validation-only selection, final test, optional fusion |
| demo.py | Synthetic end-to-end smoke data |
| tests/test_method.py | Mathematical checks, sampling constraints, relabeling, test isolation |

The test suite includes eight checks covering known nullities, exact profiles, validity masks, sampling constraints and budgets, exact relabeling, Monte Carlo accuracy/reproducibility, empty neighborhoods, and test-label isolation for single-label and multilabel evaluation.
