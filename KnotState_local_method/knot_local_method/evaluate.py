"""Validation-only selection for feature SVMs and optional WLKS fusion."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def model(y, C, kernel):
    classifier = SVC(C=C, kernel=kernel)
    return OneVsRestClassifier(classifier) if y.ndim == 2 else classifier


def score(y, pred):
    return float(f1_score(y, pred, average='micro', zero_division=0))


def normalize(K):
    d = np.sqrt(np.maximum(np.diag(K), 1e-12))
    return K / (d[:, None] * d[None, :])


def validate_kernel(K, n):
    if K.shape != (n, n) or not np.isfinite(K).all():
        raise ValueError('WLKS must be a finite n x n matrix')
    if not np.allclose(K, K.T, atol=1e-8) or np.any(np.diag(K) < 0):
        raise ValueError('WLKS must be symmetric with nonnegative diagonal')


def select_and_test(X, y, splits, columns, orders, Cs, wlks=None, lambdas=(0, .03, .1, .5, 1)):
    """Inspect test labels only after choosing the validation winner."""
    indices = {s: np.flatnonzero(splits == s) for s in ('train', 'val', 'test')}
    if any(not len(idx) for idx in indices.values()) or sum(map(len, indices.values())) != len(y):
        raise ValueError('Expected nonempty train/val/test splits and no unknown split labels')
    tr, va, te = (indices[s] for s in ('train', 'val', 'test'))
    candidates = []
    best = None
    orders = sorted(set(orders))
    Cs = sorted(set(Cs))
    if any(C <= 0 for C in Cs) or not Cs:
        raise ValueError('C values must be positive')
    if wlks is not None:
        validate_kernel(wlks, len(y))
    for order in orders:
        selected = [j for j, col in enumerate(columns) if 2 <= int(col[1]) <= order]
        if not selected:
            raise ValueError(f'No columns for order {order}')
        features = X[:, selected]
        if wlks is None:
            configs = [(C, kernel, None) for C in Cs for kernel in ('linear', 'rbf')]
        else:
            configs = [(C, 'precomputed', lam) for lam in sorted(set(lambdas)) for C in Cs]
            Kks = normalize(features @ features.T)
            Kw = normalize(wlks)
        for C, kernel, lam in configs:
            if lam is not None:
                if not 0 <= lam <= 1:
                    raise ValueError('lambda must lie in [0,1]')
                K = (1-lam)*Kw + lam*Kks
                estimator = model(y, C, kernel)
                estimator.fit(K[np.ix_(tr, tr)], y[tr])
                pred = estimator.predict(K[np.ix_(va, tr)])
            else:
                estimator = make_pipeline(StandardScaler(), model(y, C, kernel))
                estimator.fit(features[tr], y[tr])
                pred = estimator.predict(features[va])
            row = dict(order=int(order), C=float(C), kernel=kernel, **{'lambda':lam},
                       val_micro_f1=score(y[va], pred))
            candidates.append(row)
            # Iteration order supplies deterministic tie-breaking without test access.
            if best is None or row['val_micro_f1'] > best[0]['val_micro_f1']:
                best = (row.copy(), estimator, K if lam is not None else features)
    if best is None:
        raise ValueError('Empty search grid')
    row, estimator, data = best
    pred = estimator.predict(data[np.ix_(te, tr)] if row['lambda'] is not None else data[te])
    row['test_micro_f1'] = score(y[te], pred)
    return candidates, row, te, pred


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--orders', default='3,4')
    p.add_argument('--Cs', default='0.1,1,10')
    p.add_argument('--wlks', type=Path, help='NPZ with K and sample_ids matching features.npz')
    args = p.parse_args()
    cache = np.load(args.cache / 'features.npz', allow_pickle=False)
    manifest = json.loads((args.cache / 'manifest.json').read_text())
    orders = [int(x) for x in args.orders.split(',')]
    if any(x < 2 or x > manifest['config']['max_order'] for x in orders):
        raise ValueError('Requested order outside exported range')
    Cs = [float(x) for x in args.Cs.split(',')]
    y, splits = cache['labels'], cache['splits']
    wlks = None
    if args.wlks:
        external = np.load(args.wlks, allow_pickle=False)
        if not np.array_equal(external['sample_ids'], cache['sample_ids']):
            raise ValueError('WLKS sample_ids must match exported target row order')
        wlks = external['K']
        validate_kernel(wlks, len(y))
    args.out.mkdir(parents=True, exist_ok=True)
    summaries, searches, predictions = {}, {}, {}
    for name in ('target', 'rooted_mean', 'rooted_stats'):
        X = cache[name]
        if not np.isfinite(X).all():
            raise ValueError('Nonfinite feature values')
        tasks = [(name, None, (0,))]
        if wlks is not None:
            tasks.append((name + '_fusion', wlks, (0, .03, .1, .5, 1)))
            if name == 'target':
                tasks.append(('wlks', wlks, (0,)))
        for label, kernel, lambdas in tasks:
            table, summary, test_ids, pred = select_and_test(
                X, y, splits, manifest['columns'][name], orders, Cs, kernel, lambdas)
            searches[label], summaries[label], predictions[label] = table, summary, pred
            print(label, summary, flush=True)
    (args.out / 'validation_search.json').write_text(json.dumps(searches, indent=2))
    (args.out / 'summary.json').write_text(json.dumps(summaries, indent=2))
    np.savez_compressed(args.out / 'predictions.npz', sample_ids=cache['sample_ids'][test_ids],
                        labels=y[test_ids], **predictions)


if __name__ == '__main__':
    main()
