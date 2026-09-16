"""Export aligned SubGNN features and labels without pickle-based caches."""
import argparse
import hashlib
import json
from pathlib import Path
import networkx as nx
import numpy as np
from .features import extract, pool


def read_data(edges, targets):
    G = nx.Graph()
    for number, line in enumerate(edges.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f'Invalid edge line {number}')
        u, v = map(int, fields)
        G.add_node(u)
        G.add_node(v)
        if u != v:
            G.add_edge(u, v)
    rows, labels, splits = [], [], []
    for number, line in enumerate(targets.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split('\t')
        if len(fields) != 3:
            raise ValueError(f'Invalid target line {number}')
        nodes, labs, split = fields
        nodes = sorted(set(map(int, nodes.split('-'))))
        labs = labs.split('-')
        split = split.strip().lower()
        if not nodes or not all(labs) or split not in ('train', 'val', 'test'):
            raise ValueError(f'Invalid target line {number}')
        G.add_nodes_from(nodes)
        rows.append(nodes)
        labels.append(labs)
        splits.append(split)
    if not rows:
        raise ValueError('Empty target dataset')
    names = sorted({label for row in labels for label in row})
    lut = {name: i for i, name in enumerate(names)}
    if any(len(row) > 1 for row in labels):
        y = np.zeros((len(rows), len(names)), dtype=int)
        for i, row in enumerate(labels):
            y[i, [lut[label] for label in row]] = 1
    else:
        y = np.asarray([lut[row[0]] for row in labels])
    return G, rows, y, np.asarray(splits), names


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edges', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--radius', type=int, default=1)
    p.add_argument('--max-order', type=int, default=4)
    p.add_argument('--budget', type=int, default=4000)
    p.add_argument('--max-roots', type=int, default=16)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    G, targets, y, splits, names = read_data(args.edges, args.targets)
    matrices = {name: [] for name in ('target', 'rooted_mean', 'rooted_stats')}
    metadata, audit = {}, []
    local, masks, roots, offsets = [], [], [], [0]
    for i, H in enumerate(targets):
        kwargs = dict(radius=args.radius, max_order=args.max_order, budget=args.budget,
                      seed=args.seed * 1000003 + i, max_roots=args.max_roots)
        old = extract(G, H, mode='target', **kwargs)
        new = extract(G, H, mode='rooted', **kwargs)
        for name, result, kind in [('target', old, 'mean'), ('rooted_mean', new, 'mean'),
                                   ('rooted_stats', new, 'stats')]:
            vector, columns = pool(result, kind)
            if name == 'target':
                vector = result['values'][0]
                columns = [['mean', int(s), int(a), int(nu)] for s, a, nu in result['keys']]
            matrices[name].append(vector)
            metadata[name] = columns
        local.append(new['values']); masks.append(new['mask']); roots.extend(new['roots'])
        offsets.append(offsets[-1] + len(new['roots']))
        audit.append({name: {k: result[k] for k in ('evaluations', 'seconds', 'blocks')}
                      for name, result in [('target', old), ('rooted', new)]})
        print(f'{i+1}/{len(targets)} targets', flush=True)
    np.savez_compressed(args.out / 'features.npz', **{k: np.asarray(v) for k, v in matrices.items()},
                        labels=y, splits=splits, sample_ids=np.arange(len(targets)))
    np.savez_compressed(args.out / 'local.npz', values=np.concatenate(local), mask=np.concatenate(masks),
                        roots=np.asarray(roots), offsets=np.asarray(offsets), keys=new['keys'])
    manifest = dict(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    columns=metadata, label_names=names, targets=targets, audit=audit,
                    sha256={name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for name, path in [('edges', args.edges), ('targets', args.targets)]})
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
