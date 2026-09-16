"""Create a small synthetic dataset for end-to-end smoke testing only."""
import argparse
from pathlib import Path
import networkx as nx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    G = nx.Graph()
    rows = []
    for i in range(30):
        n = 6
        graph = nx.path_graph(n) if i % 2 == 0 else nx.complete_graph(n)
        graph = nx.relabel_nodes(graph, {v:i*n+v for v in graph})
        G.update(graph)
        split = 'train' if i < 18 else 'val' if i < 24 else 'test'
        rows.append('-'.join(map(str, sorted(graph))) + f'\t{i%2}\t{split}')
    nx.write_edgelist(G, args.out/'edge_list.txt', data=False)
    (args.out/'subgraphs.pth').write_text('\n'.join(rows)+'\n')
    print('Synthetic smoke data written to', args.out)


if __name__ == '__main__':
    main()
