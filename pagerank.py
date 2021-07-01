#!/usr/bin/python

# pip install -r requirements.txt

import dgl
import networkx as nx
import numpy as np
import torch
from dgl import function as fn
from matplotlib import pyplot as plt


def build_simple_graph(show_plt: bool = False) -> dgl.DGLGraph:
    g = dgl.DGLGraph()
    g.add_nodes(3)
    src = np.array([0, 0, 1, 2])
    dst = np.array([1, 2, 2, 0])
    g.add_edges(src, dst)
    print("node:", g.number_of_nodes())
    print("edge:", g.number_of_edges())
    nx.draw(g.to_networkx(), node_size=3)
    if show_plt:
        plt.show()
    return g


def pagerank_builtin(g: dgl.DGLGraph, damp: float = 0.85) -> None:
    g.ndata['pv'] /= g.ndata['deg']
    g.update_all(message_func=fn.copy_src(src='pv', out='m'),
                 reduce_func=fn.sum(msg='m', out='m_sum'))
    N = g.number_of_nodes()
    g.ndata['pv'] = (1 - damp) / N + damp * g.ndata['m_sum']


def main() -> None:
    F = build_simple_graph()
    node_cnt = F.number_of_nodes()
    F.ndata['pv'] = torch.ones(node_cnt) / node_cnt
    F.ndata['deg'] = F.out_degrees(F.nodes()).float()

    print("init vals:", F.ndata['pv'])
    for _ in range(3):
        pagerank_builtin(F)
        print('step %d:' % _, F.ndata['pv'])


if __name__ == '__main__':
    main()
