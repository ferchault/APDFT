#!/usr/bin/env python
#%%
import networkx as nx
import sys
import numpy as np
from scipy import spatial


def build_lattice():
    g = nx.Graph()
    N = 4
    unit_cell = (
        (0, 0, 0),
        (0, 2, 2),
        (2, 0, 2),
        (2, 2, 0),
        (3, 3, 3),
        (3, 1, 1),
        (1, 3, 1),
        (1, 1, 3),
    )
    pts = []
    siteid = 0
    for x in range(N):
        for y in range(N):
            for z in range(N):
                for point in unit_cell:
                    pts.append(np.array(point) + 4 * np.array((x, y, z)))
                    g.add_node(siteid)
                    siteid += 1
    pts = np.array(pts)
    tree = spatial.KDTree(pts)
    neighbors = tree.query(pts, 5)
    for i in range(len(pts)):
        distances = neighbors[0][i] ** 2
        correct_distances = np.abs(distances - 3) < 1e-3
        bonded_to = neighbors[1][i][correct_distances]
        for j in bonded_to:
            g.add_edge(i, j)
    return g


supergraph = build_lattice()

infile = sys.argv[1]


def is_diamond_representable(graph):
    matcher = nx.algorithms.isomorphism.GraphMatcher(supergraph, graph)
    return matcher.subgraph_is_isomorphic()


def test():
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    print(is_diamond_representable(g))
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    print(is_diamond_representable(g))


# %%
with open(infile) as fh:
    for line in fh:
        line = line.strip()
        graph = nx.from_graph6_bytes(line.encode("ascii"))
        if is_diamond_representable(graph):
            print(line)


# %%
