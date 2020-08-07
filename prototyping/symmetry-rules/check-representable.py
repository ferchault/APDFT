#!/usr/bin/env python
import networkx as nx
import sys
import numpy as np
from scipy import spatial


def build_diamond_lattice():
    g = nx.Graph()
    N = 4
    pad = 1
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
    for x in range(N + 2 * pad):
        for y in range(N + 2 * pad):
            for z in range(N + 2 * pad):
                for point in unit_cell:
                    pts.append(np.array(point) + 4 * np.array((x, y, z)))
                    g.add_node(siteid)
                    is_padding = True
                    if 1 <= x <= N and 1 <= y <= N and 1 <= z <= N:
                        is_padding = False
                    g.nodes[siteid]["is_padding"] = is_padding
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


def build_hexagonal_lattice():
    return nx.generators.lattice.hexagonal_lattice_graph(10, 10)


def is_representable(supergraph, graph):
    def no_padding(n1, n2):
        return not n1["is_padding"]

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        supergraph, graph, node_match=no_padding
    )
    retval = matcher.subgraph_is_isomorphic()
    return retval, matcher.mapping


def run_tests():
    # linear chain
    supergraph_diamond = build_diamond_lattice()
    supergraph_hexagonal = build_hexagonal_lattice()
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    assert True == is_representable(supergraph_diamond, g)
    assert True == is_representable(supergraph_hexagonal, g)
    # three-cycle
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    assert False == is_representable(supergraph_diamond, g)
    assert False == is_representable(supergraph_hexagonal, g)
    # three fused diamond rings
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    g.add_edge(5, 0)
    g.add_edge(5, 6)
    g.add_edge(6, 9)
    g.add_edge(3, 7)
    g.add_edge(7, 9)
    g.add_edge(1, 8)
    g.add_edge(8, 9)
    assert True == is_representable(supergraph_diamond, g)
    assert False == is_representable(supergraph_hexagonal, g)
    # graphene segment
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 11)
    g.add_edge(11, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 5)
    g.add_edge(5, 6)
    g.add_edge(6, 7)
    g.add_edge(7, 8)
    g.add_edge(8, 9)
    g.add_edge(9, 10)
    g.add_edge(10, 0)
    g.add_edge(9, 12)
    g.add_edge(11, 12)
    g.add_edge(5, 12)
    assert True == is_representable(supergraph_diamond, g)
    assert True == is_representable(supergraph_hexagonal, g)


if __name__ == "__main__":
    run_tests()
    gridtype = sys.argv[2]
    if gridtype == "DIAMOND":
        supergraph = build_diamond_lattice()
    if gridtype == "HEXAGONAL":
        supergraph = build_hexagonal_lattice()

    infile = sys.argv[1]
    with open(infile) as fh:
        for line in fh:
            line = line.strip()
            graph = nx.from_graph6_bytes(line.encode("ascii"))
            if is_representable(graph)[0]:
                print(line)
