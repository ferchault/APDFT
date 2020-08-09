#!/usr/bin/env python
import sys
from math import gcd
from functools import reduce
import check_representable
import numpy as np
import networkx as nx
import itertools as it
import copy


def simplify(lhs, rhs):
    values = []
    for k in lhs.keys():
        remove = min(lhs[k], rhs[k])
        lhs[k] -= remove
        rhs[k] -= remove
        if lhs[k] > 0:
            values.append(lhs[k])
        if rhs[k] > 0:
            values.append(rhs[k])
    if len(values) > 0:
        red = reduce(gcd, values)
        if red != 1:
            for k in lhs.keys():
                lhs[k] //= red
                rhs[k] //= red

    return lhs, rhs


def eformat(lhs, rhs):
    lhsstr = ""
    rhsstr = ""
    for k in sorted(lhs.keys()):
        if lhs[k] > 0:
            numrep = ""
            if lhs[k] > 1:
                numrep = str(lhs[k])
            lhsstr += "+%s%s" % (numrep, k)
        if rhs[k] > 0:
            numrep = ""
            if rhs[k] > 1:
                numrep = str(rhs[k])
            rhsstr += "+%s%s" % (numrep, k)
    lhsstr, rhsstr = sorted((lhsstr, rhsstr))
    eqnstr = lhsstr.strip("+") + "=" + rhsstr.strip("+")
    return eqnstr


def build_pathcache(graph, n):
    """ Builds a list of all unique paths of length n in graph."""
    paths = []
    if n not in (2, 3):
        raise NotImplementedError()
    if n == 3:
        for source in range(graph.number_of_nodes()):
            for sink in range(source + 1, graph.number_of_nodes()):
                for intermediate in graph[source]:
                    if sink in graph[intermediate]:
                        paths.append([source, intermediate, sink])
    if n == 2:
        for source in range(graph.number_of_nodes()):
            for other in graph[source]:
                if other > source:
                    paths.append([source, other])
    return np.array(paths)

def graph_to_equation(supergraph, g, pathcache, ntuples):
    mapping = check_representable.is_representable(supergraph, g)[1]
    supercolors = np.zeros(supergraph.number_of_nodes(), dtype=np.int)
    for k, v in mapping.items():
        supercolors[k] = colors[v]

    paths = supercolors[pathcache]
    paths = paths[np.sum(paths, axis=1) > 0]

    lhs = {"".join(_): 0 for _ in list(it.product(sorted(elements_up), repeat=ntuples))}
    rhs = copy.deepcopy(lhs)
    for row in paths:
        label = "".join([elements_up[_] for _ in row])
        if label[-1] < label[0]:
            label = label[::-1]
        lhs[label] += 1

        label = "".join([elements_dn[_] for _ in row])
        if label[-1] < label[0]:
            label = label[::-1]
        rhs[label] += 1

    lhs, rhs = simplify(lhs, rhs)
    return eformat(lhs, rhs)


# nv ne {col} {v1 v2}
# 4 3 0 0 0 0  0 3 1 3 2 3
deltaz = [0, -1, 1, -2, 2, -3, 3]
elements_up = "RQSPTOU"
elements_dn = "RSQTPUO"
ntuples = int(sys.argv[1])
gridtype = sys.argv[2]
if gridtype == "DIAMOND":
    supergraph = check_representable.build_diamond_lattice()
if gridtype == "HEXAGONAL":
    supergraph = check_representable.build_hexagonal_lattice()

pathcache = build_pathcache(supergraph, ntuples)
if ntuples == 3:
    # triples can only be assesed where all bonds are identical
    lower_pathcache = build_pathcache(supergraph, 2)

printed = []
for line in sys.stdin:
    parts = line.strip().split()
    nv = int(parts[0])
    ne = int(parts[1])
    colors = parts[2 : 2 + nv]
    colors = np.array([int(_) for _ in colors])
    edges = [int(_) for _ in parts[2 + nv :]]

    g = nx.Graph()
    for i in range(nv):
        g.add_node(i)
    for a, b in zip(edges[::2], edges[1::2]):
        g.add_edge(a, b)

    if ntuples == 3:
        eqn = graph_to_equation(supergraph, g, lower_pathcache, 2)
        if eqn != "=":
            # bonds not equal, three-body terms could be anything
            continue
    eqn = graph_to_equation(supergraph, g, pathcache, 3)
    if eqn == "=":
        continue
    if eqn in printed:
        continue
    print(eqn, "#", line.strip())
    printed.append(eqn)
