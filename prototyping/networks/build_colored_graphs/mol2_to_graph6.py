#!/usr/bin/env python -u

import MDAnalysis
import networkx as nx
import sys

mol2file = sys.argv[1]
outfile = sys.argv[2]
bonds = MDAnalysis.topology.MOL2Parser.MOL2Parser(mol2file).parse().bonds.values
elements = MDAnalysis.topology.MOL2Parser.MOL2Parser(mol2file).parse().names.values

graph = nx.Graph()
for idx, element in enumerate(elements):
	if element == 'C':
		graph.add_node(idx)
for a, b in bonds:
	if elements[a] == 'C' and elements[b] == 'C':
		print (a, b)
		graph.add_edge(a, b)
nx.write_graph6(graph, outfile)
