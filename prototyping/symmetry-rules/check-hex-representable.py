#!/usr/bin/env python
import networkx as nx
import sys

infile = sys.argv[1]
def is_hex_representable(graph):
	supergraph = nx.generators.lattice.hexagonal_lattice_graph(10, 10)
	matcher = nx.algorithms.isomorphism.GraphMatcher(supergraph, graph)
	return matcher.subgraph_is_isomorphic()

def test():
	g = nx.Graph()
	g.add_edge(0,1)
	g.add_edge(1,2)
	g.add_edge(2,3)
	print (is_hex_representable(g))
	g = nx.Graph()
	g.add_edge(0,1)
	g.add_edge(1,2)
	g.add_edge(2,0)
	print (is_hex_representable(g))

with open(infile) as fh:
	for line in fh:
		line = line.strip()
		graph = nx.from_graph6_bytes(line.encode("ascii"))
		if is_hex_representable(graph):
			print (line)
