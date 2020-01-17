#!/usr/bin/env python
""" Ranks all possible BN-dopings of a molecule."""

import numpy as np 
import networkx as nx 
from basis_set_exchange import lut
import sys
import unittest

class Ranker(object):
	""" Ranks BN doped molecules. Ranking in order from lowest to highest."""

	@staticmethod
	def read_xyz(filename):
		""" Extracts nuclear charges and coordinates from an xyz file."""
		with open(fn) as fh:
			lines = fh.readlines()
		numatoms = int(lines[0].strip())
		lines = lines[2 : 2 + numatoms]
		nuclear_numbers = []
		coordinates = []
		for line in lines:
			line = line.strip()
			if len(line) == 0:
				break
			parts = line.split()
			try:
				nuclear_numbers.append(int(parts[0]))
			except:
				nuclear_numbers.append(get_element_number(parts[0]))
			coordinates.append([float(_) for _ in parts[1:4]])
		return np.array(nuclear_numbers), np.array(coordinates)

	def __init__(self, nuclear_charges, coordinates):
		self._graph = nx.Graph()
		self._coordinates = coordinates
		self._nuclear_charges = np.array(nuclear_charges).astype(np.int)
		self._includeonly = np.where(self._nuclear_charges == 6)[0]

	def rank(self):
		pass

	def export(self):
		pass

	def _find_stochiometries(self):
		""" Builds a list of all possible BN-doped stoichiometries, for carbon atoms only."""
		num_carbons = len(self._includeonly)
		stoichiometries = []
		for bnpairs in range(num_carbons // 2 + 1):
			charges = np.zeros(num_carbons, dtype=np.int) + 6
			charges[:bnpairs] = 5
			charges[bnpairs:2*bnpairs] = 7
			stoichiometries.append(charges)
		return stoichiometries

	def _rank_stoichiometries(self, stoichiometries):
		""" Following Daza approach. TODO: citation, explanation, validation"""
		stoichiometries.sort(key=lambda _: len([__ for __ in _ if __ == 6]))
		return stoichiometries


class TestRanker(unittest.TestCase):
	def test_find_stoichiometries(self):
		r = Ranker((6,6,6,6), np.zeros((4,3)))
		expected = [np.array([6,6,6,6]), np.array([5,7,6,6]), np.array([5,5,7,7])]
		actual = r._find_stochiometries()
		self.assertTrue(np.allclose(np.array(expected), np.array(actual)))

	def test_rank_stoichiometries(self):
		r = Ranker((6,6,6,6), np.zeros((4,3)))
		expected = [np.array([5,5,7,7]), np.array([5,7,6,6]), np.array([6,6,6,6])]
		actual = r._rank_stoichiometries(r._find_stochiometries())
		self.assertTrue(np.allclose(np.array(expected), np.array(actual)))


if __name__ == '__main__':
	# self-test
	unittest.main(exit=False, verbosity=0)

	# run analysis
	fn = sys.argv[1]

	nuclear_charges, coordinates = Ranker.read_xyz(fn)
	r = Ranker(nuclear_charges, coordinates, explain=True)
	r.rank()
	r.export(fn + '.ranked')
