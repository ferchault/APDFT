#!/usr/bin/env python -u
""" Ranks all possible BN-dopings of a molecule."""

import itertools as it
import sys
import unittest
import cProfile
import pstats

from basis_set_exchange import lut
import MDAnalysis
import networkx as nx 
import numpy as np 
import qml
import tqdm

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
				nuclear_numbers.append(lut.element_Z_from_sym(parts[0]))
			coordinates.append([float(_) for _ in parts[1:4]])
		return np.array(nuclear_numbers), np.array(coordinates)

	def __init__(self, nuclear_charges, coordinates, filename, mol2file, explain=False):
		self._coordinates = coordinates
		self._nuclear_charges = np.array(nuclear_charges).astype(np.int)
		self._includeonly = np.where(self._nuclear_charges == 6)[0]
		self._explain = explain
		self._c = qml.Compound(filename)
		self._bonds = MDAnalysis.topology.MOL2Parser.MOL2Parser(mol2file).parse().bonds.values
		self._bondenergies = {(7., 6.): 305./4.184, (7., 7.): 160./4.184, (7., 5.): 115,(6., 6.): 346./4.184, (6., 5.): 356./4.184, (6., 1.): 411/4.184, (5., 1.): 389/4.184, (7., 1.): 386/4.184, (5., 5.): 293/4.184 }

	def rank(self):
		graphs = {}
		for stoichiometry in self._rank_stoichiometries(self._find_stochiometries()):
			if self._explain:
				print ("Working on stoichiometry: %s" % stoichiometry)

			graph = nx.Graph()

			# build all possible permutations as molecules
			for target in it.permutations(tuple(stoichiometry)):
				graph.add_node(tuple(target))
			if self._explain:
				print ("Found %d permutations." % len(graph.nodes))

			# find possible relations
			for origin in tqdm.tqdm(list(graph.nodes)):
				for opposite in self._find_opposites(origin, graph.nodes):
					graph.add_edge(origin, opposite)

			# remove spatial duplicates
			self._purge_graph_for_duplicates(graph)
			components = list(nx.connected.connected_components(graph))
			if self._explain:
				ncomponents = len(components)
				print ("Found %d molecules in %d connected components." % (len(graph.nodes), ncomponents))

			# sort molecules
			outgraph = nx.Graph()
			last = 'lowest'
			outgraph.add_node(last)

			components.sort(key=self._mean_bond_energy)
			for component in components:
				molecules = [_ for _ in component]
				molecules.sort(key=self._getNN)
				
				for molecule in molecules:
					if self._explain:
						print ('Found: %s' % str(molecule))
					outgraph.add_edge(last, molecule)
					last = molecule

			outgraph.add_edge(last, 'highest')

			graphs[tuple(stoichiometry)] = outgraph

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

	def _find_opposites(self, origin, candidates):
		""" Tests a list of candidates whether they could be opposites to a given origin."""
		origin = np.array(origin)
    
		found = []
		for opposite in candidates:
			reference = (np.array(opposite) + origin) / 2
			common_ground = self._identify_equivalent_sites(reference)
			if self._check_common_ground(origin, opposite, reference, common_ground):
				found.append(tuple(opposite))
		return found

	def _identify_equivalent_sites(self, reference):
		""" Lists all sites that are sufficiently similar in atomic environment."""
		similarities, relevant = self._get_site_similarity(reference)
		groups = []
		placed = []
		for i, j, dist in similarities:
			if dist > 3:
				continue
			for gidx, group in enumerate(groups):
				if i in group:
					if j not in group:
						groups[gidx].append(j)
						placed.append(j)
					break
				if j in group:
					if i not in group:
						groups[gidx].append(i)
						placed.append(i)
					break
			else:
				groups.append([i,j])
				placed += [i, j]
		for isolated in set(relevant) - set(placed):
			groups.append([isolated])
		return groups

	def _get_site_similarity(self, nuclear_charges):
		""" Returns i, j, distance."""
		charges = self._c.nuclear_charges.copy().astype(np.float)
		charges[self._includeonly] = nuclear_charges
		atoms = np.where(self._c.nuclear_charges == 6)[0]
		a = qml.representations.generate_coulomb_matrix(charges, self._c.coordinates, size=self._c.natoms, sorting='unsorted')
		s = np.zeros((self._c.natoms, self._c.natoms))
		s[np.tril_indices(self._c.natoms)] = a
		d = np.diag(s)
		s += s.T
		s[np.diag_indices(self._c.natoms)] = d
		sorted_elements = [np.sort(_) for _ in s[atoms]]
		ret = []
		for i in range(len(atoms)):
			for j in range(i+1, len(atoms)):
				dist = np.linalg.norm(sorted_elements[i] - sorted_elements[j])
				ret.append([atoms[i], atoms[j], dist])
		return ret, atoms

	def _check_common_ground(self, target, opposite, reference, common_ground):
		deltaZ = opposite - target

		# matching deltaZ
		values, counts = np.unique(deltaZ, return_counts=True)
		counts = dict(zip(values, counts))

		for value in values:
			if value == 0:
				continue
			if -value not in counts:
				return False
			if counts[-value] != counts[value]:
				return False

		# ignore id operation
		if max(np.abs(deltaZ)) == 0:
			return False

		# all changing atoms need to be in the same group
		assigned = []
		for value in values:
			if value <= 0:
				continue
			    
			changed_pos = np.where(deltaZ == value)[0]
			changed_neg = np.where(deltaZ == -value)[0]
			for changed in changed_pos:
				for group in common_ground:
					if changed in group:
						break
				else:
					raise ValueError("should not happen")
				partners = set(changed_neg).intersection(set(group)) - set(assigned)
				if len(partners) == 0:
					return False
				assigned.append(next(iter(partners)))
				assigned.append(changed)
		return True

	def _purge_graph_for_duplicates(self, graph):
		# make connected components smaller
		removed = []
		for component in nx.connected.connected_components(graph):
			kept = []
			for node in tqdm.tqdm(list(component)):
				if len(kept) == 0:
					kept.append(node)
					continue

				for k in kept:
					sim = self._get_molecule_similarity(k, node)
					if sim > 0.99999:
						removed.append((node, k))
						break
				else:
					kept.append(node)
		for node, kept in removed:
			# transfer all connections from node to kept
			for neighbor in graph.neighbors(node):
				graph.add_edge(neighbor, kept)

			# delete node
			graph.remove_node(node)		

	def _get_molecule_similarity(self, c1, c2):
		charges = self._c.nuclear_charges.copy()
		charges[self._includeonly] = c1
		r1 = qml.fchl.generate_representation(self._c.coordinates, charges, self._c.natoms)
		charges = self._c.nuclear_charges.copy()
		charges[self._includeonly] = c2
		r2 = qml.fchl.generate_representation(self._c.coordinates, charges, self._c.natoms)
		return qml.fchl.get_global_kernels(np.array([r1]), np.array([r2]), np.array([2])).flatten()[0]

	def _mean_bond_energy(self, component):
		def bond_energy(molecule):
			energy = 0
			charges = self._nuclear_charges.copy()
			charges[self._includeonly] = molecule
			for bond in self._bonds:
				z1, z2 = charges[bond[0]], charges[bond[1]]
				if z2 > z1:
					z1, z2 = z2, z1
				energy += self._bondenergies[(z1, z2)]
			return energy

		energies = [bond_energy(_) for _ in component]
		return -sum(energies) / len(energies)

	def _getNN(self, molecule):
		def nuclei_nuclei(coordinates, charges):
			angstrom = 1 / 0.52917721067
			natoms = len(coordinates)
			ret = 0.0
			for i in range(natoms):
				for j in range(i + 1, natoms):
					d = np.linalg.norm((coordinates[i] - coordinates[j]) * angstrom)
					ret += charges[i] * charges[j] / d
			return ret
    
		charges = self._c.nuclear_charges.copy()
		charges[self._includeonly] = molecule
		return nuclei_nuclei(self._c.coordinates, charges)

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
	#unittest.main(exit=False, verbosity=0)

	# run analysis
	fn = sys.argv[1]
	mol2file = sys.argv[2]

	# setup profiling
	pr = cProfile.Profile()
	pr.enable()

	# do work
	nuclear_charges, coordinates = Ranker.read_xyz(fn)
	r = Ranker(nuclear_charges, coordinates, fn, mol2file, explain=True)
	r.rank()

	# print profiling
	pr.disable()
	stats = pstats.Stats(pr)
	stats.sort_stats('cumulative')
	stats.print_stats(30)
