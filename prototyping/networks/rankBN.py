#!/usr/bin/env python -u
""" Ranks all possible BN-dopings of a molecule."""

import sys
import unittest

from basis_set_exchange import lut
import scipy.spatial.distance as ssd
import MDAnalysis
import math
import igraph as ig
import numpy as np 
import numba

# Ideas:
# - ranking based on nodal structure
# - ranking based on distance argument
# - precompute results of partition

@numba.jit(nopython=True)
def _do_partition(total, maxelements, maxdz):
	if maxelements == 1:
		if total not in (5, 6, 7):
			var = [1]
			return [var[0:0]]
		else:
			return [[total]]
	res = []

	# get range to cover
	first = max(5, 6 - maxdz)
	last = min(7, 6 + maxdz)
	for x in range(first, last + 1):
		limit = maxdz - abs(x - 6)
		if maxelements - 1 < limit:
			continue
		for p in _do_partition(total - x, maxelements - 1, limit):
			if len(p) != 0:
				res.append([x] + p)
	return res

def partition(maxelements, BNcount):
	total = maxelements * 6
	return _do_partition(total, maxelements, BNcount*2)

class Ranker(object):
	""" Ranks BN doped molecules. Ranking in order from lowest to highest."""

	@staticmethod
	def read_xyz(fn):
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
		self._nmodifiedatoms = len(self._includeonly)
		self._natoms = len(self._nuclear_charges)
		self._explain = explain
		self._molecule_similarity_threshold = 0.99999
		self._bonds = MDAnalysis.topology.MOL2Parser.MOL2Parser(mol2file).parse().bonds.values
		self._bondenergies = {(7., 6.): 305./4.184, (7., 7.): 160./4.184, (7., 5.): 115,(6., 6.): 346./4.184, (6., 5.): 356./4.184, (6., 1.): 411/4.184, (5., 1.): 389/4.184, (7., 1.): 386/4.184, (5., 5.): 293/4.184 }
		
		# caching
		self._prepare_getNN()
		self._prepare_site_similarity()
		self._prepare_esp_representation()
		self._prepare_molecule_comparison()

	def rank(self):
		graphs = {}
		for stoichiometry in self._rank_stoichiometries(self._find_stochiometries()):
			if self._explain:
				print ("Working on stoichiometry: %s" % stoichiometry)

			# build clusters of molecules
			self._molecules, npermutations = self._identify_molecules(stoichiometry)
			nmolecules = len(self._molecules)
			if self._explain:
				print ("Found %d molecules from %d permutations." % (nmolecules, npermutations))

			# connect molecules
			graph = ig.Graph(nmolecules)
			for mol_i in range(nmolecules):
				for mol_j in range(mol_i, nmolecules):
					# short-circuit if other relations already exist
					if not math.isinf(graph.shortest_paths(mol_i, mol_j)[0][0]):
						continue

					for origin in self._molecules[mol_i]:
						for opposite in self._molecules[mol_j]:
							reference = (np.array(opposite) + origin) / 2
							common_ground = self._identify_equivalent_sites(reference)
							if self._check_common_ground(origin, opposite, reference, common_ground):
								graph.add_edge(mol_i, mol_j)
								break
						else:
							continue
						break

			# rank components
			mean_bond_energies = []
			components = []
			for component in graph.components():
				components.append(component)
				mean_bond_energies.append(self._mean_bond_energy(component))

			# sort molecules
			if self._explain:
				print (len(components), "components found")
			for component_id in np.argsort(mean_bond_energies):
				print ("Group energy", mean_bond_energies[component_id])
				molecules = [self._molecules[_][0] for _ in components[component_id]]
				NN_energies = [self._getNN(_) for _ in molecules]
				
				for mol_id in np.argsort(NN_energies):
					if self._explain:
						print ('Found: %s' % str(molecules[mol_id]))

	def _identify_molecules(self, stoichiometry):
		permutations = self._find_possible_permutations(stoichiometry)
		molecules = []

		npermutations = len(permutations)
		for permutation in permutations:
			for midx, molecule in enumerate(molecules):
				if self._molecules_similar(molecule[0], permutation):
					molecules[midx].append(permutation)
					break
			else:
				molecules.append([permutation])
		return molecules, npermutations

	def _find_possible_permutations(self, stoichiometry):
		BNcount = len([_ for _ in stoichiometry if _ == 5])
		return partition(self._nmodifiedatoms, BNcount)

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

	def _identify_equivalent_sites(self, reference):
		""" Lists all sites that are sufficiently similar in atomic environment."""
		atomi, atomj, dists = self._get_site_similarity(reference)
		groups = []
		placed = []

		mask = dists < 1
		for i, j, dist in zip(atomi[mask], atomj[mask], dists[mask]):
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
		for isolated in set(self._includeonly) - set(placed):
			groups.append([isolated])
		return groups

	def _prepare_site_similarity(self):
		indices = np.triu_indices(self._nmodifiedatoms, 1)
		self._cache_site_similarity_indices = indices
		self._cache_site_similarity_included_i = self._includeonly[indices[0]]
		self._cache_site_similarity_included_j = self._includeonly[indices[1]]

	def _get_site_similarity(self, nuclear_charges):
		""" Returns i, j, distance."""
		esps = self._get_esp_representation(nuclear_charges)
		
		atomi, atomj = self._cache_site_similarity_indices
		return self._cache_site_similarity_included_i, self._cache_site_similarity_included_j, np.abs(esps[atomi] - esps[atomj])

	def _prepare_esp_representation(self):
		d = ssd.squareform(ssd.pdist(self._coordinates))[:self._nmodifiedatoms, :]
		d[np.diag_indices(self._nmodifiedatoms)] = 1e100
		self._esp_distance_cache = 1/d
		self._esp_cache = np.zeros((self._nmodifiedatoms, self._natoms))

	def _get_esp_representation(self, nuclear_charges):
		charges = self._nuclear_charges.copy()
		charges[:self._nmodifiedatoms] = nuclear_charges
		D = np.outer(nuclear_charges, charges, out=self._esp_cache)
		D *= self._esp_distance_cache
		return np.sum(D, axis=1)

	def _check_common_ground(self, target, opposite, reference, common_ground):
		deltaZ = [a-b for a,b in zip(opposite, target)]

		# matching deltaZ
		changes = np.bincount([_ + 2 for _ in deltaZ], minlength=5)
		
		# ensure matching counts
		if max(changes - changes[::-1]) != 0:
			return False

		# ignore identity operation
		if changes[2] == self._nmodifiedatoms:
			return False

		# all changing atoms need to be in the same group
		assigned = []
		deltaZ = np.array(deltaZ)
		for changepos in (3, 4):
			if changes[changepos] == 0:
				continue

			value = changepos - 2	
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

	def _prepare_molecule_comparison(self):
		# find equivalent sites
		similarities = self._get_site_similarity(np.zeros(self._nmodifiedatoms) + 6)
		g = ig.Graph(self._nmodifiedatoms)

		for a, b, distance in zip(*similarities):
			if distance < 10:
				g.add_edge(a, b)

		self._molecule_comparison_groups = [_ for _ in g.components()]

		# initialize graph
		self._molecule_comparison_graph = ig.Graph(self._natoms + len(self._molecule_comparison_groups))
		for a, b in self._bonds:
			self._molecule_comparison_graph.add_edge(a, b)
		for gidx, group in enumerate(self._molecule_comparison_groups):
			for site in group:
				self._molecule_comparison_graph.add_edge(self._natoms + gidx, site)

		# prepare charge vectors
		self._cache_molecule_color1 = np.append(self._nuclear_charges, np.arange(-len(self._molecule_comparison_groups), 0))
		self._cache_molecule_color2 = np.append(self._nuclear_charges, np.arange(-len(self._molecule_comparison_groups), 0))

	def _molecules_similar(self, c1, c2):
		graph = self._molecule_comparison_graph

		self._cache_molecule_color1[self._includeonly] = c1
		self._cache_molecule_color2[self._includeonly] = c2

		return graph.isomorphic_vf2(graph, color1=self._cache_molecule_color1, color2=self._cache_molecule_color2)

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

		energies = [bond_energy(self._molecules[molid][0]) for molid in component]
		return -sum(energies) / len(energies)

	def _prepare_getNN(self):
		angstrom = 1 / 0.52917721067
		d = ssd.squareform(ssd.pdist(self._coordinates)) * angstrom
		d[np.diag_indices(self._natoms)] = 1e100
		self._cache_NN_distance = 1/d
		self._cache_NN_charges = self._nuclear_charges.copy()
		self._cache_NN_D = np.zeros((self._natoms, self._natoms))

	def _getNN(self, molecule):    
		self._cache_NN_charges[self._includeonly] = molecule
		D = np.outer(self._cache_NN_charges, self._cache_NN_charges, out=self._cache_NN_D)
		D *= self._cache_NN_distance
		return 0.5*np.sum(D)

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


def do_main(fn, mol2file):
	nuclear_charges, coordinates = Ranker.read_xyz(fn)
	r = Ranker(nuclear_charges, coordinates, fn, mol2file, explain=True)
	r.rank()

if __name__ == '__main__':
	# self-test
	#unittest.main(exit=False, verbosity=0)

	fn = sys.argv[1]
	mol2file = sys.argv[2]
	do_main(fn, mol2file)