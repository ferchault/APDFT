#!/usr/bin/env python -u
""" Ranks all possible BN-dopings of molecule 12 in the reference data set."""

import sys
import time
import unittest

from basis_set_exchange import lut
import scipy.spatial.distance as ssd
import MDAnalysis
import math
import igraph as ig
import numpy as np 
import numba
import pymatgen
import pymatgen.io.xyz

import qml

# Ideas:
# - ranking based on nodal structure
# - ranking based on distance argument
# - precompute results of partition

@numba.jit(nopython=True)
def numba_exit_norm(A, B, n, limit):
	delta = 0.
	for i in range(n):
		delta += (A[i] - B[i])**2
		if delta > limit:
			return limit + 1
	return delta**0.5

@numba.jit(nopython=True)
def numba_loop(atomi, atomj, sorted_elements, natoms, limit):
	npairs = len(atomi)
	ret = np.zeros(npairs)
	for i in range(npairs):
		dist = numba_exit_norm(sorted_elements[atomi[i]], sorted_elements[atomj[i]], natoms, limit)
		ret[i] = dist
	return ret

@numba.jit(nopython=True)
def _precheck(target, opposite):	
	deltaZ = opposite - target

	# matching deltaZ
	changes = np.zeros(5, dtype=np.int32)
	for i in deltaZ:
		changes[i +2] +=1 

	# ensure matching counts
	if max(changes - changes[::-1]) != 0:
		return False

	# ignore identity operation
	if changes[2] == 10:
		return False

	return True

def detect_automorphisms(filename):
	xyz = pymatgen.io.xyz.XYZ.from_file(filename)
	psa = pymatgen.symmetry.analyzer.PointGroupAnalyzer(xyz.molecule)

	m = xyz.molecule.get_centered_molecule()
	carbons = np.where(np.array(m.atomic_numbers, dtype=np.int) == 6)[0]

	operations = psa.get_symmetry_operations()
	mapping = np.zeros((len(carbons), len(operations)), dtype=np.int)
	for opidx, op in enumerate(operations):
		for bidx, base in enumerate(carbons):
			ds = np.linalg.norm(op.operate(m.cart_coords[base]) - m.cart_coords[carbons], axis=1)
			onto = np.argmin(ds)
			if ds[onto] > 1e-3:
				raise ValueError('Irregular geometry')
			mapping[bidx, opidx] = onto

	return mapping

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

	def __init__(self, nuclear_charges, coordinates, filename, mol2file, explain=False, sim=1, simmode=None):
		self._similarity_parameter = sim
		
		if simmode == "ESP":
			self._get_site_similarity = self._get_site_similarity_ESP
		if simmode == "CM":
			self._get_site_similarity = self._get_site_similarity_CM

		self._coordinates = coordinates
		self._c = qml.Compound(filename)
		self._nuclear_charges = np.array(nuclear_charges).astype(np.int)
		self._includeonly = np.where(self._nuclear_charges == 6)[0]
		self._nmodifiedatoms = len(self._includeonly)
		self._natoms = len(self._nuclear_charges)
		self._explain = explain
		self._bonds = MDAnalysis.topology.MOL2Parser.MOL2Parser(mol2file).parse().bonds.values
		#self._bondenergies = {(7., 6.): 305./4.184, (7., 7.): 160./4.184, (7., 5.): 115,(6., 6.): 346./4.184, (6., 5.): 356./4.184, (6., 1.): 411/4.184, (5., 1.): 389/4.184, (7., 1.): 386/4.184, (5., 5.): 293/4.184 }
		#self._bondenergies = {(7., 6.): 4.54016298, (7., 7.): 3.02677532, (7., 5.): 3.37601863, (6., 6.): 6.05355064, (6., 5.): 4.88940629, (6., 1.): 6.40279395, (5., 1.): 5.23864959, (7., 1.): 6.98486612, (5., 5.): 3.72526193}
		
		# caching
		self._prepare_getNN()
		self._prepare_site_similarity()
		self._prepare_esp_representation()
		self._prepare_precheck()

		# debug
		self._automorphism_cache = [[0,1,2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [13,14,15,10,11,12,9,8,7,6,3,4,5,0,1,2,21,20,19,18,17,16]]

	def rank(self, candidates):
		nmolecules = len(self._molecules)
		if self._explain:
			print ("#Read %d molecules." % (nmolecules))

		bnn = np.zeros(nmolecules)
		for idx in range(nmolecules):
			bnn[idx] = self._getNN(self._molecules[idx])

		graph = ig.Graph(nmolecules)
		graph.add_edges(candidates)

		components = []
		for component in graph.components():
			components.append(component)

		if self._explain:
			print ("# Found %d components" % len(components))

		# build bondenergies
		sizes = []
		coefficients = []
		bondorder = ['BB', 'BC', 'BH', 'BN', 'CC', 'CH', 'CN', 'HN', 'NN']
		for group in components:
			if len(group) < 2:
				continue

			this = self._solve_group(group, bondorder)
			if this is not None and (this>0).all():
				sizes.append(len(group))
            
				# only insert if no duplicate
				for e in coefficients:
					delta = np.linalg.norm(e-this)
					if delta < 1e-10:
						break
				else:
					coefficients.append(this)
		#coeff = -np.array(coefficients).mean(axis=0)
		coeff = [-0.122015 ,  -0.44260741, -0.24407662 ,-0.24389064, -3.71681812 ,-2.25259099, -0.44260741, -0.24407662, -0.122015  ]
		#coeff = -np.array([293/4.184 , 356./4.184, 389/4.184, 115,  346./4.184, 411/4.184, 305./4.184 ,386/4.184, 160./4.184])

		group_energies = []
		group_nn_energies = []
		for component in components:
			Atmp = self._build_mat(component, bondorder)
			group_energies.append(np.dot(Atmp, coeff).mean())

			nn = np.array([self._getNN(self._molecules[_]) for _ in component]).mean()
			group_nn_energies.append(nn)

		reranked = []
		for component_id in np.lexsort((group_nn_energies, group_energies)):
			component = components[component_id]
			nn = np.array([self._getNN(self._molecules[_]) for _ in component])
			
			for mol_id in np.argsort(nn):
				reranked.append(component[mol_id])

		mixed = np.array(reranked).astype(np.int32)
		mixed.tofile('ranking')

		# mix in NN ranking to ensure stability for few groups with valid regressions
		rankNN = np.argsort(bnn)
		newrank = np.array(reranked)
		N = max(1, len(coefficients))
		mixed = np.argsort(rankNN/N + newrank * N)


	def _bond_count(self, label):
		bonds = {'BH': 0, 'CH': 0, 'HN': 0, 'BB': 0, 'BC': 0, 'BN': 0, 'CC': 0, 'CN': 0, 'NN': 0, }
		infile = self._bonds
		chars = '01234BCN'
		for valididx in (0,1,21,20,17,16,14,13,12,11,8,7,4,5):
			k = ''.join(sorted([chars[label[valididx]], "H"]))
			bonds[k] += 1

		for a,b in infile:
			if a > 21 or b > 21:
				continue
			k = ''.join(sorted([chars[label[_]] for _ in (a, b)]))
			bonds[k] += 1
		return bonds

	def _build_mat(self, group, bondorder):
		A = []
		for label in group:
			counts = self._bond_count(self._molecules[label])
			A.append([counts[_] for _ in bondorder])
		return np.array(A)

	def _solve_group(self, group, bondorder):
		A = self._build_mat(group, bondorder)
		if 0 in np.sum(A, axis=0):
			return None
		coeffs = np.linalg.lstsq(A, np.zeros(len(A))+100, rcond=None)[0]
		return coeffs

	def _identify_molecules(self, stoichiometry):
		nbn = len([_ for _ in stoichiometry if _ == 5])
		molecules = np.load("out-nbn-%d.npy" % nbn)
		return molecules

	def _find_possible_permutations(self, stoichiometry):
		BNcount = len([_ for _ in stoichiometry if _ == 5])
		return partition(self._nmodifiedatoms, BNcount)

	def _find_stochiometries(self):
		""" Builds a list of all possible BN-doped stoichiometries, for carbon atoms only."""
		num_carbons = len(self._includeonly)
		stoichiometries = []
		for bnpairs in range(num_carbons // 2 + 1):
		#for bnpairs in (5,):
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
		dists = self._get_site_similarity(reference)
		groups = []
		placed = []

		atomi, atomj = self._cache_site_similarity_indices
		mask = dists < self._similarity_parameter
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
		self._sitesimCM_squareform_outcache = np.zeros((self._c.natoms, self._c.natoms))
		self._sitesimCM_squareform_mask = np.tril(np.ones((self._c.natoms, self._c.natoms),dtype=bool))
		self._sitesimCM_charges = self._c.nuclear_charges.copy().astype(np.float)

	def _get_site_similarity_ESP(self, nuclear_charges):
		""" Returns i, j, distance."""
		esps = self._get_esp_representation(nuclear_charges)
		
		atomi, atomj = self._cache_site_similarity_indices
		return self._cache_site_similarity_included_i, self._cache_site_similarity_included_j, np.abs(esps[atomi] - esps[atomj])

	def _get_site_similarity_CM(self, nuclear_charges):
		""" Returns i, j, distance."""
		charges = self._sitesimCM_charges
		charges[self._includeonly] = nuclear_charges
		a = qml.representations.generate_coulomb_matrix(charges, self._c.coordinates, size=self._c.natoms, sorting='unsorted')

		# to squareform
		self._sitesimCM_squareform_outcache[self._sitesimCM_squareform_mask] = a
		self._sitesimCM_squareform_outcache.T[self._sitesimCM_squareform_mask] = a

		sorted_elements = np.sort(self._sitesimCM_squareform_outcache[self._includeonly], axis=1, kind="stable")

		limit = self._similarity_parameter**2.
		atomi, atomj = self._cache_site_similarity_indices

		return numba_loop(atomi, atomj, sorted_elements, self._c.natoms, limit)

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

	def _prepare_precheck(self):
		# we don't actually need the iteration over groups to be dynamic
		# numba cannot unroll this loop (or I don't know how to tell it to) since it does not know that these lists are static
		# typed lists are faster but not as fast, so we want to unroll the loop
		# ugly workaround: generate the code here, jit it and (re-)place the class method

		code = []
		for group in self._identify_equivalent_sites([6] * self._nmodifiedatoms):
			code.append('(' + ' + '.join(['opposite[%d] - target[%s]' % (_, _) for _ in group]) + ') == 0')
		code = ' and '.join(code)
		code = f'lambda target, opposite: True if {code} else False'
		self._group_precheck = numba.jit(nopython=True)(eval(code))


	def _check_common_ground(self, deltaZ, changes, common_ground):
		# all changing atoms need to be in the same group
		assigned = []
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

		energies = [bond_energy(self._molecules[molid]) for molid in component]
		return -sum(energies) / len(energies)

	def _mean_nn_energy(self, component):
		energies = [self._getNN(self._molecules[molid]) for molid in component]
		return sum(energies) / len(energies)

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


def do_main(fn, mol2file, similarity, similarity_mode, candidates,mollist):
	nuclear_charges, coordinates = Ranker.read_xyz(fn)
	r = Ranker(nuclear_charges, coordinates, fn, mol2file, explain=True, sim=similarity, simmode=similarity_mode)
	r._molecules = np.fromfile(mollist, dtype=np.int8).reshape(-1, 22)
	r._molecules += 6
	r._molecules[r._molecules == 8] = 5
	begin = time.time()
	r.rank(candidates)
	end = time.time()
	print ("# done, %d molecules in %ds, %f cmp/s" % (len(candidates), (end-begin), len(candidates)/(end-begin)))

def read_prescan(fn):
	return np.loadtxt(fn, dtype=np.int)

if __name__ == '__main__':
	# self-test
	#unittest.main(exit=False, verbosity=0)

	fn = sys.argv[1]
	mol2file = sys.argv[2]
	similarity = 2.2
	similarity_mode = "CM"
	mollist = sys.argv[3]
	candidates = read_prescan(sys.argv[4])
	do_main(fn, mol2file, similarity, similarity_mode, candidates, mollist)
