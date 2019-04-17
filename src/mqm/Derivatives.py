#!/usr/bin/env python
import itertools as it
import os
import glob

import numpy as np
import pyscf
from pyscf import dft
import basis_set_exchange as bse

import mqm.Calculator as mqmc

class Derivatives(object):
	""" Collects common code for derivative implementations."""

	angstrom = 1/0.52917721067

	def calculate_delta_nuc_nuc(self, target):
		natoms = len(self._coordinates)
		ret = 0.
		deltaZ = target - self._nuclear_numbers
		for i in range(natoms):
			for j in range(i + 1, natoms):
				d = np.linalg.norm((self._coordinates[i] - self._coordinates[j])*self.angstrom)
				ret = deltaZ[i]*deltaZ[j]/d
		return ret

	@staticmethod
	def _Z_to_label(Z):
		if Z == 0:
			return '-'
		return bse.lut.element_sym_from_Z(Z, normalize=True)

	def _print_energies(self, targets, energies):
		print ('Energies [Hartree]')
		for target, energy in zip(targets, energies):
			print ('%20.10f %s' % (energy, ' '.join(map(lambda _: Derivatives._Z_to_label(_).ljust(3), target))))


class DerivativeFolders(Derivatives):
	def __init__(self, calculator, highest_order, nuclear_numbers, coordinates, method, basisset):
		self._calculator = calculator
		if highest_order > 2:
			raise NotImplementedError()
		self._orders = list(range(0, highest_order+1))
		self._nuclear_numbers = nuclear_numbers
		self._coordinates = coordinates
		self._basisset = basisset
		self._method = method
		self._reader_cache = dict()

	def _enumerate_all_targets(self):
		""" Builds a list of all integer partitions. """
		def get_partition(protons, sites):
			if sites == 1:
				return [[protons]]
			res = []
			for x in range(protons + 1):
				for p in get_partition(protons - x, sites - 1):
					res.append([x] + p)
			return res

		return get_partition(sum(self._nuclear_numbers), len(self._nuclear_numbers))

	@staticmethod
	def _calculate_delta_Z_vector(numatoms, order, sites, direction):
		baseline = np.zeros(numatoms)

		if order > 0:
			sign = {'up': 1, 'dn': -1}[direction] * 0.05
			baseline[list(sites)] += sign

		return baseline

	def prepare(self):
		""" Builds a complete folder list of all relevant calculations."""
		for order in self._orders:
			# only upper triangle with diagonal
			for combination in it.combinations_with_replacement(list(range(len(self._nuclear_numbers))), order):
				if order > 0:
					label = '-' + '-'.join(map(str, combination))
					directions = ['up', 'dn']
				else:
					directions = ['cc']
					label = '-all'

				for direction in directions:
					path = 'multiqm-run/order-%d/site%s-%s' % (order, label, direction)
					os.makedirs(path, exist_ok=True)

					charges = self._nuclear_numbers + DerivativeFolders._calculate_delta_Z_vector(len(self._nuclear_numbers), order, combination, direction)
					inputfile = self._calculator.get_input(self._coordinates, self._nuclear_numbers, charges, None, self._method, self._basisset)
					with open('%s/run.inp' % path, 'w') as fh:
						fh.write(inputfile)
					with open('%s/run.sh' % path, 'w') as fh:
						fh.write(self._calculator.get_runfile(self._coordinates, self._nuclear_numbers, charges, None, self._method, self._basisset))

	def run(self, remote_host=None):
		""" Executes all calculations if not done so already."""
		# find folders to execute
		folders = [os.path.dirname(_) for _ in glob.glob('multiqm-run/**/run.sh', recursive=True)]

		for folder in folders:
			mqmc.Calculator.execute(folder, remote_host)
			break

		raise NotImplementedError()

		if remote_host is None:
			raise NotImplementedError()
		else:
			pass



	def _cached_reader(self, folder, gridcoords):
		if folder not in self._reader_cache:
			self._reader_cache[folder] = self._calculator.get_density_on_grid(folder, gridcoords)

		return self._reader_cache[folder]

	def _get_grid(self):
		mol = pyscf.gto.Mole()
		for nuclear, coord in zip(self._nuclear_numbers, self._coordinates):
			# pyscf molecule init is in Angstrom
			mol.atom.extend([[nuclear, *coord]])
		mol.build()
		grid = dft.gen_grid.Grids(mol)
		grid.level = 3
		grid.build()
		# pyscf grid is in a.u.
		return grid.coords/self.angstrom, grid.weights

	def analyse(self):
		""" Performs actual analysis and integration. Prints results"""
		targets = self._enumerate_all_targets()
		energies = np.zeros(len(targets))
		natoms = len(self._coordinates)

		# get base information
		gridcoords, gridweights = self._get_grid()
		ds = []
		for atomidx, site in enumerate(self._coordinates):
			ds.append(np.linalg.norm((gridcoords - site)*self.angstrom, axis=1))
		refenergy = self._calculator.get_total_energy('multiqm-run/order-0/site-all-cc')

		# get target predictions
		for targetidx, target in enumerate(targets):
			deltaZ = target - self._nuclear_numbers

			deltaV = np.zeros(len(gridweights))
			for atomidx in range(natoms):
				deltaV += deltaZ[atomidx] / ds[atomidx]

			# zeroth order
			rho = self._cached_reader('multiqm-run/order-0/site-all-cc', gridcoords)
			rhotilde = rho.copy()

			# first order
			for atomidx in range(natoms):
				rhoup = self._cached_reader('multiqm-run/order-1/site-%d-up' % atomidx, gridcoords)
				rhodn = self._cached_reader('multiqm-run/order-1/site-%d-dn' % atomidx, gridcoords)
				deriv = (rhoup - rhodn)/(2*0.05)
				rhotilde += deriv * deltaZ[atomidx] / 2

			# second order
			for i in range(natoms):
				rhoiup = self._cached_reader('multiqm-run/order-1/site-%d-up' % i, gridcoords)
				rhoidn = self._cached_reader('multiqm-run/order-1/site-%d-dn' % i, gridcoords)
				for j in range(natoms):
					rhojup = self._cached_reader('multiqm-run/order-1/site-%d-up' % j, gridcoords)
					rhojdn = self._cached_reader('multiqm-run/order-1/site-%d-dn' % j, gridcoords)
					rhoup = self._cached_reader('multiqm-run/order-2/site-%d-%d-up' % (min(i, j), max(i, j)), gridcoords)
					rhodn = self._cached_reader('multiqm-run/order-2/site-%d-%d-dn' % (min(i, j), max(i, j)), gridcoords)

					if i == j:
						deriv = (rhoup + rhodn - 2 * rho)/(0.05**2)
					else:
						deriv = (rhoup + rhodn + 2 * rho - rhoiup - rhoidn - rhojup - rhojdn) / (2*0.05)

					rhotilde += (deriv * deltaZ[i] * deltaZ[j])/6

			energies[targetidx] = np.sum(rhotilde * deltaV * gridweights) + self.calculate_delta_nuc_nuc(target)

		energies += refenergy

		self._print_energies(targets, energies)
