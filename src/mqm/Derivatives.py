#!/usr/bin/env python
import itertools as it
import os
import glob
import multiprocessing as mp
import functools
import traceback

import numpy as np
import pyscf
from pyscf import dft
import basis_set_exchange as bse

import mqm
import mqm.Calculator as mqmc
import mqm.math

class Derivatives(object):
	""" Collects common code for derivative implementations."""

	angstrom = 1/0.52917721067

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

	def calculate_delta_nuc_nuc(self, target):
		natoms = len(self._coordinates)
		ret = 0.
		deltaZ = target - self._nuclear_numbers
		for i in range(natoms):
			for j in range(i + 1, natoms):
				d = np.linalg.norm((self._coordinates[i] - self._coordinates[j])*self.angstrom)
				ret = deltaZ[i]*deltaZ[j]/d
		return ret

	def calulate_nuclear_dipole(self, nuclear_charges):
		""" Calculates the nuclear dipole moment w.r.t. the origin."""
		return np.sum(self._coordinates.T * nuclear_charges, axis=1)

	@staticmethod
	def _Z_to_label(Z):
		if Z == 0:
			return '-'
		return bse.lut.element_sym_from_Z(Z, normalize=True)

	def _print_energies(self, targets, energies, comparison_energies):
		if comparison_energies is None:
			for target, energy in zip(targets, energies):
				targetname = ','.join([Derivatives._Z_to_label(_) for _ in target])
				mqm.log.log('Energy calculated', level='RESULT', value=energy, kind='total_energy', target=target, targetname=targetname)
		else:
			for target, energy, comparison in zip(targets, energies, comparison_energies):
				targetname = ','.join([Derivatives._Z_to_label(_) for _ in target])
				mqm.log.log('Energy calculated', level='RESULT', value=energy, kind='total_energy', target=target, targetname=targetname, reference=comparison, error=energy - comparison)

	def _print_dipoles(self, targets, dipoles, comparison_dipoles):
		if comparison_dipoles is not None:
			for target, dipole, comparison in zip(targets, dipoles, comparison_dipoles):
				targetname = ','.join([Derivatives._Z_to_label(_) for _ in target])
				mqm.log.log('Dipole calculated',
					level='RESULT',
					kind='total_dipole',
					reference=list(comparison),
					value=list(dipole),
					target=target,
					targetname=targetname
				)

	def _get_grid(self):
		""" Returns the integration grid in Angstrom."""
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

	def _enumerate_all_targets(self):
		""" Builds a list of all possible targets.

		Note that the order is not guaranteed to be stable.

		Args:
			self:		Class instance from which the total charge and numebr of sites is determined.
		Returns:
			A list of lists with the integer nuclear charges."""
		return mqm.math.IntegerPartitions.partition(sum(self._nuclear_numbers), len(self._nuclear_numbers))


class DerivativeFolders(Derivatives):
	@staticmethod
	def _calculate_delta_Z_vector(numatoms, order, sites, direction):
		baseline = np.zeros(numatoms)

		if order > 0:
			sign = {'up': 1, 'dn': -1}[direction] * 0.05
			baseline[list(sites)] += sign

		return baseline

	def prepare(self, explicit_reference=False):
		""" Builds a complete folder list of all relevant calculations."""
		if os.path.isdir('multiqm-run'):
			mqm.log.log('Project folder exists. Reusing existing data.', level='warning')
			return

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
		if explicit_reference:
			for target in self._enumerate_all_targets():
				path = 'multiqm-run/comparison-%s' % ('-'.join(map(str, target)))
				os.makedirs(path, exist_ok=True)

				inputfile = self._calculator.get_input(self._coordinates, self._nuclear_numbers, target, None, self._method, self._basisset)
				with open('%s/run.inp' % path, 'w') as fh:
					fh.write(inputfile)
				with open('%s/run.sh' % path, 'w') as fh:
					fh.write(self._calculator.get_runfile(self._coordinates, self._nuclear_numbers, target, None, self._method, self._basisset))

	@staticmethod
	def _wrapper(_, remote_host, remote_preload):
		try:
			mqmc.Calculator.execute(_, remote_host, remote_preload)
		except Exception as e:
			return traceback.format_exc()

	def run(self, parallel=None, remote_host=None, remote_preload=None):
		""" Executes all calculations if not done so already."""

		# Obtain number of parallel executions
		if parallel is None and remote_host is not None:
			raise NotImplementedError('Remote parallelisation only with explicit process count.')

		if parallel == 0:
			parallel = mp.cpu_count()
		if parallel is None:
			parallel = 1

		# find folders to execute
		folders = [os.path.dirname(_) for _ in glob.glob('multiqm-run/**/run.sh', recursive=True)]
		haslog = [os.path.dirname(_) for _ in glob.glob('multiqm-run/**/run.log', recursive=True)]
		folders = set(folders) - set(haslog)

		with mp.Pool(parallel) as pool:
			results = pool.map(functools.partial(DerivativeFolders._wrapper, remote_host=remote_host, remote_preload=remote_preload), folders)

		failed = [_ for _ in results if _ is not None]
		return len(failed) == 0

	def _cached_reader(self, folder, gridcoords, gridweights, num_electrons):
		if folder not in self._reader_cache:
			rho = self._calculator.get_density_on_grid(folder, gridcoords)
			self._reader_cache[folder] = rho
			int_electrons = np.sum(rho * gridweights)
			if abs(num_electrons - int_electrons) > 1e-4:
				mqm.log.log('Electron count mismatch.', level='error', expected=num_electrons, found=int_electrons, path=folder)

		return self._reader_cache[folder]

	def analyse(self, explicit_reference=False):
		""" Performs actual analysis and integration. Prints results"""
		targets = self._enumerate_all_targets()
		energies = np.zeros(len(targets))
		dipoles = np.zeros((len(targets), 3))
		comparison_energies = np.zeros(len(targets))
		comparison_dipoles = np.zeros((len(targets), 3))
		natoms = len(self._coordinates)

		num_electrons = np.sum(self._nuclear_numbers)

		# get base information
		gridcoords, gridweights = self._get_grid()
		grid_ds = np.linalg.norm(gridcoords * self.angstrom, axis=1)
		ds = []
		for site in self._coordinates:
			ds.append(np.linalg.norm((gridcoords - site)*self.angstrom, axis=1))
		refenergy = self._calculator.get_total_energy('multiqm-run/order-0/site-all-cc')

		# get target predictions
		for targetidx, target in enumerate(targets):
			deltaZ = target - self._nuclear_numbers

			deltaV = np.zeros(len(gridweights))
			for atomidx in range(natoms):
				deltaV += deltaZ[atomidx] / ds[atomidx]

			# zeroth order
			rho = self._cached_reader('multiqm-run/order-0/site-all-cc', gridcoords, gridweights, num_electrons)
			rhotilde = rho.copy()
			rhotarget = rho.copy()

			# first order
			for atomidx in range(natoms):
				rhoup = self._cached_reader('multiqm-run/order-1/site-%d-up' % atomidx, gridcoords, gridweights, num_electrons)
				rhodn = self._cached_reader('multiqm-run/order-1/site-%d-dn' % atomidx, gridcoords, gridweights, num_electrons)
				deriv = (rhoup - rhodn)/(2*0.05)
				rhotilde += deriv * deltaZ[atomidx] / 2
				rhotarget += deriv * deltaZ[atomidx]

			# second order
			for i in range(natoms):
				rhoiup = self._cached_reader('multiqm-run/order-1/site-%d-up' % i, gridcoords, gridweights, num_electrons)
				rhoidn = self._cached_reader('multiqm-run/order-1/site-%d-dn' % i, gridcoords, gridweights, num_electrons)
				for j in range(natoms):
					rhojup = self._cached_reader('multiqm-run/order-1/site-%d-up' % j, gridcoords, gridweights, num_electrons)
					rhojdn = self._cached_reader('multiqm-run/order-1/site-%d-dn' % j, gridcoords, gridweights, num_electrons)
					rhoup = self._cached_reader('multiqm-run/order-2/site-%d-%d-up' % (min(i, j), max(i, j)), gridcoords, gridweights, num_electrons)
					rhodn = self._cached_reader('multiqm-run/order-2/site-%d-%d-dn' % (min(i, j), max(i, j)), gridcoords, gridweights, num_electrons)

					if i == j:
						deriv = (rhoiup + rhoidn - 2 * rho)/(0.05**2)
					else:
						deriv = (rhoup + rhodn + 2 * rho - rhoiup - rhoidn - rhojup - rhojdn) / (2*0.05**2)

					rhotilde += (deriv * deltaZ[i] * deltaZ[j])/6
					rhotarget += (deriv * deltaZ[i] * deltaZ[j])/2

			energies[targetidx] = -np.sum(rhotilde * deltaV * gridweights) + self.calculate_delta_nuc_nuc(target)
			dipoles[targetidx] = -np.sum(gridcoords.T * rhotarget * gridweights, axis=1) + self.calulate_nuclear_dipole(target)

		# optional comparison to true properties
		if explicit_reference:
			for targetidx, target in enumerate(targets):
				path = 'multiqm-run/comparison-%s' % ('-'.join(map(str, target)))
				comparison_energies[targetidx] = self._calculator.get_total_energy(path)
				comparison_dipoles[targetidx] = self._calculator.get_electronic_dipole(path, gridcoords, gridweights) + self.calulate_nuclear_dipole(target)
		else:
			comparison_energies = None

		energies += refenergy

		self._print_energies(targets, energies, comparison_energies)
		self._print_dipoles(targets, dipoles, comparison_dipoles)

		return targets, energies, comparison_energies
