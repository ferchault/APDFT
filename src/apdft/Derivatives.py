#!/usr/bin/env python
import itertools as it
import os
import glob
import multiprocessing as mp
import functools
import traceback

import numpy as np
import basis_set_exchange as bse

import apdft
import apdft.Calculator as apc
import apdft.math
import apdft.physics

class DerivativeFolders(apdft.physics.APDFT):
	def assign_calculator(self, calculator):
		self._calculator = calculator

	def _print_energies(self, targets, energies, comparison_energies):
		if comparison_energies is None:
			for target, energy in zip(targets, energies):
				targetname = ','.join([apdft.physics.charge_to_label(_) for _ in target])
				apdft.log.log('Energy calculated', level='RESULT', value=energy, kind='total_energy', target=target, targetname=targetname)
		else:
			for target, energy, comparison in zip(targets, energies, comparison_energies):
				targetname = ','.join([apdft.physics.charge_to_label(_) for _ in target])
				apdft.log.log('Energy calculated', level='RESULT', value=energy, kind='total_energy', target=target, targetname=targetname, reference=comparison, error=energy - comparison)

	def _print_dipoles(self, targets, dipoles, comparison_dipoles):
		if comparison_dipoles is not None:
			for target, dipole, comparison in zip(targets, dipoles, comparison_dipoles):
				targetname = ','.join([apdft.physics.charge_to_label(_) for _ in target])
				apdft.log.log('Dipole calculated',
					level='RESULT',
					kind='total_dipole',
					reference=list(comparison),
					value=list(dipole),
					target=target,
					targetname=targetname
				)

	def _calculate_delta_Z_vector(self, numatoms, order, sites, direction):
		baseline = np.zeros(numatoms)

		if order > 0:
			sign = {'up': 1, 'dn': -1}[direction] * self._delta
			baseline[list(sites)] += sign

		return baseline

	def prepare(self, explicit_reference=False):
		""" Builds a complete folder list of all relevant calculations."""
		if os.path.isdir('multiqm-run'):
			apdft.log.log('Project folder exists. Reusing existing data.', level='warning')
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

					charges = self._nuclear_numbers + self._calculate_delta_Z_vector(len(self._nuclear_numbers), order, combination, direction)
					inputfile = self._calculator.get_input(self._coordinates, self._nuclear_numbers, charges, None)
					with open('%s/run.inp' % path, 'w') as fh:
						fh.write(inputfile)
					with open('%s/run.sh' % path, 'w') as fh:
						fh.write(self._calculator.get_runfile(self._coordinates, self._nuclear_numbers, charges, None))
		if explicit_reference:
			targets = self.enumerate_all_targets()
			apdft.log.log('All targets listed for comparison run.', level='info', count=len(targets))
			for target in targets:
				path = 'multiqm-run/comparison-%s' % ('-'.join(map(str, target)))
				os.makedirs(path, exist_ok=True)

				inputfile = self._calculator.get_input(self._coordinates, self._nuclear_numbers, target, None)
				with open('%s/run.inp' % path, 'w') as fh:
					fh.write(inputfile)
				with open('%s/run.sh' % path, 'w') as fh:
					fh.write(self._calculator.get_runfile(self._coordinates, self._nuclear_numbers, target, None))

	@staticmethod
	def _wrapper(_, remote_host, remote_preload):
		try:
			apc.Calculator.execute(_, remote_host, remote_preload)
		except Exception as e:
			return _, traceback.format_exc()

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
		for fail in failed:
			apdft.log.log('Calculation failed.', level='error', folder=fail[0])
		return len(failed) == 0

	def _cached_reader(self, folder):
		if folder not in self._reader_cache:
			rho = self._calculator.get_density_on_grid(folder, self._gridcoords)
			self._reader_cache[folder] = rho
			int_electrons = np.sum(rho * self._gridweights)
			num_electrons = sum(self._nuclear_numbers)
			if abs(num_electrons - int_electrons) > 1e-2:
				apdft.log.log('Electron count mismatch.', level='error', expected=num_electrons, found=int_electrons, path=folder)

		return self._reader_cache[folder]

	def get_density_derivative(self, sites):
		num_electrons = sum(self._nuclear_numbers)
		if len(sites) == 0:
			return self._cached_reader('multiqm-run/order-0/site-all-cc')
		if len(sites) == 1:
			rhoup = self._cached_reader('multiqm-run/order-1/site-%d-up' % sites[0])
			rhodn = self._cached_reader('multiqm-run/order-1/site-%d-dn' % sites[0])
			return (rhoup - rhodn) / (2 * self._delta)
		if len(sites) == 2:
			i, j = sites
			rho = self.get_density_derivative([])
			rhoiup = self._cached_reader('multiqm-run/order-1/site-%d-up' % i)
			rhoidn = self._cached_reader('multiqm-run/order-1/site-%d-dn' % i)
			rhojup = self._cached_reader('multiqm-run/order-1/site-%d-up' % j)
			rhojdn = self._cached_reader('multiqm-run/order-1/site-%d-dn' % j)
			rhoup = self._cached_reader('multiqm-run/order-2/site-%d-%d-up' % (min(i, j), max(i, j)))
			rhodn = self._cached_reader('multiqm-run/order-2/site-%d-%d-dn' % (min(i, j), max(i, j)))

			if i == j:
				deriv = (rhoiup + rhoidn - 2 * rho)/(self._delta**2)
			else:
				deriv = (rhoup + rhodn + 2 * rho - rhoiup - rhoidn - rhojup - rhojdn) / (2 * self._delta**2)
			return deriv
		raise NotImplementedError()

	def get_density_from_reference(self, nuclear_charges):
		return self._cached_reader('multiqm-run/comparison-%s' % ('-'.join(map(str, nuclear_charges))))

	def get_energy_from_reference(self, nuclear_charges):
		return self._calculator.get_total_energy('multiqm-run/comparison-%s' % ('-'.join(map(str, nuclear_charges))))

	def analyse(self, explicit_reference=False, do_energies=True, do_dipoles=True):
		""" Performs actual analysis and integration. Prints results"""
		targets, energies, dipoles = self.predict_all_targets(do_energies, do_dipoles)

		if explicit_reference:
			comparison_energies = np.zeros(len(targets))
			comparison_dipoles = np.zeros((len(targets), 3))
			for targetidx, target in enumerate(targets):
				path = 'multiqm-run/comparison-%s' % ('-'.join(map(str, target)))
				comparison_energies[targetidx] = self._calculator.get_total_energy(path)

				rho = self._cached_reader(path)
				nd = apdft.physics.Dipoles.point_charges([0, 0, 0], self._coordinates, target)
				ed = apdft.physics.Dipoles.electron_density([0, 0, 0], self._gridcoords, rho * self._gridweights)
				comparison_dipoles[targetidx] = ed + nd
		else:
			comparison_energies = None
			comparison_dipoles = None

		self._print_energies(targets, energies, comparison_energies)
		self._print_dipoles(targets, dipoles, comparison_dipoles)

		return targets, energies, comparison_energies
