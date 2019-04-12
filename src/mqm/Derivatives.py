#!/usr/bin/env python
import itertools as it
import os

class DerivativeFolders(object):
	def __init__(self, calculator, highest_order, nuclear_numbers, coordinates, method, basisset):
		self._calculator = calculator
		if highest_order > 2:
			raise NotImplementedError()
		self._orders = list(range(0, highest_order+1))
		self._nuclear_numbers = nuclear_numbers
		self._coordinates = coordinates
		self._basisset = basisset
		self._method = method

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
			sign = {'up': 1, 'dn': -1}[direction]
			baseline[sites] += sign

		return baseline

	def prepare(self):
		""" Builds a complete folder list of all relevant calculations."""
		for order in self._orders:
			# only upper triangle with diagonal
			for combination in it.combinations_with_replacement((1, 2, 3, 4), order):
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


	def run(self):
		""" Executes all calculations in the current folder if not done so already."""
		pass

	def analyse(self):
		""" Performs actual analysis and integration. Prints results"""
		targets = self._enumerate_all_targets()
		pass
