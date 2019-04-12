#!/usr/bin/env python

class DerivativeFolders(object):
	def __init__(self, calculator, highest_order, nuclear_numbers, coordinates):
		self._calculator = calculator
		self._orders = list(range(0, highest_order+1))
		self._nuclear_numbers = nuclear_numbers
		self._coordinates = coordinates

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

	def prepare(self):
		""" Builds a complete folder list of all relevant calculations."""
		targets = self._enumerate_all_targets()

	def run(self):
		""" Executes all calculations in the current folder if not done so already."""
		pass

	def analyse(self):
		""" Performs actual analysis and integration. Prints results"""
		pass
