#!/usr/bin/env python

import random
import os
import string

import jinja2 as j

class Calculator(object):
	""" A concurrency-safe blocking interface for an external QM code."""
	_methods = {}

	def __init__(self, method, basisset, superimpose=False):
		self._method = method
		self._basisset = basisset
		self._superimpose = superimpose

	def get_methods(self):
		return list(self._methods.keys())

	def get_density_on_grid(self, folder, gridpoints):
		raise NotImplementedError()

	@staticmethod
	def _get_tempname():
		return 'apdft-tmp-' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

	@staticmethod
	def get_grid(nuclear_numbers, coordinates, outputfolder):
		""" Returns the integration grid used by this calculator for a given set of nuclei and geometry.

		Grid weights and coordinates may be in internal units. Return value should be coords, weights. If return value is None, a default grid is used."""
		return None

class MockCalculator(Calculator):
	_methods = {}
	@classmethod
	def get_runfile(self, coordinates, nuclear_numbers, nuclear_charges, grid):
		basedir = os.path.dirname(os.path.abspath(__file__))
		with open('%s/templates/mock-run.sh' % basedir) as fh:
			template = j.Template(fh.read())
		return template.render()
