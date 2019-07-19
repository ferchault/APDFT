#!/usr/bin/env python
from apdft import log
from apdft import calculator

import os
import sys
import numpy as np
import basis_set_exchange as bse
import jinja2 as j
import cclib

# load local orbkit
basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append('%s/../../../dep/orbkit/orbkit' % basedir)
import orbkit

class GaussianCalculator(calculator.Calculator):
	_methods = {
		'CCSD': 'CCSD(Full,MaxCyc=100)',
		'PBE0': 'PBE1PBE',
		'PBE': 'PBEPBE',
		'HF': 'UHF',
	}

	@staticmethod
	def _format_coordinates(nuclear_numbers, coordinates):
		ret = ''
		for Z, coords in zip(nuclear_numbers, coordinates):
			ret += '%d %f %f %f\n' % (Z, coords[0], coords[1], coords[2])
		return ret[:-1]

	@staticmethod
	def _format_basisset(nuclear_charges, basisset, superimposed=False):
		res = ''
		for atomid, nuclear_charge in enumerate(nuclear_charges):
			if superimposed:
				elements = set([max(1, int(_(nuclear_charge))) for _ in (np.round, lambda _: np.round(_ + 1), lambda _: np.round(_ - 1))])
			else:
				elements = set([max(1, int(_(nuclear_charge))) for _ in (np.round,)])
			output = bse.get_basis(basisset, elements=list(elements), fmt='gaussian94')

			res += '%d 0\n' % (atomid + 1)
			skipnext = False
			for line in output.split('\n'):
				if line.startswith('!'):
					skipnext = False
					continue
				if len(line.strip()) == 0 or line.strip() == '****':
					skipnext = True
					continue
				if skipnext:
					skipnext = False
					continue
				res += line + '\n'
			res += '****\n'

		return res.strip()

	@staticmethod
	def _format_nuclear(nuclear_charges):
		return '\n'.join(['%d Nuc %f' % (_[0] + 1, _[1]) for _ in enumerate(nuclear_charges)])

	@staticmethod
	def density_on_grid(inputfile, grid):
		orbkit.options.quiet = True
		orbkit.grid.x = grid[:, 0]*1.88973
		orbkit.grid.y = grid[:, 1]*1.88973
		orbkit.grid.z = grid[:, 2]*1.88973
		orbkit.grid.is_initialized = True

		try:
			qc = orbkit.read.main_read(inputfile, itype='gaussian.fchk')
			rho = orbkit.core.rho_compute(qc, numproc=1)
		except:
			log.log('Unable to read fchk file with orbkit.', level='error', filename=inputfile)
			return grid[:, 0] * 0
		return rho

	def get_input(self, coordinates, nuclear_numbers, nuclear_charges, grid, iscomparison=False):
		basedir = os.path.dirname(os.path.abspath(__file__))
		with open('%s/templates/gaussian.txt' % basedir) as fh:
			template = j.Template(fh.read())

		env_coord = GaussianCalculator._format_coordinates(nuclear_numbers, coordinates)
		env_basis = GaussianCalculator._format_basisset(nuclear_charges, self._basisset, self._superimpose)
		env_nuc = GaussianCalculator._format_nuclear(nuclear_charges)
		env_molcharge = int(np.sum(nuclear_charges) - np.sum(nuclear_numbers))
		return template.render(coordinates=env_coord, method=self._methods[self._method], basisset=env_basis, nuclearcharges=env_nuc, moleculecharge=env_molcharge)

	@classmethod
	def get_runfile(self, coordinates, nuclear_numbers, nuclear_charges, grid):
		basedir = os.path.dirname(os.path.abspath(__file__))
		with open('%s/templates/gaussian-run.sh' % basedir) as fh:
			template = j.Template(fh.read())
		return template.render()

	def get_density_on_grid(self, folder, gridpoints):
		return GaussianCalculator.density_on_grid(folder + '/run.fchk', gridpoints)

	@staticmethod
	def get_total_energy(folder):
		""" Returns the total energy in Hartree."""
		logfile = '%s/run.log' % folder
		try:
			data = cclib.io.ccread(logfile)
		except:
			apdft.log.log('Unable to read energy from log file.', filename=logfile, level='error')
			return 0
		energy = None
		energy = data.scfenergies
		try:
			energy = data.ccenergies
		except AttributeError:
			pass
		return energy / 27.21138602   

	@staticmethod
	def get_electronic_dipole(folder, gridcoords, gridweights):
		""" Returns the electronic dipole moment."""
		#data = cclib.io.ccread('%s/run.log' % folder)
		#return data.moments[1]

		rho = GaussianCalculator.density_on_grid('%s/run.fchk' % folder, gridcoords)
		return -np.sum(gridcoords.T * rho * gridweights, axis=1)

