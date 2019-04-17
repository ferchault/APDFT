#!/usr/bin/env python
import os
import sys
import glob

import numpy as np
import jinja2 as j
import basis_set_exchange as bse
import cclib
import subprocess

# load local orbkit
basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append('%s/../../dep/orbkit/orbkit' % basedir)
import orbkit

class Calculator(object):
	""" A concurrency-safe blocking interface for an external QM code."""
	def get_methods(self):
		return list(self._methods.keys())

	def get_density_on_grid(self, folder, gridpoints):
		raise NotImplementedError()

	@staticmethod
	def _parse_ssh_constr(constr):
		""" Parses a connection string.

		Format: username:password@host+port:path/to/dir"""
		userpass, rest = constr.split('@')
		username, password = userpass.split(':')
		rest, path = rest.split(':')
		host, port = rest.split('+')
		return username, password, host, port, path

	def execute(self, folder, remote_constr=None):
		""" Run a calculation with the input file in folder."""

		if remote_constr == None:
			subprocess.run('%s/run.sh' % folder)
		else:
			import paramiko
			with paramiko.SSHClient() as s:
				# connect
				s.load_system_host_keys()
				username, password, host, port, path = Calculator._parse_ssh_constr(remote_constr)
				s.connect(host, port, username, password)
				s.exec_command('cd %s' % path)

				# copy files
				sftp = s.open_sftp()
				sftp.chdir(path)
				for fn in glob.glob('%s/*' % folder):
					sftp.put(fn, os.path.basename(fn))

				# run
				s.exec_command('./run.sh')

				# copy back
				for fn in sftp.listdir():
					sftp.get(fn, '%s/%s' % (folder, fn))


class MockCalculator(Calculator):
	_methods = {}
	@classmethod
	def get_runfile(self, coordinates, nuclear_numbers, nuclear_charges, grid, method, basisset):
		basedir = os.path.dirname(os.path.abspath(__file__))
		with open('%s/templates/mock-run.sh' % basedir) as fh:
			template = j.Template(fh.read())
		return template.render()


class GaussianCalculator(Calculator):
	_methods = {
		'CCSD': 'CCSD(Full,MaxCyc=100)',
		'PBE0': 'PBE1PBE',
		'PBE': 'PBEPBE',
	}

	@staticmethod
	def _format_coordinates(nuclear_numbers, coordinates):
		ret = ''
		for Z, coords in zip(nuclear_numbers, coordinates):
			ret += '%d %f %f %f\n' % (Z, coords[0], coords[1], coords[2])
		return ret[:-1]

	@staticmethod
	def _format_basisset(nuclear_charges, basisset):
		res = ''
		for atomid, nuclear_charge in enumerate(nuclear_charges):
			elements = set([int(_(nuclear_charge)) for _ in (np.ceil, np.floor)])
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

		qc = orbkit.read.main_read(inputfile, itype='gaussian.fchk')
		rho = orbkit.core.rho_compute(qc, numproc=1)
		return rho

	def get_input(self, coordinates, nuclear_numbers, nuclear_charges, grid, method, basisset):
		basedir = os.path.dirname(os.path.abspath(__file__))
		with open('%s/templates/gaussian.txt' % basedir) as fh:
			template = j.Template(fh.read())

		env_coord = GaussianCalculator._format_coordinates(nuclear_numbers, coordinates)
		env_basis = GaussianCalculator._format_basisset(nuclear_numbers, basisset)
		env_nuc = GaussianCalculator._format_nuclear(nuclear_charges)
		return template.render(coordinates=env_coord, method=self._methods[method], basisset=env_basis, nuclearcharges=env_nuc)

	@classmethod
	def get_runfile(self, coordinates, nuclear_numbers, nuclear_charges, grid, method, basisset):
		basedir = os.path.dirname(os.path.abspath(__file__))
		with open('%s/templates/gaussian-run.sh' % basedir) as fh:
			template = j.Template(fh.read())
		return template.render()

	def get_density_on_grid(self, folder, gridpoints):
		return GaussianCalculator.density_on_grid(folder + '/run.fchk', gridpoints)

	def get_total_energy(self, folder):
		data = cclib.io.ccread('%s/run.log' % folder)
		energy = None
		energy = data.scfenergies
		try:
			energy = data.ccenergies
		except AttributeError:
			pass
		return energy / 27.21138602


class HortonCalculator(Calculator):
	_methods = {
		'HF': 'tbd',
		'LDA': 'tbd',
		'PBE': 'tbd',
		'PBE0': 'tbd',
		}

	def __init__(self):
		pass
