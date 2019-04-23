#!/usr/bin/env python
import os
import sys
import glob
import random
import string
import warnings

import numpy as np
import jinja2 as j
import basis_set_exchange as bse
import cclib
import subprocess # nosec
import re
import getpass

import mqm

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

		Accepted formats:
		username:password@host+port:path/to/dir
		username@host+port:path/to/dir
		username@host+port:
		username@host:path/to/dir
		username@host
		host
		"""
		regex = r"((?P<username>[^:@]+)(:(?P<password>[^@]+))?@)?(?P<host>[^+:]+)(\+(?P<port>[^:]+))?:?(?P<path>[^:@]*)"
		matches = re.search(regex, constr)
		groups = matches.groupdict()

		if groups['port'] is None:
			groups['port'] = 22

		if groups['username'] is None:
			groups['username'] = getpass.getuser()

		if groups['path'] is '':
			groups['path'] = '.'

		return groups['username'], groups['password'], groups['host'], groups['port'], groups['path']

	@staticmethod
	def _get_tempname():
		return 'mqmc-tmp-' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

	@staticmethod
	def execute(folder, remote_constr=None, remote_preload=None):
		""" Run a calculation with the input file in folder."""

		if remote_constr == None:
			p = subprocess.run('%s/run.sh' % folder, universal_newlines=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)  # nosec
			if p.returncode != 0:
				print ('E + Error running %s/run.sh:')
				for line in p.stdout.split('\n'):
					print ('E | %s' % line)
				print ('E + Run skipped.\n')
		else:
			import paramiko
			with paramiko.SSHClient() as s:
				# connect
				#s.load_system_host_keys()

				s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
				username, password, host, port, path = Calculator._parse_ssh_constr(remote_constr)

				try:
					with warnings.catch_warnings():
						warnings.simplefilter('ignore')
						s.connect(host, port, username, password)
				except paramiko.ssh_exception.NoValidConnectionsError:
					mqm.log.log('Unable to establish SSH connection.', level='error', host=host, port=port, username=username, password=password)
					return
				except:
					mqm.log.log('General SSH error.', level='error', host=host, port=port, username=username, password=password)
					return
				sftp = s.open_sftp()
				sftp.chdir(path)

				# create temporary folder
				tmpname = Calculator._get_tempname()
				sftp.mkdir(tmpname)
				sftp.chdir(tmpname)
				
				# copy files
				for fn in glob.glob('%s/*' % folder):
					sftp.put(fn, os.path.basename(fn))
				sftp.chmod('run.sh', 0o700)

				# run
				_ = s.exec_command('cd %s; cd %s' % (path, tmpname)) # nosec
				stdout = _[0]
				if stdout.channel.recv_exit_status() != 0:
					mqm.log.log('Unable to navigate on remote machine.', level='error', host=host, port=port, username=username, password=password, path="%s/%s" % (path, tmpname))
					return

				if remote_preload == None:
					remote_preload = ''
				else:
					remote_preload = '%s; ' % remote_preload
				_ = s.exec_command('%scd %s; cd %s; ./run.sh' % (remote_preload, path, tmpname)) # nosec
				stdout = _[0]
				status = stdout.channel.recv_exit_status()
				if status != 0:
					msglines = stdout.readlines() + stderr.readlines()
					mqm.log.log('Unable to execute runscript on remote machine.', level='error', host=host, port=port, username=username, password=password, path=folder, remotemsg=msglines)

				# copy back
				for fn in sftp.listdir():
					sftp.get(fn, '%s/%s' % (folder, fn))

				# clear
				s.exec_command('cd %s; rm -rf "%s"' % (path, tmpname)) # nosec


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
		'HF': 'UHF',
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

	def get_input(self, coordinates, nuclear_numbers, nuclear_charges, grid, method, basisset, iscomparison=False):
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

	@staticmethod
	def get_total_energy(folder):
		""" Returns the total energy in Hartree."""
		data = cclib.io.ccread('%s/run.log' % folder)
		energy = None
		energy = data.scfenergies
		try:
			energy = data.ccenergies
		except AttributeError:
			pass
		return energy / 27.21138602

	@staticmethod
	def get_dipole(folder):
		""" Returns the dipole moment in Debye."""
		data = cclib.io.ccread('%s/run.log' % folder)
		return data.moments[1]

