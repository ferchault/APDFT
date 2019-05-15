#!/usr/bin/env python
import pytest
import numpy as np
import os
import tempfile
import apdft.Calculator as apc
import getpass

def test_local_execution():
	method = 'CCSD'
	basisset = 'STO-3G'
	c = apc.MockCalculator(method, basisset)
	c2 = apc.GaussianCalculator(method, basisset)
	coordinates = np.array([[0., 0., 0.], [0., 0., 1.]])
	nuclear_numbers = np.array([1, 1])
	nuclear_charges = np.array([0.95, 1.05])
	grid = None
	inputfile = c2.get_input(coordinates, nuclear_numbers, nuclear_charges, grid)

	with tempfile.TemporaryDirectory() as tmpname:
		os.chdir(tmpname)
		with open('run.inp', 'w') as fh:
			fh.write(inputfile)
		with open('run.sh', 'w') as fh:
			fh.write(c.get_runfile(coordinates, nuclear_numbers, nuclear_charges, grid))
		os.chmod('run.sh', 0o777)

		c.execute('.')

		with open('run.log') as fh:
			assert set(' '.join(fh.readlines()).strip().split()) == set(['run.inp', 'run.sh'])
		os.chdir('..')

def test_ssh_constr():
	result = apc.Calculator._parse_ssh_constr('username:password@host+port:path/to/dir')
	assert result == ('username', 'password', 'host', 'port', 'path/to/dir')
	result = apc.Calculator._parse_ssh_constr('username@host+port:path/to/dir')
	assert result == ('username', None, 'host', 'port', 'path/to/dir')
	result = apc.Calculator._parse_ssh_constr('username@host+port:')
	assert result == ('username', None, 'host', 'port', '.')
	result = apc.Calculator._parse_ssh_constr('username@host:path/to/dir')
	assert result == ('username', None, 'host', 22, 'path/to/dir')
	result = apc.Calculator._parse_ssh_constr('username@host')
	assert result == ('username', None, 'host', 22, '.')
	result = apc.Calculator._parse_ssh_constr('host')
	assert result == (getpass.getuser(), None, 'host', 22, '.')

def test_gaussian_input():
	method = 'CCSD'
	basisset = 'STO-3G'
	c = apc.GaussianCalculator(method, basisset)
	coordinates = np.array([[0., 0., 0.], [0., 0., 1.]])
	nuclear_numbers = np.array([1, 1])
	nuclear_charges = np.array([0.95, 1.05])
	grid = None
	inputfile = c.get_input(coordinates, nuclear_numbers, nuclear_charges, grid)
	expected = '''%Chk=run.chk
#CCSD(Full,MaxCyc=100) Gen scf=tight Massage integral=NoXCTest Pop=Dipole Density=Current NoSymm

run

0 1
1 0.000000 0.000000 0.000000
1 0.000000 0.000000 1.000000

1 0
S   3   1.00
      0.3425250914D+01       0.1543289673D+00
      0.6239137298D+00       0.5353281423D+00
      0.1688554040D+00       0.4446345422D+00
****
2 0
S   3   1.00
      0.3425250914D+01       0.1543289673D+00
      0.6239137298D+00       0.5353281423D+00
      0.1688554040D+00       0.4446345422D+00
****

1 Nuc 0.950000
2 Nuc 1.050000'''
	assert expected == inputfile
