#!/usr/bin/env python
import pytest
import numpy as np
import os
import basis_set_exchange as bse
from apdft.calculator import MockCalculator
from apdft.calculator.gaussian import GaussianCalculator
from apdft.calculator.pyscf import PyscfCalculator
from apdft.calculator.mrcc import MrccCalculator
import getpass

BASEPATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

def test_local_execution():
	method = 'CCSD'
	basisset = 'STO-3G'
	c2 = GaussianCalculator(method, basisset)
	coordinates = np.array([[0., 0., 0.], [0., 0., 1.]])
	nuclear_numbers = np.array([1, 1])
	nuclear_charges = np.array([0.95, 1.05])
	grid = None
	c2.get_input(coordinates, nuclear_numbers, nuclear_charges, grid)

def test_gaussian_input():
	method = 'CCSD'
	basisset = 'STO-3G'
	c = GaussianCalculator(method, basisset)
	coordinates = np.array([[0., 0., 0.], [0., 0., 1.]])
	nuclear_numbers = np.array([1, 1])
	nuclear_charges = np.array([0.95, 1.05])
	grid = None
	inputfile = c.get_input(coordinates, nuclear_numbers, nuclear_charges, grid)
	expected = '''%Chk=run.chk
#CCSD(Full,MaxCyc=100) Gen scf=tight Massage integral=NoXCTest Density=Current Prop=EFG

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

def test_mrcc_input():
	method = 'CCSD'
	basisset = 'STO-3G'
	c = MrccCalculator(method, basisset)
	coordinates = np.array([[0., 0., 0.], [0., 0., 1.]])
	nuclear_numbers = np.array([1, 1])
	nuclear_charges = np.array([0.95, 1.05])
	grid = None
	inputfile = c.get_input(coordinates, nuclear_numbers, nuclear_charges, grid)
	expected = '''calc=ccsd
mem=5GB
core=frozen
ccprog=mrcc
basis=STO-3G
grdens=on
rgrid=Log3
grtol=10
agrid=LD0770
unit=angs
geom=xyz
2

1 0.000000 0.000000 0.000000
1 0.000000 0.000000 1.000000
qmmm=Amber
pointcharges
2
0.000000 0.000000 0.000000 -0.050000
0.000000 0.000000 1.000000 0.050000'''
	assert expected == inputfile

def test_gaussian_epn():
	epn = GaussianCalculator.get_epn('%s/data/apdft-run/order-1/site-0-up/' % BASEPATH, np.array([[0., 0.,0.], [0.,0.,1.0]]), [0,1], [7.05, 7])
	assert (abs(epn[0] - 21.65248147) < 1e-5)
	assert (abs(epn[1] - 21.60575834) < 1e-5)

def test_mrcc_dft_energy():
	energy = MrccCalculator.get_total_energy('%s/data/mrcc-dft-energy/' % BASEPATH)
	assert (abs(energy - -925.1422270624525481) < 1e-7)

def test_pyscf_read_energy():
	energy = PyscfCalculator.get_total_energy('%s/data/pyscf-hf-run/' % BASEPATH)
	assert (abs(energy - -109.75023802562146) < 1e-7)

def test_pyscf_read_dipole():
	dipole = PyscfCalculator.get_electronic_dipole('%s/data/pyscf-hf-run/' % BASEPATH)
	assert (abs(dipole[0] - 0) < 1e-7)
	assert (abs(dipole[1] - 0) < 1e-7)
	assert (abs(dipole[2] - -0.053729479129616675) < 1e-7)

def test_pyscf_basiset():
	res = PyscfCalculator._format_basis([7], "6-31G")
	assert (res == str({7: bse.get_basis("6-31G", "N", fmt="nwchem")}))
	res = PyscfCalculator._format_basis([1], "6-31G")
	assert (res == str({1: bse.get_basis("6-31G", "H", fmt="nwchem")}))
	res = PyscfCalculator._format_basis([1], "STO-3G")
	assert (res == str({1: bse.get_basis("STO-3G", "H", fmt="nwchem")}))

def test_pyscf_input():
	method = 'CCSD'
	basisset = 'STO-3G'
	c = PyscfCalculator(method, basisset)
	coordinates = np.array([[0., 0., 0.], [0., 0., 1.]])
	nuclear_numbers = np.array([1, 1])
	nuclear_charges = np.array([0.95, 1.05])
	grid = None
	c.get_input(coordinates, nuclear_numbers, nuclear_charges, grid)

def test_pyscf_nan():
	lines = ['TOTAL_ENERGY 42']
	assert (PyscfCalculator._read_value(None, 'TOTAL_ENERGY', False, lines) == np.array(42.))

	lines = ['TOTAL_ENERGY nan']
	with pytest.raises(ValueError):
		PyscfCalculator._read_value(None, 'TOTAL_ENERGY', False, lines)

def test_pyscf_array_read():
	lines = ['ELECTRONIC_EPN 0 10', 'ELECTRONIC_EPN 1 11']
	assert (np.allclose(PyscfCalculator._read_value(None, 'ELECTRONIC_EPN', True, lines)[:, 1], np.array((10, 11))))
