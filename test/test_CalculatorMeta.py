#!/usr/bin/env python
import pytest
import numpy as np
import mqm.Calculator as mqmc

def test_horton_has_methods():
	assert 'HF' in mqmc.HortonCalculator._methods.keys()

def test_horton():
	return
	c = mqmc.HortonCalculator()
	coordinates = np.array([[0., 0., 0.], [0., 0., 1.]])
	nuclear_numbers = np.array([1, 1])
	nuclear_charges = np.array([1., 1.])
	grid = None
	method = 'HF'
	basisset = 'STO-3G'
	c.evaluate(coordinates, nuclear_numbers, nuclear_charges, grid, method, basisset)

def test_gaussian_input():
	c = mqmc.GaussianCalculator()
	coordinates = np.array([[0., 0., 0.], [0., 0., 1.]])
	nuclear_numbers = np.array([1, 1])
	nuclear_charges = np.array([0.95, 1.05])
	grid = None
	method = 'CCSD'
	basisset = 'STO-3G'
	inputfile = c.get_input(coordinates, nuclear_numbers, nuclear_charges, grid, method, basisset)
	expected = '''%Chk=run.chk
#CCSD(Full,MaxCyc=100) Gen scf=tight Massage guess=indo integral=NoXCTest Pop=None Density=Current

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
