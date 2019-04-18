#!/usr/bin/env python
import numpy as np
from basis_set_exchange import lut
from . import Calculator
from . import Derivatives

def get_methods():
	calculators = [getattr(Calculator, _) for _ in dir(Calculator) if 'Calculator' in _ and _ != 'Calculator']
	methods = []
	for calculator in calculators:
		try:
			c = calculator().get_methods()
		except ImportError:
			continue
		methods += c
	return sorted(set(methods))

def get_element_number(element):
	return lut.element_Z_from_sym(element)

def read_xyz(fn):
	with open(fn) as fh:
		lines = fh.readlines()[2:]
	nuclear_numbers = []
	coordinates = []
	for line in lines:
		parts = line.strip().split()
		nuclear_numbers.append(get_element_number(parts[0]))
		coordinates.append([float(_) for _ in parts[1:4]])
	return np.array(nuclear_numbers), np.array(coordinates)