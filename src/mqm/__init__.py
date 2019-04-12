#!/usr/bin/env python
from . import Calculator

def get_methods():
	calculators = [getattr(Calculator, _) for _ in dir(Calculator) if 'Calculator' in _ and _ != 'Calculator']
	methods = []
	for calculator in calculators:
		try:
			c = calculator().get_methods()
		except:
			continue
		methods += c
	return methods
