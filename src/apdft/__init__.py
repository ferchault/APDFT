#!/usr/bin/env python
import numpy as np
import sys
from basis_set_exchange import lut
from . import Calculator
from . import Derivatives
from . import commandline
from . import settings

# setup output
def _setup_logging():
	import structlog
	my_styles = structlog.dev.ConsoleRenderer.get_default_level_styles()
	my_styles["RESULT"] = my_styles["info"]

	if sys.stdout.isatty():
		renderer = structlog.dev.ConsoleRenderer(level_styles=my_styles)
	else:
		renderer = structlog.processors.JSONRenderer()

	structlog.configure(
		processors=[
			structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M"),
			structlog.processors.StackInfoRenderer(),
			structlog.processors.format_exc_info,
			renderer
		],
		context_class=dict,
		cache_logger_on_first_use=True,
	)
	return structlog.get_logger()
try:
	log = _setup_logging()
except TypeError:
	# mocking environment
	log = None

def get_methods():
	calculators = [getattr(Calculator, _) for _ in dir(Calculator) if 'Calculator' in _ and _ != 'Calculator']
	methods = []
	for calculator in calculators:
		c = calculator._methods.keys()
		methods += c
	return sorted(set(methods))

def get_element_number(element):
	return lut.element_Z_from_sym(element)

def read_xyz(fn):
	with open(fn) as fh:
		lines = fh.readlines()
	numatoms = int(lines[0].strip())
	lines = lines[2:2 + numatoms]
	nuclear_numbers = []
	coordinates = []
	for line in lines:
		line = line.strip()
		if len(line) == 0:
			break
		parts = line.split()
		nuclear_numbers.append(get_element_number(parts[0]))
		coordinates.append([float(_) for _ in parts[1:4]])
	return np.array(nuclear_numbers), np.array(coordinates)