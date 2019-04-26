#!/usr/bin/env python
import argparse
import sys
import mqm

parser = argparse.ArgumentParser(description='QM calculations on multiple systems.')
parser.add_argument('geometry', help='An XYZ file with the input molecule.', type=str)
parser.add_argument('method', help='A QM method.', choices=mqm.get_methods())
parser.add_argument('basisset', help='A basis set. All of Basis Set Exchange supported.')
parser.add_argument('--remote-host', help='A SSH host to run the calculations in the format username:password@host+port:path/to/dir')
parser.add_argument('--remote-preload', help='A command to run on the remote host to make QM codes available.')
parser.add_argument('--parallel', type=int, help='Number of parallel executions allowed. If 0, uses all available CPU.')
parser.add_argument('--do-explicit-reference', action='store_true', help='Whether to do a reference calculation for every target.')
parser.add_argument('--max-charge', type=int, help='The maximal formal molecular charge for targets.')
parser.add_argument('--max-deltaz', type=int, help='The maximal difference in the Z vector for targets.')

if __name__ == '__main__':
	args = parser.parse_args()

	calculator = mqm.Calculator.GaussianCalculator(args.method, args.basisset)
	nuclear_numbers, coordinates = mqm.read_xyz(args.geometry)

	derivatives = mqm.Derivatives.DerivativeFolders(2, nuclear_numbers, coordinates, args.max_charge, args.max_deltaz)
	derivatives.assign_calculator(calculator)
	derivatives.prepare(args.do_explicit_reference)
	success = derivatives.run(args.parallel, args.remote_host, args.remote_preload)
	if not success:
		mqm.log.log('Incomplete calculations. Aborting', level='critical')
		sys.exit(1)
	derivatives.analyse(args.do_explicit_reference)
