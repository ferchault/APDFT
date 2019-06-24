#!/usr/bin/env python
import argparse
import sys
import apdft
import apdft.commandline as acmd
import apdft.settings as aconf

if __name__ == '__main__':
	parser = acmd.build_main_commandline()
	mode, mode_args, conf = acmd.parse_into(parser)

	if conf.energy_code == aconf.CodeEnum.MRCC:
		calculator = apdft.Calculator.MrccCalculator(args.method, args.basisset, args.superimpose)
	else:
		calculator = apdft.Calculator.GaussianCalculator(args.method, args.basisset, args.superimpose)

	nuclear_numbers, coordinates = apdft.read_xyz(args.geometry)

	# included atoms
	if args.include_atoms is not None:
		args.include_atoms = [int(_) for _ in args.include_atoms.split(',')]

	derivatives = apdft.Derivatives.DerivativeFolders(2, nuclear_numbers, coordinates, args.max_charge, args.max_deltaz, args.include_atoms)
	if args.dry_run:
		cost, coverage = derivatives.estimate_cost_and_coverage()
		if args.do_explicit_reference:
			cost += coverage
		apdft.log.log('Cost estimated.', number_calculations=cost, number_prediction=coverage, level='RESULT')
	else:
		derivatives.assign_calculator(calculator, args.projectname)
		derivatives.prepare(args.do_explicit_reference)
		success = derivatives.run(args.parallel, args.remote_host, args.remote_preload)
		if not success:
			apdft.log.log('Incomplete calculations. Aborting', level='critical')
			sys.exit(1)
		derivatives.analyse(args.do_explicit_reference)
