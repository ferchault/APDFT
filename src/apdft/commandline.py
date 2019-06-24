#!/usr/bin/env python
import argparse
import enum

import apdft
import apdft.settings as aconf
import apdft.Calculator as acalc

def mode_energies(conf):
    if conf.energy_code == aconf.CodeEnum.MRCC:
        calculator = acalc.MrccCalculator(conf.apdft_method, conf.apdft_basisset, conf.debug_superimpose)
    else:
        calculator = acalc.GaussianCalculator(conf.apdft_method, conf.apdft_basisset, conf.debug_superimpose)

    nuclear_numbers, coordinates = apdft.read_xyz(conf.energy_geometry)

    derivatives = apdft.Derivatives.DerivativeFolders(2, nuclear_numbers, coordinates, conf.apdft_maxcharge, conf.apdft_maxdeltaz, conf.apdft_includeonly)
    if conf.energy_dryrun:
        cost, coverage = derivatives.estimate_cost_and_coverage()
        if conf.debug_validation:
            cost += coverage
        apdft.log.log('Cost estimated.', number_calculations=cost, number_prediction=coverage, level='RESULT')
    else:
        derivatives.assign_calculator(calculator, args.projectname)
        derivatives.prepare(conf.debug_validation)
        success = derivatives.run(args.parallel, args.remote_host, args.remote_preload)
        if not success:
            apdft.log.log('Incomplete calculations. Aborting', level='critical')
            sys.exit(1)
        derivatives.analyse(conf.debug_validation)

def build_main_commandline():
    """ Builds an argparse object of the user-facing command line interface."""

    c = apdft.settings.Configuration()

    parser = argparse.ArgumentParser(description='QM calculations on multiple systems at once.')

    # options
    for category in sorted(c.list_sections()):
        for option_name in c.list_options(category):
            option = c[option_name]
            choices = None
            try:
                if issubclass(option.get_validator(), enum.Enum):
                    choices = [_.name for _ in option.get_validator()]
            except TypeError:
                pass                
            parser.add_argument('--%s' % option.get_attribute_name(), type=option.get_validator(), help=option.get_description(), choices=choices)
    
    # modes
    subparsers = parser.add_subparsers(dest='mode')
    energies = subparsers.add_parser('energies')

    return parser

def parse_into(parser, configuration=None):
    """ Updates the configuration with the values specified on the command line.
    
    Args:
        parser:         An argparse parser instance.
        configuration:  A :class:`apdft.settings.Configuration` instance. If `None`, a new instance will be returned.
    Returns:
        Mode of operation, updated configuration."""
    
    if configuration is None:
        configuration = apdft.settings.Configuration()
    
    args = parser.parse_args()
    valid_options = configuration.list_options()
    mode = None
    for k, v in vars(args).items():
        if k in valid_options:
            if v is not None:
                configuration[k].set_value(v)
        else:
            mode = v

    return mode, configuration