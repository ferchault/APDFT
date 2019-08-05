#!/usr/bin/env python
import argparse
import enum

import apdft
import apdft.settings as aconf
import apdft.calculator as acalc
import apdft.physics as ap


def mode_energies(conf, modeshort=None):
    # select QM code
    calculator_options = conf.apdft_method, conf.apdft_basisset, conf.debug_superimpose
    calculator = conf.energy_code.get_calculator_class()(*calculator_options)

    # parse input
    try:
        nuclear_numbers, coordinates = apdft.read_xyz(conf.energy_geometry)
    except FileNotFoundError:
        apdft.log.log(
            'Unable to open input file "%s".' % conf.energy_geometry, level="error"
        )
        return

    # call APDFT library
    derivatives = ap.APDFT(
        2,
        nuclear_numbers,
        coordinates,
        '.',
        calculator,
        conf.apdft_maxcharge,
        conf.apdft_maxdz,
        conf.apdft_includeonly,
    )

    cost, coverage = derivatives.estimate_cost_and_coverage()
    if conf.debug_validation:
        cost += coverage
    apdft.log.log(
        "Cost estimated.",
        number_calculations=cost,
        number_predictions=coverage,
        level="RESULT",
    )
    if not conf.energy_dryrun:
        derivatives.prepare(conf.debug_validation)
        derivatives.analyse(conf.debug_validation)


def build_main_commandline(set_defaults=True):
    """ Builds an argparse object of the user-facing command line interface."""

    c = apdft.settings.Configuration()

    parser = argparse.ArgumentParser(
        description="QM calculations on multiple systems at once.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # mode selection
    modes = ["energies"]
    parser.add_argument(
        "mode",
        choices=modes,
        nargs=1,
        help="Mode to use. Supported: %s" % ", ".join(modes),
        metavar="mode",
    )

    # allow for shortcut where a mode gets one single argument of any kind
    parser.add_argument("modeshort", type=str, nargs="?", help=argparse.SUPPRESS)

    # options
    for category in sorted(c.list_sections()):
        group = parser.add_argument_group(category)
        for option_name in c.list_options(category):
            option = c[option_name]
            choices = None
            try:
                if issubclass(option.get_validator(), enum.Enum):
                    choices = list(option.get_validator())
            except TypeError:
                pass
            default = None
            if set_defaults:
                default = option.get_value()
            group.add_argument(
                "--%s" % option.get_attribute_name(),
                type=option.get_validator(),
                help=option.get_description(),
                choices=choices,
                default=default,
                metavar="",
            )

    return parser


def parse_into(parser, configuration=None, cliargs=None):
    """ Updates the configuration with the values specified on the command line.
    
    Args:
        parser:         An argparse parser instance.
        configuration:  A :class:`apdft.settings.Configuration` instance. If `None`, a new instance will be returned.
        args:           List of split arguments from the command line.
    Returns:
        Mode of operation, single optional argument, updated configuration."""

    if configuration is None:
        configuration = apdft.settings.Configuration()

    # help specified?
    args = parser.parse_args(cliargs)
    nodefaultparser = build_main_commandline(set_defaults=False)
    args = nodefaultparser.parse_args(cliargs)
    valid_options = configuration.list_options()
    mode = None
    modeshort = None  # single argument for a mode for simplicity
    for k, v in vars(args).items():
        if k in valid_options:
            if v is not None:
                configuration[k].set_value(v)
        else:
            if k == "mode":
                mode = v[0]
            elif k == "modeshort":
                modeshort = v
            else:
                raise ValueError("Unknown argument found.")

    if mode == "energies":
        if modeshort is not None:
            configuration.energy_geometry = modeshort
    return mode, modeshort, configuration
