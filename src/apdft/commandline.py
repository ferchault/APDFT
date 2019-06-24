#!/usr/bin/env python
import argparse
import enum

import apdft.settings

def build_main_commandline():
    """ Builds an argparse object of the user-facing command line interface."""

    c = apdft.settings.Configuration()

    parser = argparse.ArgumentParser(description='QM calculations on multiple systems at once.')

    # options
    for category in sorted(c.list_sections()):
        for option_name in c.list_options(category):
            option = c[option_name]
            if issubclass(option.get_validator(), enum.Enum):
                choices = [_.name for _ in option.get_validator()]
            else:
                choices = None
            parser.add_argument('--%s' % option.get_attribute_name(), type=option.get_validator(), help=option.get_description(), choices=choices)
    
    # modes
    subparsers = parser.add_subparsers(dest='mode')
    energies = subparsers.add_parser('energies')
    energies.add_argument('geometry', help='Input file', nargs='?')

    return parser

def parse_into(parser, configuration=None):
    """ Updates the configuration with the values specified on the command line.
    
    Args:
        parser:         An argparse parser instance.
        configuration:  A :class:`apdft.settings.Configuration` instance. If `None`, a new instance will be returned.
    Returns:
        Mode of operation, arguments to operation, updated configuration."""
    
    if configuration is None:
        configuration = apdft.settings.Configuration()
    
    args = parser.parse_args()
    valid_options = configuration.list_options()
    mode = None
    mode_args = {}
    for k, v in vars(args).items():
        if k in valid_options:
            if v is not None:
                configuration[k].set_value(v)
        else:
            if k != 'mode':
                mode_args[k] = v
            else:
                mode = v

    return mode, mode_args, configuration