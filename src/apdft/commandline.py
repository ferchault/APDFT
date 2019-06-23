#!/usr/bin/env python
import argparse
import enum

import apdft.settings

def build_main_commandline():
    """ Builds an argparse object of the user-facing command line interface."""

    c = apdft.settings.Configuration()

    parser = argparse.ArgumentParser(description='QM calculations on multiple systems at once.')
    for category in sorted(c.list_sections()):
        for option_name in c.list_options(category):
            option = c[option_name]
            if issubclass(option.get_validator(), enum.Enum):
                choices = [_.name for _ in option.get_validator()]
            else:
                choices = None
            parser.add_argument('--%s' % option.get_attribute_name(), type=option.get_validator(), help=option.get_description(), choices=choices)
    
    return parser