#!/usr/bin/env python
""" Manages settings and config file parsing."""
import enum

class CodeEnum(enum.Enum):
    MRCC = 'MRCC'
    G09 = 'G09'

class Option():
    """ Represents a single configuration option. """
    def __init__(self, category, name, validator, default, description):
        self._category = category
        self._name = name
        self._description = description
        self._validator = validator
        self._value = default
    
    def get_attribute_name(self):
        return '%s_%s' % (self._category, self._name)
    
    def get_value(self):
        return self._value
    
    def get_validator(self):
        return self._validator

    def get_description(self):
        return self._description
    
    def set_value(self, value):
        self._value = self._validator(value)

class Configuration():
    """ A complete set of configuration values. Merges settings and default values. 
    
    Settings are referred to as category.variablename. """
    def __init__(self):
        options = [
            # Section apdft: relevant for all invocations
            Option('apdft', 'maxdz', int, 3, 'Restricts target molecules to have at most this change in nuclear charge per atom'),
            Option('apdft', 'basisset', str, 'def2-TZVP', 'The basis set to be used.'),
            Option('apdft', 'method', str, 'CCSD', 'Method to be used.'),
            Option('apdft', 'validation', bool, False, 'Whether to perform validation calculations for all target molecules.'),
            Option('energy', 'code', CodeEnum, CodeEnum.MRCC, 'QM code to be used.'),
        ]
        self.__dict__['_options'] = {}
        for option in options:
            self.__dict__['_options'][option.get_attribute_name()] = option
    
    def __getattr__(self, attribute):
        """ Read access to configuration options."""
        return self.__dict__['_options'][attribute].get_value()

    def __setattr__(self, attribute, value):
        """ Write access to configuration options."""
        self.__dict__['_options'][attribute].set_value(value)
    
    def __getitem__(self, attribute):
        return self.__dict__['_options'][attribute]
    
    def list_options(self, section):
        """ Gives all configurable options for a given section."""
        return [_ for _ in self.__dict__['_options'].keys() if _.startswith('%s_' % section)]
    
    def list_sections(self):
        """ Returns a list of all sections."""
        return list(set([_.split('_')[0] for _ in self.__dict__['_options'].keys()]))
        