#!/usr/bin/env python
""" Manages settings and config file parsing."""

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
    
    def set_value(self, value):
        self._value = self._validator(value)

class Configuration():
    """ A complete set of configuration values. Merges settings and default values. 
    
    Settings are referred to as category.variablename. """
    def __init__(self):
        options = [
            Option('apdft', 'maxdz', int, 3, 'Restricts target molecules to have at most this change in nuclear charge per atom'),
            Option('apdft', 'basisset', str, 'def2-TZVP', 'Sets the basis set to be used.'),
        ]
        self.__dict__['_options'] = {}
        for option in options:
            self._options[option.get_attribute_name()] = option
    
    def __getattr__(self, attribute):
        return self._options[attribute].get_value()

    def __setattr__(self, attribute, value):
        self.__dict__['_options'][attribute].set_value(value)
        
    
