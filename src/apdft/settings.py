#!/usr/bin/env python
""" Manages settings and config file parsing."""
import enum
import configparser


class CodeEnum(enum.Enum):
    MRCC = "MRCC"
    G09 = "G09"
    PYSCF = "PYSCF"

    def __str__(self):
        return self.value


def intrange(val):
    if val is None:
        return val
    return [int(_) for _ in val.split(",")]


def boolean(val):
    if val == "True":
        return True
    if val == "False":
        return False
    return bool(val)


class Option:
    """ Represents a single configuration option. """

    def __init__(self, category, name, validator, default, description):
        self._category = category
        self._name = name
        self._description = description
        self._validator = validator
        self._value = self._validator(default)

    def get_attribute_name(self):
        return "%s_%s" % (self._category, self._name)

    def get_value(self):
        return self._value

    def get_validator(self):
        return self._validator

    def get_description(self):
        return self._description

    def set_value(self, value):
        self._value = self._validator(value)


class Configuration:
    """ A complete set of configuration values. Merges settings and default values. 
    
    Settings are referred to as category.variablename. """

    def __init__(self):
        options = [
            # Section apdft: relevant for all invocations
            Option(
                "apdft",
                "maxdz",
                int,
                3,
                "Restricts target molecules to have at most this change in nuclear charge per atom",
            ),
            Option(
                "apdft",
                "maxcharge",
                int,
                0,
                "Restricts target molecules to have at most this total molecular charge",
            ),
            Option("apdft", "basisset", str, "def2-TZVP", "The basis set to be used"),
            Option("apdft", "method", str, "CCSD", "Method to be used"),
            Option(
                "apdft",
                "includeonly",
                intrange,
                None,
                "Include only these atom indices, e.g. 0,1,5,7",
            ),
            Option(
                "debug",
                "validation",
                boolean,
                False,
                "Whether to perform validation calculations for all target molecules",
            ),
            Option(
                "debug",
                "superimpose",
                boolean,
                False,
                "Whether to superimpose atomic basis set functions from neighboring elements for fractional nuclear charges",
            ),
            Option("energy", "code", CodeEnum, "MRCC", "QM code to be used"),
            Option(
                "energy",
                "dryrun",
                boolean,
                False,
                "Whether to just estimate the number of targets",
            ),
            Option(
                "energy",
                "geometry",
                str,
                "inp.xyz",
                "XYZ file of the reference molecule",
            ),
        ]
        self.__dict__["_options"] = {}
        for option in options:
            self.__dict__["_options"][option.get_attribute_name()] = option

    def __getattr__(self, attribute):
        """ Read access to configuration options."""
        return self.__dict__["_options"][attribute].get_value()

    def __setattr__(self, attribute, value):
        """ Write access to configuration options."""
        self.__dict__["_options"][attribute].set_value(value)

    def __getitem__(self, attribute):
        return self.__dict__["_options"][attribute]

    def list_options(self, section=None):
        """ Gives all configurable options for a given section."""
        options = [_ for _ in self.__dict__["_options"].keys()]
        if section is not None:
            options = [_ for _ in options if _.startswith("%s_" % section)]
        return options

    def list_sections(self):
        """ Returns a list of all sections."""
        return list(set([_.split("_")[0] for _ in self.__dict__["_options"].keys()]))

    def from_file(self):
        config = configparser.ConfigParser()
        config.read("apdft.conf")

        for section in config.sections():
            for option in config[section]:
                val = config[section][option]
                if val == "None":
                    val = None
                self[option].set_value(val)

    def to_file(self):
        config = configparser.ConfigParser()
        for section in sorted(self.list_sections()):
            vals = dict()
            for option in self.list_options(section):
                try:
                    vals[option] = self[option].get_value().name
                except AttributeError:
                    vals[option] = str(self[option].get_value())
            config[section] = vals
        with open("apdft.conf", "w") as configfile:
            config.write(configfile)
