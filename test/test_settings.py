#!/usr/bin/env python
import pytest

import apdft.settings as s
import apdft.commandline as acmd

def test_basis_set_default():
    conf = s.Configuration()
    assert conf.apdft_basisset == 'def2-TZVP'

def test_typechecking():
    conf = s.Configuration()
    with pytest.raises(ValueError):
        conf.apdft_maxdz = 'fail'

def test_conversion():
    conf = s.Configuration()
    conf.apdft_maxdz = '42'
    assert conf.apdft_maxdz == 42

def test_enum():
    conf = s.Configuration()
    conf.energy_code = 'MRCC'
    assert conf.energy_code == s.CodeEnum.MRCC
    with pytest.raises(ValueError):
        conf.energy_code = 'invalid'

def test_sections():
    conf = s.Configuration()
    assert 'apdft' in conf.list_sections()
    assert 'energy' in conf.list_sections()

def test_read_write():
    conf = s.Configuration()
    conf.apdft_maxdz = '42'
    conf.to_file()
    
    conf2 = s.Configuration()
    conf2.from_file()
    assert conf.apdft_maxdz == 42

def test_parse_into_same_parameter():
    conf = s.Configuration()
    conf.apdft_maxdz = '42'
    parser = acmd.build_main_commandline()
    args = ['energies', '--apdft_maxdz', '43']
    acmd.parse_into(parser, configuration=conf, cliargs=args)
    assert conf.apdft_maxdz == 43

def test_parse_into_other_parameter():
    conf = s.Configuration()
    conf.apdft_maxdz = '42'
    parser = acmd.build_main_commandline()
    args = ['energies', '--apdft_maxcharge', '2']
    acmd.parse_into(parser, configuration=conf, cliargs=args)
    assert conf.apdft_maxdz == 42

def test_parse_into_implicit_parameter():
    conf = s.Configuration()
    conf.apdft_maxdz = '42'
    assert conf.energy_geometry == 'inp.xyz'
    parser = acmd.build_main_commandline()
    args = ['energies', 'co2.xyz']
    acmd.parse_into(parser, configuration=conf, cliargs=args)
    assert conf.apdft_maxdz == 42
    assert conf.energy_geometry == 'co2.xyz'