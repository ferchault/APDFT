#!/usr/bin/env python
import pytest

import apdft.settings as s

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