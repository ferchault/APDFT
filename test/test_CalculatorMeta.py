#!/usr/bin/env python
import pytest

import mqm.Calculator as mqmc

def test_force_init():
	with pytest.raises(NotImplementedError):
		mqmc.Calculator()

def test_horton_has_methods():
	assert 'HF' in mqmc.HortonCalculator._methods.keys()
