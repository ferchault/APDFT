#!/usr/bin/env python
import unittest

import mqm.Calculator as mqmc

class TestCalculatorMeta(unittest.TestCase):
	def test_force_init(self):
		with self.assertRaises(NotImplementedError):
			mqmc.Calculator()

	def test_horton_has_methods(self):
		self.assertTrue('HF' in mqmc.HortonCalculator._methods.keys())
