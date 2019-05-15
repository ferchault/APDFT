#!/usr/bin/env python
import pytest
import numpy as np

import apdft.math as math

def test_partition_single():
	res = math.IntegerPartitions.partition(2, 1)
	assert res == [[2,]]

def test_partition_two():
	res = math.IntegerPartitions.partition(2, 2)
	assert len(res) == 3
	assert [2, 0] in res
	assert [1, 1] in res
	assert [0, 2] in res

def test_partition_around():
	def compare(total, maxelements, around, maxdz):
		res = math.IntegerPartitions.partition(total, maxelements)
		expected = [_ for _ in res if np.sum(np.abs(np.array(_) - around)) <= maxdz]

		actual = math.IntegerPartitions.partition(total, maxelements, around, maxdz)
		assert len(expected) == len(actual)
		for element in expected:
			assert element in actual

	compare(5, 2, np.array([2, 2]), 1)
	compare(5, 2, np.array([12, 12]), 0)
	compare(5, 2, np.array([2, 2]), 0)
	compare(14, 2, np.array([7, 7]), 2)
	compare(14, 2, np.array([7, 7]), 3)
	compare(5, 2, np.array([2, 2]), 10)
