#!/usr/bin/env python
import pytest
import numpy as np

import mqm.math as math

def test_partition_single():
	res = math.IntegerPartitions.partition(2, 1)
	assert res == [[2,]]

def test_partition_two():
	res = math.IntegerPartitions.partition(2, 2)
	assert len(res) == 3
	assert [2, 0] in res
	assert [1, 1] in res
	assert [0, 2] in res
