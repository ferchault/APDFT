#!/usr/bin/env python
import numpy as np

class IntegerPartitions(object):
	@staticmethod
	def partition(total, maxelements):
		""" Builds all integer partitions of *total* split into *maxelements* parts.

		Note that ordering matters, i.e. (2, 1) and (1, 2) are district partitions. Moreover, elements of zero value are allowed. In all cases, the sum of all elements is equal to *total*.
		There is no guarantee as for the ordering of elements.

		Args:
			total:			The sum of all entries. [Integer]
			maxelements:	The number of elements to split into. [Integer]
		Returns:
			A list of all partitions as lists.
		"""
		if maxelements == 1:
			return [[total]]
		res = []
		for x in range(total + 1):
			for p in IntegerPartitions.partition(total - x, maxelements - 1):
				res.append([x] + p)
		return res
