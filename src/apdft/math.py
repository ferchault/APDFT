#!/usr/bin/env python
import numpy as np
import functools


class IntegerPartitions(object):
    @staticmethod
    @functools.lru_cache(maxsize=64)
    def _do_partition(total, maxelements, around=None, maxdz=None):
        """ Builds all integer partitions of *total* split into *maxelements* parts.

		Note that ordering matters, i.e. (2, 1) and (1, 2) are district partitions. Moreover, elements of zero value are allowed. In all cases, the sum of all elements is equal to *total*.
		There is no guarantee as for the ordering of elements.

		If a center *around* is given, then a radius *maxdz* is required.
		Only those partitions are listed where the L1 norm of the distance between partition and *around* is less or equal to *maxdz*.

		Args:
			total:			The sum of all entries. [Integer]
			maxelements:	The number of elements to split into. [Integer]
			around:			Tuple of N entries. Center around which partitions are listed. [Integer]
			maxdz:			Maximum absolute difference in Z space from center *around*. [Integer]
		Returns:
			A list of all partitions as lists.
		"""
        if (around is None) != (maxdz is None):
            raise ValueError("Cannot define center or radius alone.")

        if maxelements == 1:
            if around is not None and maxdz < abs(total - around[-maxelements]):
                return []
            else:
                return [[total]]
        res = []

        # get range to cover
        if around is None:
            first = 0
            last = total
            limit = None
        else:
            first = max(0, around[-maxelements] - maxdz)
            last = min(total, around[-maxelements] + maxdz)
        for x in range(first, last + 1):
            if around is not None:
                limit = maxdz - abs(x - around[-maxelements])
            for p in IntegerPartitions._do_partition(
                total - x, maxelements - 1, around, limit
            ):
                res.append([x] + p)
        return res

    @staticmethod
    def partition(total, maxelements, around=None, maxdz=None):
        """ Builds all integer partitions of *total* split into *maxelements* parts.

		Note that ordering matters, i.e. (2, 1) and (1, 2) are district partitions. Moreover, elements of zero value are allowed. In all cases, the sum of all elements is equal to *total*.
		There is no guarantee as for the ordering of elements.

		If a center *around* is given, then a radius *maxdz* is required.
		Only those partitions are listed where the L1 norm of the distance between partition and *around* is less or equal to *maxdz*.

		Args:
			total:			The sum of all entries. [Integer]
			maxelements:	The number of elements to split into. [Integer]
			around:			Iterable of N entries. Center around which partitions are listed. [Integer]
			maxdz:			Maximum absolute difference in Z space from center *around*. [Integer]
		Returns:
			A list of all partitions as lists.
		"""
        if around is not None:
            return IntegerPartitions._do_partition(
                total, maxelements, tuple(around), maxdz
            )
        else:
            return IntegerPartitions._do_partition(total, maxelements)
