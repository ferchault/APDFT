#!/usr/bin/env python
import itertools as it
import os
import glob
import multiprocessing as mp
import functools
import traceback

import numpy as np
import pandas as pd
import basis_set_exchange as bse
import pyscf
from pyscf import dft

import apdft
import apdft.math
import apdft.physics


class Derivative(apdft.physics.APDFT):
    def assign_calculator(self, calculator):
        self._calculator = calculator

    @staticmethod
    def _get_target_name(target):
        return ",".join([apdft.physics.charge_to_label(_) for _ in target])




    def _cached_reader(self, folder):
        if folder not in self._reader_cache:
            rho = self._calculator.get_density_on_grid(folder, self._gridcoords)
            self._reader_cache[folder] = rho
            int_electrons = np.sum(rho * self._gridweights)
            num_electrons = sum(self._nuclear_numbers)
            if abs(num_electrons - int_electrons) > 1e-2:
                apdft.log.log(
                    "Electron count mismatch.",
                    level="error",
                    expected=num_electrons,
                    found=int_electrons,
                    path=folder,
                )

        return self._reader_cache[folder]

    def get_density_derivative(self, sites):
        num_electrons = sum(self._nuclear_numbers)
        if len(sites) == 0:
            return self._cached_reader("QM/order-0/site-all-cc")
        if len(sites) == 1:
            rhoup = self._cached_reader("QM/order-1/site-%d-up" % sites[0])
            rhodn = self._cached_reader("QM/order-1/site-%d-dn" % sites[0])
            return (rhoup - rhodn) / (2 * self._delta)
        if len(sites) == 2:
            i, j = sites
            rho = self.get_density_derivative([])
            rhoiup = self._cached_reader("QM/order-1/site-%d-up" % i)
            rhoidn = self._cached_reader("QM/order-1/site-%d-dn" % i)
            if i != j:
                rhojup = self._cached_reader("QM/order-1/site-%d-up" % j)
                rhojdn = self._cached_reader("QM/order-1/site-%d-dn" % j)
                rhoup = self._cached_reader(
                    "QM/order-2/site-%d-%d-up" % (min(i, j), max(i, j))
                )
                rhodn = self._cached_reader(
                    "QM/order-2/site-%d-%d-dn" % (min(i, j), max(i, j))
                )

            if i == j:
                deriv = (rhoiup + rhoidn - 2 * rho) / (self._delta ** 2)
            else:
                deriv = (
                    rhoup + rhodn + 2 * rho - rhoiup - rhoidn - rhojup - rhojdn
                ) / (2 * self._delta ** 2)
            return deriv
        raise NotImplementedError()

    def get_density_from_reference(self, nuclear_charges):
        return self._cached_reader(
            "QM/comparison-%s" % "-".join(map(str, nuclear_charges))
        )

    

    