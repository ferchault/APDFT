#!/usr/bin/env python
import ase.io
from ase.calculators.vasp import Vasp

box = ase.io.read('base.cif')
box *= (2,3,2)

special = [49,52,9,12]

newz = [7,9,9,7]
label = 'up'

#newz = [9,7,7,9]
#label = 'dn'

basenumbers = box.get_atomic_numbers().copy()
basenumbers[special] = newz
box.set_atomic_numbers(basenumbers)

calc = Vasp(xc='PBE', lreal=False)
box.set_calculator(calc)
box.get_potential_energy()


