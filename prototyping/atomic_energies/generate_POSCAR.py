#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:12:24 2019

@author: misa
"""

import ase
import ase.io.cif
import ase.io.vasp
import numpy as np
from ase.calculators.vasp import Vasp
import qml
import os

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies')
from explore_qml_data import shift2center


# from numpy arrays
H1 = [5, 5, 4.45]
H2 = [5, 5, 5.55]
boxsize = 10.0
elements = ['H2', 'Li2', 'Be2', 'B2', 'C2', 'N2']
for el in elements:
    mol = ase.Atoms(el, positions = [H1, H2], pbc=True)
    mol.set_cell((boxsize, boxsize, boxsize,   90.00,  90.00,  90.00))
    path = '/home/misa/APDFT/prototyping/atomic_energies/results/different_resevoir/{}/POSCAR'.format(el)
    ase.io.vasp.write_vasp(path, mol)
# from qml compound

#boxsize = 65.0
#b = [45, 50, 55, 60]
#for boxsize in b:
#    
#    path = '/home/misa/datasets/qm9/dsgdb9nsd_001212.xyz'
#    qml_comp = qml.Compound(path)
#
#    box_center = np.array([boxsize/2.0, boxsize/2.0, boxsize/2.0])
#    qml_comp.coordinates = shift2center(qml_comp.coordinates, box_center)
#    
#    # switch positions of carbon oxygens, for order in POTCAR file
#    qml_comp.coordinates[[4,2]] = qml_comp.coordinates[[2,4]]
#    qml_comp.coordinates[[5,3]] = qml_comp.coordinates[[3,5]]
#    
#    qml_comp.atomtypes[2], qml_comp.atomtypes[4] = qml_comp.atomtypes[4], qml_comp.atomtypes[2]
#    qml_comp.atomtypes[3], qml_comp.atomtypes[5] = qml_comp.atomtypes[5], qml_comp.atomtypes[3]
#    
#    #atoms = ase.io.read(path)
#    
#    atoms = ase.Atoms(qml_comp.atomtypes, positions=qml_comp.coordinates, pbc=True)
#    atoms.set_cell((boxsize, boxsize, boxsize,   90.00,  90.00,  90.00))
#    ##numbers = atoms.get_atomic_numbers()
#    ##numbers[70-1] = 9
#    ##numbers[81] = 7
#    ##numbers[126] = 7
#    ##numbers[117] = 9
#    ##atoms.set_atomic_numbers(numbers)
#    ##ase.io.write('up.pdb', atoms)
#    
#    POSCAR_path = "/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/vasp/dsgdb9nsd_001212/box{}".format(boxsize)
#    
#    
#    if not os.path.exists(POSCAR_path):
#        os.mkdir(POSCAR_path)
#    
#    ase.io.vasp.write_vasp("/home/misa/APDFT/prototyping/atomic_energies/results/test_calculations/vasp/dsgdb9nsd_001212/box{}/POSCAR".format(boxsize), atoms)
#calc = Vasp(xc='PBE', lreal=False)
#atoms.set_calculator(calc)
#atoms.get_potential_energy()