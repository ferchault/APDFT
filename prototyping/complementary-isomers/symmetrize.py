#!/usr/bin/env python
import sys

import scipy.spatial as scs
import numpy as np
from pymatgen.symmetry import analyzer as psa
from pymatgen.core import Molecule
from pymatgen.io import xyz

def residuals(mol):
	a = psa.PointGroupAnalyzer(mol)
	deviations = []
	coords = mol.cart_coords
	for symop in a.get_symmetry_operations():
		newcoord = symop.operate_multi(coords)
		ds = scs.distance.cdist(coords, newcoord)
		deviations.append(max(ds[np.where(ds < 1e-1)]))
	return np.array(deviations)


basemol = xyz.XYZ.from_file(sys.argv[1]).molecule
initgeo = basemol.cart_coords
initgeo -= initgeo.mean(axis=0)
basegeo = initgeo.copy()
for step in range(1000):
	mol = Molecule(basemol.atomic_numbers, initgeo)
	acc = np.mean(np.log(residuals(mol))/np.log(10))
	if acc < -12:
		break
	print ('Log maximum distance between symmetry equivalent sites:', acc)
	a = psa.PointGroupAnalyzer(mol)
	mol = a.symmetrize_molecule()['sym_mol']
	initgeo = mol.cart_coords
	initgeo -= initgeo.mean(axis=0)

print ('Maximum change in position for a single site:', max(np.linalg.norm(initgeo - basegeo, axis=1)))
finalmol = xyz.XYZ(mol, coord_precision=17)
print (str(finalmol))
