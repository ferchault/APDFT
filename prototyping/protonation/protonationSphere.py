#!/usr/bin/env python

import pyscf
import pyscf.gto
import pyscf.qmmm
import pyscf.scf
import pyscf.dft
import pyscf.lib
from pyscf.data import nist
import matplotlib.pyplot as plt
import quadpy
import numpy as np
import sys

def nuclei_nuclei(coordinates, charges, query):
	natoms = len(coordinates)
	ret = 0.0
	for i in range(natoms):
		d = np.linalg.norm((coordinates[i] - query))
		ret += charges[i]  / d
	return ret

def mol_from_file(fn):
	mol = pyscf.gto.Mole()
	lines = open(fn).readlines()
	natoms = int(lines[0].strip())
	lines = lines[2:2+natoms]
	atomspec = []
	for line in lines:
		atomspec.append(' '.join(line.strip().split()[:4]))
	atomspec = '\n'.join(atomspec)
	mol.atom = atomspec
	mol.build()
	return mol

if __name__ == "__main__":
	try:
		xyzfile, radius, basisset = sys.argv[1:]
	except:
		print ("Usage: %s xzyfile radius basisset" % sys.argv[0])
		sys.exit(1)

	mol = mol_from_file(xyzfile)
	mol.basis = basisset
	calc = pyscf.scf.RHF(mol)
	hfe = calc.kernel(verbose=0)
	dm1_ao = calc.make_rdm1()

	scheme = quadpy.sphere.lebedev_031()

	for atomidx, charge in enumerate(mol.atom_charges()):
		if charge == 1:
			continue

		site = mol.atom_coords()[atomidx]/1.88973
		pts = site + scheme.points*float(radius)
		querys = pts*1.88973
		esp = []
		for query in querys:
			mol.set_rinv_orig_(query)
			electronic = np.matmul(dm1_ao, mol.intor("int1e_rinv")).trace()
			nuclear = nuclei_nuclei(mol.atom_coords(), mol.atom_charges(), query)
			esp.append(-electronic + nuclear)

		minpos = pts[np.argmin(esp)]
		print ("Site %d, nuclear charge %d, ESP %f, proton position %f, %f, %f [bohr]" % (atomidx, charge, min(esp), *minpos))