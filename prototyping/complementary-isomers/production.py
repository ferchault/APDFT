#!/usr/bin/env python

import sys
import qml
import pyscf
import numpy as np
import pandas as pd
from pyscf import gto, dft

fn = sys.argv[1]
def get_electronic_energy(nuclear_charges, coordinates):
    mol = gto.Mole(verbose=0)
    mol.build(atom=list(zip(nuclear_charges, coordinates)), basis='631G')
    calc = pyscf.scf.RHF(mol)
    return calc.scf()-mol.energy_nuc()
def get_electronic_potential(nuclear_charges, coordinates, atoms):
    mol = gto.Mole(verbose=0)
    mol.build(atom=list(zip(nuclear_charges, coordinates)), basis='631G')
    calc = pyscf.scf.RHF(mol)
    total = calc.scf()
    
    dm1 = calc.make_rdm1()
    dm1_ao = np.einsum('pi,ij,qj->pq', calc.mo_coeff, dm1, calc.mo_coeff.conj())
    
    epns = []
    for site in atoms:
        mol.set_rinv_orig_(mol.atom_coords()[site])
        epns.append(np.matmul(dm1_ao, mol.intor('int1e_rinv')).trace())
        
    return epns
def get_site_similarity(c):
    atoms = np.where(c.nuclear_charges == 6)[0]
    a = qml.representations.generate_coulomb_matrix(c.nuclear_charges, c.coordinates, size=c.natoms, sorting='unsorted')
    s = np.zeros((c.natoms, c.natoms))
    s[np.tril_indices(c.natoms)] = a
    d = np.diag(s)
    s += s.T
    s[np.diag_indices(c.natoms)] = d
    sorted_elements = [np.sort(_) for _ in s[atoms]]
    ret = []
    for i in range(len(atoms)):
            for j in range(i+1, len(atoms)):
                dist = np.linalg.norm(sorted_elements[i] - sorted_elements[j])
                ret.append([atoms[i], atoms[j], dist])
    return atoms, ret
def scan_mol(fn):
    c = qml.Compound(fn)
    atoms, iterator = get_site_similarity(c)
    epns = get_electronic_potential(c.nuclear_charges, c.coordinates, atoms)
    for i, j, dist in iterator:
        try:
            q = c.nuclear_charges.copy()
            q[i] = 5
            q[j] = 7
            up = get_electronic_energy(q, c.coordinates)
            uprep = qml.representations.generate_coulomb_matrix(q, c.coordinates, size=c.natoms, sorting='row-norm')
            q = c.nuclear_charges.copy()
            q[i] = 7
            q[j] = 5
            dn = get_electronic_energy(q, c.coordinates)
            dnrep = qml.representations.generate_coulomb_matrix(q, c.coordinates, size=c.natoms, sorting='row-norm')
            gap = up-dn
            molsim = np.linalg.norm(dnrep - uprep)
            print fn.split('_')[-1], i, j, dist, gap, molsim, epns[list(atoms).index(i)], epns[list(atoms).index(j)]
        except:
            continue
scan_mol(fn)

