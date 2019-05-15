#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:51:43 2019

@author: giorgiod
"""

from horton import*

mol=IOData.from_file('co.xyz')
print (mol.numbers)
obasis=get_gobasis(mol.coordinates,mol.numbers,'sto-3G')

lf=DenseLinalgFactory(obasis.nbasis)

#orbital integrals
olp=obasis.compute_overlap(lf)
kin=obasis.compute_kinetic(lf)
na=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
er=obasis.compute_electron_repulsion(lf)

exp_alpha=lf.create_expansion()
exp_beta=lf.create_expansion()
#print(exp_alpha.coeffs)
print(exp_alpha.coeffs.shape)

exp_alpha.randomize()
exp_beta.randomize()

guess_core_hamiltonian(olp, kin, na, exp_alpha,exp_beta)
#print(exp_alpha.coeffs)

occ_model=AufbauOccModel(7,7)

external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
terms = [
    UTwoIndexTerm(kin, 'kin'),
    UDirectTerm(er, 'hartree'),
    UExchangeTerm(er, 'x_hf'),
    UTwoIndexTerm(na, 'ne'),
]
ham = UEffHam(terms, external)


# Converge WFN with Optimal damping algorithm (ODA) SCF
# - Construct the initial density matrix (needed for ODA).

#dm_alpha = exp_alpha.to_dm()
#dm_beta = exp_beta.to_dm()
# - SCF solver
scf_solver = PlainSCFSolver(1e-6)
scf_solver(ham, lf, olp, occ_model, exp_alpha, exp_beta)

#
#scfs=PlainSCFSolver(1e-10,maxiter=400)
#scfs(ham ,lf, ovl, occ_model, exp_alpha, exp_beta)





