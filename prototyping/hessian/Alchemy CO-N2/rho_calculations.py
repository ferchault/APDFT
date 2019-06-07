#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:05:22 2019

@author: giorgiod
"""
import numpy as np
from horton import*
from co_fun import SCFonCO

distance=1.*angstrom
pseudo_numbers=[6.,8.]

natoms=np.array([6,8] )
coordinates=np.array([[-distance/2.,0.,0.],[distance/2.,0.,0.]])

mol=IOData(title='CO')
mol.numbers=natoms
mol.coordinates=coordinates
mol.pseudo_numbers=pseudo_numbers

obasis1=get_gobasis(mol.coordinates,mol.numbers,'sto-3G')
obasis2=get_gobasis(mol.coordinates,np.flipud(mol.numbers),'sto-3G')
obasis=GOBasis.concatenate(obasis1,obasis2)



lf=DenseLinalgFactory(obasis.nbasis)

#orbital integrals
olp=obasis.compute_overlap(lf)
kin=obasis.compute_kinetic(lf)
na=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
er=obasis.compute_electron_repulsion(lf)

exp_alpha=lf.create_expansion()
exp_beta=lf.create_expansion()
#print(exp_alpha.coeffs)


exp_alpha.randomize()
exp_beta.randomize()

guess_core_hamiltonian(olp, kin, na, exp_alpha,exp_beta)
exp_alpha.rotate_random()
exp_beta.rotate_random()
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


#dm_alpha = exp_alpha.to_dm()
#dm_beta = exp_beta.to_dm()
# - SCF solver
scf_solver = PlainSCFSolver(1e-5,maxiter=500)
i=0
#scf_solver(ham, lf, olp, occ_model, exp_alpha, exp_beta)
while  i<10 : 
    try:
#            print 'attempt No.', i
        scf_solver(ham, lf, olp, occ_model, exp_alpha, exp_beta)
        break
    except NoSCFConvergence:
        exp_alpha.rotate_random()
        exp_alpha.rotate_2orbitals()
        exp_beta.rotate_random()
        exp_beta.rotate_2orbitals()
        i+=1
        
e_CO=ham.cache['energy']
e_N2=SCFonCO(distance,[7.,7.])[0]
grid=BeckeMolGrid(mol.coordinates,mol.numbers,mol.numbers, random_rotate=False)

dm_alpha = exp_alpha.to_dm()
dm_beta = exp_beta.to_dm()


rho_alpha = obasis.compute_grid_density_dm(dm_alpha, grid.points)
rho_beta = obasis.compute_grid_density_dm(dm_beta, grid.points)
rho=rho_alpha+rho_beta

#r = (grid.points[:, 0]**2 + grid.points[:, 1]**2 + grid.points[:, 2]**2)**0.5
#ds1 = np.linalg.norm(grid.points - mol.coordinates[0], axis=1)
V1=((grid.points[:, 0]-mol.coordinates[0][0])**2 + (grid.points[:, 1]-mol.coordinates[0][1])**2 + (grid.points[:, 2]-mol.coordinates[0][2])**2)**-0.5
V2=((grid.points[:, 0]-mol.coordinates[1][0])**2 + (grid.points[:, 1]-mol.coordinates[1][1])**2 + (grid.points[:, 2]-mol.coordinates[1][2])**2)**-0.5
anm1=V2-V1

h=.1


dRho=(SCFonCO(distance,[6.+h/2,8.-h/2])[1]-SCFonCO(distance,[6.-h/2,8.+h/2])[1])/h

d1=grid.integrate(rho,anm1)
d2=grid.integrate(dRho,anm1)
print d1, d2
e_ext=e_CO+d1+d2/2.

print 'energy for CO = ', e_CO
print 'energy for N2 = ', e_N2
print 'energy extimate =', e_ext


#print(grid.integrate(rho*na))
