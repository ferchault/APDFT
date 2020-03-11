#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:51:43 2019

@author: giorgiod
"""

from horton import*
import numpy as np

def uhf(distance,pseudo_numbers,basis_set='cc-pvtz'):
    natoms=np.array([6,8] )
    coordinates=np.array([[-distance/2.,0.,0.],[distance/2.,0.,0.]])
    
    mol=IOData(title='CO')
    mol.numbers=natoms
    mol.coordinates=coordinates
    mol.pseudo_numbers=pseudo_numbers
#print (mol.numbers)
    obasis1=get_gobasis(mol.coordinates,mol.numbers,basis_set)
    obasis2=get_gobasis(mol.coordinates,np.array([7,7]),basis_set)
    obasis=GOBasis.concatenate(obasis1,obasis2)
    
    lf=DenseLinalgFactory(obasis.nbasis)
    
    #orbital integrals
    olp=obasis.compute_overlap(lf)
    kin=obasis.compute_kinetic(lf)
    na=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
    er=obasis.compute_electron_repulsion(lf)

    exp_alpha=lf.create_expansion()
    exp_beta=lf.create_expansion()
    
    guess_core_hamiltonian(olp, kin, na, exp_alpha,exp_beta)

    occ_model=AufbauOccModel(7,7)
    occ_model.assign(exp_alpha, exp_beta)
    
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
    
    dm_alpha = exp_alpha.to_dm()
    dm_beta = exp_beta.to_dm()
    # - SCF solver
    scf_solver = EDIIS2SCFSolver(1e-9,maxiter=300)
    scf_solver(ham, lf, olp, occ_model, dm_alpha, dm_beta)
    
    grid=BeckeMolGrid(mol.coordinates,mol.numbers,mol.numbers, random_rotate=False)
    rho_alpha = obasis.compute_grid_density_dm(dm_alpha, grid.points)
    rho_beta = obasis.compute_grid_density_dm(dm_beta, grid.points)
    rho=rho_alpha+rho_beta

    return(ham.cache['energy'],rho)
#
#scfs=PlainSCFSolver(1e-10,maxiter=400)
#scfs(ham ,lf, ovl, occ_model, exp_alpha, exp_beta)

def b3lyp(distance,pseudo_numbers,basis_set='cc-pvtz'):    
    natoms=np.array([6,8] )
    coordinates=np.array([[-distance/2.,0.,0.],[distance/2.,0.,0.]])
    mol=IOData(title='CO')
    mol.numbers=natoms
    mol.coordinates=coordinates
    mol.pseudo_numbers=pseudo_numbers
#print (mol.numbers)
    obasis1=get_gobasis(mol.coordinates,mol.numbers,basis_set)
    obasis2=get_gobasis(mol.coordinates,np.array([7,7]),basis_set)
    obasis=GOBasis.concatenate(obasis1,obasis2)    
    lf=DenseLinalgFactory(obasis.nbasis)
    
    #orbital integrals
    olp=obasis.compute_overlap(lf)
    kin=obasis.compute_kinetic(lf)
    na=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
    er=obasis.compute_electron_repulsion(lf)    
    exp_alpha=lf.create_expansion()
    exp_beta=lf.create_expansion()
    
    guess_core_hamiltonian(olp, kin, na, exp_alpha,exp_beta)    
    occ_model=AufbauOccModel(7,7)
    occ_model.assign(exp_alpha, exp_beta)
    grid=BeckeMolGrid(mol.coordinates,mol.numbers,mol.numbers, random_rotate=False)
    
    libxc_term = ULibXCHybridGGA('xc_b3lyp')
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        UTwoIndexTerm(kin, 'kin'),
        UDirectTerm(er, 'hartree'),
        UGridGroup(obasis, grid, [libxc_term]),
        UExchangeTerm(er, 'x_hf', libxc_term.get_exx_fraction()),
        UTwoIndexTerm(na, 'ne'),
    ]
    ham = UEffHam(terms, external)
  
    
    # Converge WFN with Optimal damping algorithm (ODA) SCF
    # - Construct the initial density matrix (needed for ODA).
#    
    dm_alpha = exp_alpha.to_dm()
    dm_beta = exp_beta.to_dm()
    # - SCF solver
    scf_solver = EDIIS2SCFSolver(1e-9,maxiter=300)
    scf_solver(ham, lf, olp, occ_model, dm_alpha, dm_beta)
    
    rho_alpha = obasis.compute_grid_density_dm(dm_alpha, grid.points)
    rho_beta = obasis.compute_grid_density_dm(dm_beta, grid.points)
    rho=rho_alpha+rho_beta

    
    return(ham.cache['energy'],rho)
    
    
def rhf(distance,pseudo_numbers,basis_set='cc-pvtz'):
    natoms=np.array([6,8] )
    coordinates=np.array([[-distance/2.,0.,0.],[distance/2.,0.,0.]])
    
    mol=IOData(title='CO')
    mol.numbers=natoms
    mol.coordinates=coordinates
    mol.pseudo_numbers=pseudo_numbers
#print (mol.numbers)
    obasis1=get_gobasis(mol.coordinates,mol.numbers,basis_set)
    obasis2=get_gobasis(mol.coordinates,np.array([7,7]),basis_set)
    obasis=GOBasis.concatenate(obasis1,obasis2)
    
    lf=DenseLinalgFactory(obasis.nbasis)
    
    #orbital integrals
    olp=obasis.compute_overlap(lf)
    kin=obasis.compute_kinetic(lf)
    na=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
    er=obasis.compute_electron_repulsion(lf)

    exp=lf.create_expansion()
    
    guess_core_hamiltonian(olp, kin, na, exp)
    #exp_alpha.randomize()
    #exp_beta.randomize()
    
    occ_model=AufbauOccModel(7)
    occ_model.assign(exp)
    
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        RTwoIndexTerm(kin, 'kin'),
        RDirectTerm(er, 'hartree'),
        RExchangeTerm(er, 'x_hf'),
        RTwoIndexTerm(na, 'ne'),
    ]
    ham = REffHam(terms, external)
    
    
    # Converge WFN with Optimal damping algorithm (ODA) SCF
    # - Construct the initial density matrix (needed for ODA).
    
    dm = exp.to_dm()
    # - SCF solver
    scf_solver = EDIIS2SCFSolver(1e-9,maxiter=300)
    scf_solver(ham, lf, olp, occ_model, dm)
    
    grid=BeckeMolGrid(mol.coordinates,mol.numbers,mol.numbers, random_rotate=False)
    rho = 2*obasis.compute_grid_density_dm(dm, grid.points)

    return(ham.cache['energy'],rho)


