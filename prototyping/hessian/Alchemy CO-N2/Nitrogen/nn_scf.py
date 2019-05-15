#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:51:43 2019

@author: giorgiod
"""

from horton import*
import numpy as np


#################################################################################  RHF---

def rhf(distance,basis_set='cc-pvtz'):
    natoms=np.array([7,7] )
    coordinates=np.array([[-distance/2.,0.,0.],[distance/2.,0.,0.]])
    mol=IOData(title='NN')
    mol.numbers=natoms
    mol.coordinates=coordinates
    mol.pseudo_numbers=[7.,7.]
#print (mol.numbers)
    obasis=get_gobasis(mol.coordinates,mol.numbers,basis_set)
    
    lf=DenseLinalgFactory(obasis.nbasis)
    
    #orbital integrals
    olp=obasis.compute_overlap(lf)
    kin=obasis.compute_kinetic(lf)
    na=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
    er=obasis.compute_electron_repulsion(lf)
    exp=lf.create_expansion()
    guess_core_hamiltonian(olp, kin, na, exp)
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
    dm = exp.to_dm()
    # - SCF solver
    scf_solver = EDIIS2SCFSolver(1e-10,maxiter=300)
    scf_solver(ham, lf, olp, occ_model, dm)            
    return(ham.cache['energy'])
    
    
#################################################################################  UHF -- 

def uhf(distance,basis_set='cc-pvtz'):
    natoms=np.array([7,7] )
    coordinates=np.array([[-distance/2.,0.,0.],[distance/2.,0.,0.]])
    mol=IOData(title='NN')
    mol.numbers=natoms
    mol.coordinates=coordinates
    mol.pseudo_numbers=[7.,7.]
#print (mol.numbers)

    obasis=get_gobasis(mol.coordinates,mol.numbers,basis_set)

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
    dm_alpha = exp_alpha.to_dm()
    dm_beta = exp_beta.to_dm()
    # - SCF solver
    scf_solver = EDIIS2SCFSolver(1e-10,maxiter=300)
    scf_solver(ham, lf, olp, occ_model, dm_alpha, dm_beta)    
    return(ham.cache['energy'])

#################################################################################  PBE ---

def pbe(distance,basis_set='cc-pvtz'):    
    natoms=np.array([7,7] )
    coordinates=np.array([[-distance/2.,0.,0.],[distance/2.,0.,0.]])
    mol=IOData(title='NN')
    mol.numbers=natoms
    mol.coordinates=coordinates
    mol.pseudo_numbers=[7.,7.]

    obasis=get_gobasis(mol.coordinates,mol.numbers,basis_set)
    lf=DenseLinalgFactory(obasis.nbasis)
    
    #orbital integrals
    olp=obasis.compute_overlap(lf)
    kin=obasis.compute_kinetic(lf)
    na=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
    er=obasis.compute_electron_repulsion(lf)
    
    exp_alpha=lf.create_expansion()
    exp_beta=lf.create_expansion()
    
    guess_core_hamiltonian(olp, kin, na, exp_alpha,exp_beta)
    #exp_alpha.randomize()
    #exp_beta.randomize()
    
    occ_model=AufbauOccModel(7,7)
    occ_model.assign(exp_alpha, exp_beta)
    grid=BeckeMolGrid(mol.coordinates,mol.numbers,mol.numbers, random_rotate=False)

    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        UTwoIndexTerm(kin, 'kin'),
        UDirectTerm(er, 'hartree'),
        UGridGroup(obasis, grid, [
            ULibXCGGA('x_pbe'),
            ULibXCGGA('c_pbe'),
        ]),
        UTwoIndexTerm(na, 'ne'),
    ]
    ham = UEffHam(terms, external)
 
    dm_alpha = exp_alpha.to_dm()
    dm_beta = exp_beta.to_dm()
    # - SCF solver
    scf_solver = EDIIS2SCFSolver(1e-10,maxiter=300)
    scf_solver(ham, lf, olp, occ_model, dm_alpha, dm_beta)            
    return(ham.cache['energy'])
    
###################################################################################### PBE0 --
    
def pbe0(distance,basis_set='cc-pvtz'):    
    natoms=np.array([7,7] )
    coordinates=np.array([[-distance/2.,0.,0.],[distance/2.,0.,0.]])
    mol=IOData(title='NN')
    mol.numbers=natoms
    mol.coordinates=coordinates
    mol.pseudo_numbers=[7.,7.]

    obasis=get_gobasis(mol.coordinates,mol.numbers,basis_set)
    lf=DenseLinalgFactory(obasis.nbasis)
    
    #orbital integrals
    olp=obasis.compute_overlap(lf)
    kin=obasis.compute_kinetic(lf)
    na=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
    er=obasis.compute_electron_repulsion(lf)
    
    exp_alpha=lf.create_expansion()
    exp_beta=lf.create_expansion()
    
    guess_core_hamiltonian(olp, kin, na, exp_alpha,exp_beta)
    #exp_alpha.randomize()
    #exp_beta.randomize()
    
    occ_model=AufbauOccModel(7,7)
    occ_model.assign(exp_alpha, exp_beta)
    grid=BeckeMolGrid(mol.coordinates,mol.numbers,mol.numbers, random_rotate=False)
    
    libxc_term = ULibXCHybridGGA('xc_pbe0_13') #gga_x_pbeint
    external = {'nn': compute_nucnuc(mol.coordinates, mol.pseudo_numbers)}
    terms = [
        UTwoIndexTerm(kin, 'kin'),
        UDirectTerm(er, 'hartree'),
        UGridGroup(obasis, grid, [libxc_term]),
        UExchangeTerm(er, 'x_hf', libxc_term.get_exx_fraction()),
        UTwoIndexTerm(na, 'ne'),
    ]
    ham = UEffHam(terms, external)
 
    dm_alpha = exp_alpha.to_dm()
    dm_beta = exp_beta.to_dm()
    # - SCF solver
    scf_solver = EDIIS2SCFSolver(1e-10,maxiter=300)
    scf_solver(ham, lf, olp, occ_model, dm_alpha, dm_beta)            
    return(ham.cache['energy'])
    
 ###################################################################################### B3LYP 
    
def b3lyp(distance,basis_set='cc-pvtz'):    
    natoms=np.array([7,7] )
    coordinates=np.array([[-distance/2.,0.,0.],[distance/2.,0.,0.]])
    mol=IOData(title='NN')
    mol.numbers=natoms
    mol.coordinates=coordinates
    mol.pseudo_numbers=[7.,7.]

    obasis=get_gobasis(mol.coordinates,mol.numbers,basis_set)
    lf=DenseLinalgFactory(obasis.nbasis)
    
    #orbital integrals
    olp=obasis.compute_overlap(lf)
    kin=obasis.compute_kinetic(lf)
    na=obasis.compute_nuclear_attraction(mol.coordinates,mol.pseudo_numbers,lf)
    er=obasis.compute_electron_repulsion(lf)
    
    exp_alpha=lf.create_expansion()
    exp_beta=lf.create_expansion()
    
    guess_core_hamiltonian(olp, kin, na, exp_alpha,exp_beta)
    #exp_alpha.randomize()
    #exp_beta.randomize()
    
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
 
    dm_alpha = exp_alpha.to_dm()
    dm_beta = exp_beta.to_dm()
    # - SCF solver
    scf_solver = EDIIS2SCFSolver(1e-10,maxiter=300)
    scf_solver(ham, lf, olp, occ_model, dm_alpha, dm_beta)            
    return(ham.cache['energy'])
    






