#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:23:43 2019

@author: misa

Testing of CPMD class
"""

from CPMD_algo import CPMD
from gpaw.projections import Projections
from gpaw import GPAW
from ase import Atoms
from gpaw.eigensolvers import CG
from gpaw.mixer import Mixer
from gpaw import setup_paths
setup_paths.insert(0, '/home/misa/APDFT/prototyping/gpaw/OFDFT/setups')

import numpy as np
from gpaw.forces import calculate_forces
###############################################################################
##                      ref calc                                             ##
###############################################################################
def create_ref_Calc():
    from gpaw.eigensolvers import CG
    from gpaw.mixer import Mixer
    from gpaw import setup_paths
    setup_paths.insert(0, '/home/misa/APDFT/prototyping/gpaw/OFDFT/setups')
    a = 12.0
    c = a/2
    d = 1.3
    d_list = np.linspace(1.3, 1.3, 1) # bond distance
    # XC functional + kinetic functional (minus the Tw contribution) to be used
    xcname = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    # Fraction of Tw
    lambda_coeff = 1.0
    name = 'lambda_{0}'.format(lambda_coeff)
    elements = 'H2'
    mixer = Mixer()
    eigensolver = CG(tw_coeff=lambda_coeff)
    
    energy_arr = np.empty(len(d_list))
    Calc_ref = None
    for idx, d in enumerate(d_list):
        molecule = Atoms(elements,
                         positions=[(c,c,c-d/2), (c, c, c+d/2)] ,
                         cell=(a,a,a), pbc=True)
        
        Calc_ref = GPAW(gpts=(32, 32, 32),
                    xc=xcname,
                    maxiter=500,
                    eigensolver=eigensolver,
                    mixer=mixer,
                    setups=name, txt='test.txt')
            
        molecule.set_calculator(Calc_ref)
        energy_arr[idx] = molecule.get_total_energy()
        #Calc_ref.write('result_32_gpts_d_'+str(d)+'.gpw', mode='all')
    return(Calc_ref)

###############################################################################
### test initialization with keywords
def TEST_intialization_with_kwargs_():
    # Molecule
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    Molecule = Atoms(positions = coords, **kwargs_mol)
    
    ## GPAW calculator
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    Calc_NEW = GPAW(**kwargs_Calc)


###############################################################################
### test initialize_Calc_basics()
def TEST_initialize_Calc_basics_():
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)

###############################################################################
### test if wavefunction array correctly initialized

def TEST_wavefunction_array_initialized_():
    # create reference calc object
    Calc_ref = create_ref_Calc()
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    
    # test
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    
    hard_comp_psit_nG = np.alltrue(CPMD_obj.Calc_obj.wfs.kpt_u[0].psit_nG == Calc_ref.wfs.kpt_u[0].psit_nG)
    hard_comp_psit_array = np.alltrue(CPMD_obj.Calc_obj.wfs.kpt_u[0].psit.array == Calc_ref.wfs.kpt_u[0].psit.array)
    
    if hard_comp_psit_nG and hard_comp_psit_array:
        print('Wavefunction array correctly initialized!')
    else:
        print('Wavefunction array NOT correctly initialized!')
        
    if hard_comp_psit_nG and hard_comp_psit_array:
        return('Passed')
    else:
        return('Failed')


###############################################################################
### test if electron density correctly initialized

def TEST_electron_density_initialized_():
# create reference calc object
    Calc_ref = create_ref_Calc()
    Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs)
    Calc_ref.density.interpolate_pseudo_density()
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    
    # test
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    
    hard_comp_nt_sG = np.alltrue(CPMD_obj.Calc_obj.density.nt_sG == Calc_ref.density.nt_sG)
    hard_comp_nt_sg = np.alltrue(CPMD_obj.Calc_obj.density.nt_sg == Calc_ref.density.nt_sg)
    
    soft_comp_nt_sG = np.allclose(CPMD_obj.Calc_obj.density.nt_sG, Calc_ref.density.nt_sG)
    soft_comp_nt_sg = np.allclose(CPMD_obj.Calc_obj.density.nt_sg, Calc_ref.density.nt_sg)
    
    if hard_comp_nt_sG and hard_comp_nt_sg:
        print('Electron densities identical!')
    else:
        print('Electron density NOT identical!')
    
    if soft_comp_nt_sG and soft_comp_nt_sg:
        print('Electron densities correctly initialized!')
    else:
        print('Electron density NOT correctly initialized!')
        
    if hard_comp_nt_sG and hard_comp_nt_sg:
        return('Passed')
    else:
        return('Failed')
        
###############################################################################
def TEST_projectors_initialized_():
    # create reference calc object
    Calc_ref = create_ref_Calc()
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups]
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=Calc_ref.wfs.nspins, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
        
    
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    # test
    P_new = CPMD_obj.Calc_obj.wfs.kpt_u[0].P.array
    P_ref = Calc_ref.wfs.kpt_u[0].P.array
    hard = np.alltrue(P_new == P_ref)
    soft = np.allclose(P_new, P_ref)
    
    if hard:
        print('Projectors identical!')
    else:
        print('Projectors NOT identical!')
    
    if soft:
        print('Projectors correctly initialized!')
    else:
        print('Projectors NOT correctly initialized!')
        
    if hard:
        return('Passed')
    else:
        return('Failed')

###############################################################################
def TEST_atomic_density_matrix_initialized_():
    #create reference calc object
    Calc_ref = create_ref_Calc()
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups]
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=Calc_ref.wfs.nspins, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)
    
    Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    
    
    
    hard_comp_D_asp0 = np.alltrue(CPMD_obj.Calc_obj.density.D_asp.data[0] == Calc_ref.density.D_asp.data[0])
    hard_comp_D_asp1 = np.alltrue(CPMD_obj.Calc_obj.density.D_asp.data[1] == Calc_ref.density.D_asp.data[1])
    
    soft_comp_D_asp0 = np.allclose(CPMD_obj.Calc_obj.density.D_asp.data[0], Calc_ref.density.D_asp.data[0])
    soft_comp_D_asp1 = np.allclose(CPMD_obj.Calc_obj.density.D_asp.data[1], Calc_ref.density.D_asp.data[1])
    
    if hard_comp_D_asp0 and hard_comp_D_asp1:
        print('Atomic density matrices identical!')
    else:
        print('Atomic density matrices NOT identical!')
    
    if soft_comp_D_asp0 and soft_comp_D_asp1:
        print('Atomic density matrix correctly initialized!')
    else:
        print('Atomic density matrix NOT correctly initialized!')

    if hard_comp_D_asp0 and hard_comp_D_asp1:
        return('Passed')
    else:
        return('Failed')

###############################################################################    
### test if cahrge density correctly initialized

def TEST_charged_density_initialized_():
# create reference calc object
    Calc_ref = create_ref_Calc()
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups] # atomic density matirx
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=Calc_ref.wfs.nspins, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)    
    Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
    
    Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs) # electron density
    Calc_ref.density.interpolate_pseudo_density()
    
    Calc_ref.density.calculate_pseudo_charge() # charge density
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    
    # test
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    
    hard = np.alltrue(CPMD_obj.Calc_obj.density.rhot_g == Calc_ref.density.rhot_g)    
    soft = np.allclose(CPMD_obj.Calc_obj.density.rhot_g, Calc_ref.density.rhot_g)
    
    if hard:
        print('Charge densities identical!')
    else:
        print('Charge densities NOT identical!')
    
    if soft:
        print('Charge density correctly initialized!')
    else:
        print('Charge density NOT correctly initialized!')
        
    if hard:
        return('Passed')
    else:
        return('Failed')
        
###############################################################################
def TEST_intialize_Calc_electronic_():
    
    print('\n##################################################################\n')
          
    wf_array = TEST_wavefunction_array_initialized_()
    density = TEST_electron_density_initialized_()
    projectors = TEST_projectors_initialized_()
    atomic_density_matrix = TEST_atomic_density_matrix_initialized_()
    charge_density = TEST_charged_density_initialized_()
    
    print('\n##################################################################\n')
          
    results = [wf_array, density, projectors, atomic_density_matrix, charge_density]
    
    if results == ['Passed', 'Passed', 'Passed', 'Passed', 'Passed']:
        print('Electronic properties correctly initialized')
    else:
        print('Failed')
    
    
###############################################################################
#   test if correct energy is calculated for pseudo density    
def TEST_pseudo_energy_():
    # create reference calc object
    Calc_ref = create_ref_Calc()
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups] # atomic density matirx
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=Calc_ref.wfs.nspins, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)    
    Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
    Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs) # electron density
    Calc_ref.density.interpolate_pseudo_density()
    Calc_ref.density.calculate_pseudo_charge() # charge density
    energy_ref = Calc_ref.hamiltonian.update_pseudo_potential(Calc_ref.density)
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    
    # for this test
    energy_new = CPMD_obj.Calc_obj.hamiltonian.update_pseudo_potential(CPMD_obj.Calc_obj.density)
    
    # assert
    hard = np.alltrue( energy_new == energy_ref )
    soft = np.allclose( energy_new, energy_ref )
    
    if hard:
        print('Energies are identical!')
    else:
        print('Energies are NOT identical!')
    
    if soft:
        print('Correct energies calculated!')
    else:
        print('NOT correct energies calculated!')
    
    if hard:
        return('Passed')
    else:
        return('Failed')
  
###############################################################################
# test if vbar correctly initialized
    
def TEST_vbar_():
    # create reference calc object
    Calc_ref = create_ref_Calc()
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups] # atomic density matirx
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=Calc_ref.wfs.nspins, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)    
    Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
    Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs) # electron density
    Calc_ref.density.interpolate_pseudo_density()
    Calc_ref.density.calculate_pseudo_charge() # charge density
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    
    # test: vbar
    hard = np.alltrue(CPMD_obj.Calc_obj.hamiltonian.vbar_g == Calc_ref.hamiltonian.vbar_g)
    soft = np.allclose(CPMD_obj.Calc_obj.hamiltonian.vbar_g, Calc_ref.hamiltonian.vbar_g)
    
    if hard:
        print('Vbar is identical!')
    else:
        print('Vbar is NOT identical!')
    
    if soft:
        print('Correct vbar calculated!')
    else:
        print('NOT correct vbar calculated!')
    
    if hard:
        return('Passed')
    else:
        return('Failed')
    
###############################################################################
# test if vHtg (electrostatic potential) correctly initialized
        
def TEST_vHtg_():
    # create reference calc object
    Calc_ref = create_ref_Calc()
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups] # atomic density matirx
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=Calc_ref.wfs.nspins, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)    
    Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
    Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs) # electron density
    Calc_ref.density.interpolate_pseudo_density()
    Calc_ref.density.calculate_pseudo_charge() # charge density
    Calc_ref.hamiltonian.update_pseudo_potential(Calc_ref.density) # calculate vHtg
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    
    # test: calculate vHtg
    CPMD_obj.Calc_obj.hamiltonian.update_pseudo_potential(CPMD_obj.Calc_obj.density)
    
    # assert
    hard = np.alltrue(CPMD_obj.Calc_obj.hamiltonian.vHt_g == Calc_ref.hamiltonian.vHt_g)
    soft = np.allclose(CPMD_obj.Calc_obj.hamiltonian.vHt_g, Calc_ref.hamiltonian.vHt_g)
    
    if hard:
        print('vHt_g is identical!')
    else:
        print('vHt_g is NOT identical!')
    
    if soft:
        print('Correct vHt_g calculated!')
    else:
        print('NOT correct vHt_g calculated!')
    
    if hard:
        return('Passed')
    else:
        return('Failed')
    
    
###############################################################################
# vxc if vHtg (electrostatic potential) correctly initialized

def TEST_vxc_():
    # create reference calc object
    Calc_ref = create_ref_Calc()
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups] # atomic density matirx
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=Calc_ref.wfs.nspins, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)    
    Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
    Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs) # electron density
    Calc_ref.density.interpolate_pseudo_density()
    Calc_ref.density.calculate_pseudo_charge() # charge density
    Calc_ref.hamiltonian.vt_sg[0].fill(0.0) # make sure that no other values (vbar e.g. in vt_sg)
    e_xc_ref = Calc_ref.hamiltonian.xc.calculate(Calc_ref.hamiltonian.finegd, Calc_ref.density.nt_sg, Calc_ref.hamiltonian.vt_sg) # calculate vxc
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    
    # test: calculation of vxc
    CPMD_obj.Calc_obj.hamiltonian.vt_sg[0].fill(0.0) # make sure that no other values (vbar e.g. in vt_sg)
    e_xc_new = CPMD_obj.Calc_obj.hamiltonian.xc.calculate(CPMD_obj.Calc_obj.hamiltonian.finegd, CPMD_obj.Calc_obj.density.nt_sg, CPMD_obj.Calc_obj.hamiltonian.vt_sg) # calculate vxc
    
    # assert
    hard = np.alltrue(CPMD_obj.Calc_obj.hamiltonian.vt_sg == Calc_ref.hamiltonian.vt_sg)
    soft = np.allclose(CPMD_obj.Calc_obj.hamiltonian.vt_sg, Calc_ref.hamiltonian.vt_sg)
    
    if hard:
        print('vt_sg is identical!')
    else:
        print('vt_sg is NOT identical!')
    
    if soft:
        print('Correct vt_sg calculated!')
    else:
        print('NOT correct vt_sg calculated!')
    
    if hard:
        return('Passed')
    else:
        return('Failed')

###############################################################################
# test if effective potential on fine grid is correctly initialized
def TEST_hamiltonian_update_pseudo_potential_():
    # create reference calc object
    Calc_ref = create_ref_Calc()
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups] # atomic density matirx
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=Calc_ref.wfs.nspins, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)    
    Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
    Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs) # electron density
    Calc_ref.density.interpolate_pseudo_density()
    Calc_ref.density.calculate_pseudo_charge() # charge density
    Calc_ref.hamiltonian.update_pseudo_potential(Calc_ref.density) # calculate effective potential
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    
    # test: calculate effective potential
    CPMD_obj.Calc_obj.hamiltonian.update_pseudo_potential(CPMD_obj.Calc_obj.density)
    
    # assert
    hard = np.alltrue(CPMD_obj.Calc_obj.hamiltonian.vt_sg == Calc_ref.hamiltonian.vt_sg)
    soft = np.allclose(CPMD_obj.Calc_obj.hamiltonian.vt_sg, Calc_ref.hamiltonian.vt_sg)
    
    if hard:
        print('Effective potential vt_sg is identical!')
    else:
        print('Effective potential vt_sg is NOT identical!')
    
    if soft:
        print('Correct effective potential vt_sg calculated!')
    else:
        print('NOT correct effective potential vt_sg calculated!')
    
    if hard:
        return('Passed')
    else:
        return('Failed')
    
    
###############################################################################
# test restriction of vt_sg from fine grid to coarse grid vt_sG
def TEST_restrict_and_collect():
        # create reference calc object
    Calc_ref = create_ref_Calc()
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups] # atomic density matirx
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=Calc_ref.wfs.nspins, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)    
    Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
    Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs) # electron density
    Calc_ref.density.interpolate_pseudo_density()
    Calc_ref.density.calculate_pseudo_charge() # charge density
    Calc_ref.hamiltonian.update_pseudo_potential(Calc_ref.density) # calculate effective potential
    Calc_ref.hamiltonian.restrict_and_collect(Calc_ref.hamiltonian.vt_sg, Calc_ref.hamiltonian.vt_sG) # restrict to coarse grid
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    CPMD_obj.Calc_obj.hamiltonian.update_pseudo_potential(CPMD_obj.Calc_obj.density)
    
    # test: restriction to coarse grid
    CPMD_obj.Calc_obj.hamiltonian.restrict_and_collect(CPMD_obj.Calc_obj.hamiltonian.vt_sg, CPMD_obj.Calc_obj.hamiltonian.vt_sG)
    
    # assert
    hard = np.alltrue(CPMD_obj.Calc_obj.hamiltonian.vt_sG == Calc_ref.hamiltonian.vt_sG)
    soft = np.allclose(CPMD_obj.Calc_obj.hamiltonian.vt_sG, Calc_ref.hamiltonian.vt_sG)
    
    if hard:
        print('Effective potential on coarse grid vt_sG is identical!')
    else:
        print('Effective potential on coarse grid vt_sG is NOT identical!')
    
    if soft:
        print('Correct restriction of vt_sg to vt_sG!')
    else:
        print('NOT correct restriction of vt_sg to vt_sG')
    
    if hard:
        return('Passed')
    else:
        return('Failed')
    
        
###############################################################################
        
# test calculate_effective_potential()

def TEST_calculate_effective_potential_():
    # create reference calc object
    Calc_ref = create_ref_Calc()
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups] # atomic density matirx
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=Calc_ref.wfs.nspins, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)    
    Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
    Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs) # electron density
    Calc_ref.density.interpolate_pseudo_density()
    Calc_ref.density.calculate_pseudo_charge() # charge density
    Calc_ref.hamiltonian.update_pseudo_potential(Calc_ref.density) # calculate effective potential
    Calc_ref.hamiltonian.restrict_and_collect(Calc_ref.hamiltonian.vt_sg, Calc_ref.hamiltonian.vt_sG) # restrict to coarse grid
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    
    # test: calculate_effective_potential()
    CPMD_obj.calculate_effective_potential()
    
    # assert
    hard = np.alltrue(CPMD_obj.Calc_obj.hamiltonian.vt_sG == Calc_ref.hamiltonian.vt_sG)
    soft = np.allclose(CPMD_obj.Calc_obj.hamiltonian.vt_sG, Calc_ref.hamiltonian.vt_sG)
    
    if hard:
        print('Effective potential on coarse grid vt_sG is identical!')
    else:
        print('Effective potential on coarse grid vt_sG is NOT identical!')
    
    if soft:
        print('Correct restriction of vt_sg to vt_sG!')
    else:
        print('NOT correct restriction of vt_sg to vt_sG')
    
    if hard:
        return('Passed')
    else:
        return('Failed')
    
        
###############################################################################
        
# test get_effective_potential
def TEST_get_effective_potential_():
    # create reference calc object
    Calc_ref = create_ref_Calc()
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups] # atomic density matirx
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=Calc_ref.wfs.nspins, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)    
    Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
    Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs) # electron density
    Calc_ref.density.interpolate_pseudo_density()
    Calc_ref.density.calculate_pseudo_charge() # charge density
    Calc_ref.hamiltonian.update_pseudo_potential(Calc_ref.density) # calculate effective potential
    Calc_ref.hamiltonian.restrict_and_collect(Calc_ref.hamiltonian.vt_sg, Calc_ref.hamiltonian.vt_sG) # restrict to coarse grid
    vt_G_ref = Calc_ref.hamiltonian.gd.collect(Calc_ref.hamiltonian.vt_sG[0], broadcast=True) # get effective potential
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    CPMD_obj.calculate_effective_potential()
    
    # test: get_effective_potential()
    vt_G_test = CPMD_obj.get_effective_potential()
    
    # assert
    hard = np.alltrue(vt_G_test == vt_G_ref)
    soft = np.allclose(vt_G_test, vt_G_ref)
    
    if hard:
        print('get_effective_potential() is identical!')
    else:
        print('get_effective_potential() is NOT identical!')
    
    if soft:
        print('get_effective_potential() returns correct values!')
    else:
        print('get_effective_potential() DOES NOT return correct values!')
    
    if hard:
        return('Passed')
    else:
        return('Failed')

###############################################################################
# test calculation of kinetic energy gradient

def TEST_calculate_kinetic_en_gradient_():
    # create reference calc object
    Calc_ref = create_ref_Calc()
    kinetic_energy_op_ref = None # calculate dT/drho
    for kpt in Calc_ref.wfs.kpt_u:
        kinetic_energy_op_ref = np.zeros(kpt.psit_nG[0].shape, dtype = float)
        Calc_ref.wfs.kin.apply(kpt.psit_nG[0], kinetic_energy_op_ref, phase_cd=None)
        kinetic_energy_op_ref = kinetic_energy_op_ref/kpt.psit_nG[0] # scaling for OFDFT (see paper Lehtomaeki)
    
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    CPMD_obj.initialize_Calc_basics(kwargs_Calc, kwargs_mol, coords)
    pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
    occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
    CPMD_obj.intialize_Calc_electronic(pseudo_wf, occupation_numbers)
    
    # test: calcualte kinetic energy gradient
    kinetic_energy_op_test = CPMD_obj.calculate_kinetic_en_gradient()
    
    # assert
    hard = np.alltrue(kinetic_energy_op_test == kinetic_energy_op_ref)
    soft = np.allclose(kinetic_energy_op_test, kinetic_energy_op_ref)
    
    if hard:
        print('Kinetic energy derivative is identical!')
    else:
        print('Kinetic energy derivative is NOT identical!')
    
    if soft:
        print('calculate_kinetic_en_gradient() returns correct values!')
    else:
        print('calculate_kinetic_en_gradient() DOES NOT return correct values!')
    
    if hard:
        return('Passed')
    else:
        return('Failed')
    
###############################################################################

### test  calculate_dE_drho
def TEST_calculate_dE_drho_():
    # create reference calc object
    Calc_ref = create_ref_Calc()
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups] # atomic density matirx
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=Calc_ref.wfs.nspins, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)    
    Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
    Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs) # electron density
    Calc_ref.density.interpolate_pseudo_density()
    Calc_ref.density.calculate_pseudo_charge() # charge density
    Calc_ref.hamiltonian.update_pseudo_potential(Calc_ref.density) # calculate effective potential
    Calc_ref.hamiltonian.restrict_and_collect(Calc_ref.hamiltonian.vt_sg, Calc_ref.hamiltonian.vt_sG) # restrict to coarse grid
    vt_G_ref = Calc_ref.hamiltonian.gd.collect(Calc_ref.hamiltonian.vt_sG[0], broadcast=True) # get effective potential           
    kinetic_energy_op_ref = np.zeros(kpt.psit_nG[0].shape, dtype = float) # calculate dT/drho
    Calc_ref.wfs.kin.apply(kpt.psit_nG[0], kinetic_energy_op_ref, phase_cd=None)
    kinetic_energy_op_ref = kinetic_energy_op_ref/kpt.psit_nG[0] # scaling for OFDFT (see paper Lehtomaeki)
    dE_drho_ref = kinetic_energy_op_ref + vt_G_ref
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'    
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    #CPMD_obj.initialize_GPAW_calculator(kwargs_Calc, kwargs_mol, coords, Calc_ref.wfs.kpt_u[0].psit_nG[0], Calc_ref.wfs.kpt_u[0].f_n)
    
    # test: calculate dE_drho
    CPMD_obj.calculate_dE_drho(kwargs_Calc, kwargs_mol, coords, Calc_ref.wfs.kpt_u[0].psit_nG[0], Calc_ref.wfs.kpt_u[0].f_n)

    # assert    
    hard = np.alltrue(CPMD_obj.dE_drho == dE_drho_ref)
    soft = np.allclose(CPMD_obj.dE_drho, dE_drho_ref)
    
    if hard:
        print('dE/drho is identical!')
    else:
        print('dE/drho is NOT identical!')
    
    if soft:
        print('calculate_dE_drho() returns correct values!')
    else:
        print('calculate_dE_drho() DOES NOT return correct values!')

    if hard:
        return('Passed')
    else:
        return('Failed')
        
###############################################################################
# test calculation of atomic forces
def TEST_calculate_forces_el_nuc_():
    # create reference calc object
    Calc_ref = create_ref_Calc()
    forces_ref_before_update = calculate_forces(Calc_ref.wfs, Calc_ref.density, Calc_ref.hamiltonian)
    nproj_a = [setup.ni for setup in Calc_ref.wfs.setups] # atomic density matirx
    for kpt in Calc_ref.wfs.kpt_u:
    
        kpt.P = Projections(
                    Calc_ref.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc_ref.wfs.bd.comm,
                    collinear=True, spin=0, dtype=Calc_ref.wfs.dtype)
    
    kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)    
    Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
    Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs) # electron density
    Calc_ref.density.interpolate_pseudo_density()
    Calc_ref.density.calculate_pseudo_charge() # charge density
    Calc_ref.hamiltonian.update_pseudo_potential(Calc_ref.density) # calculate effective potential
    Calc_ref.hamiltonian.restrict_and_collect(Calc_ref.hamiltonian.vt_sg, Calc_ref.hamiltonian.vt_sG) # restrict to coarse grid
    vt_G_ref = Calc_ref.hamiltonian.gd.collect(Calc_ref.hamiltonian.vt_sG[0], broadcast=True) # get effective potential           
    kinetic_energy_op_ref = np.zeros(kpt.psit_nG[0].shape, dtype = float) # calculate dT/drho
    Calc_ref.wfs.kin.apply(kpt.psit_nG[0], kinetic_energy_op_ref, phase_cd=None)
    kinetic_energy_op_ref = kinetic_energy_op_ref/kpt.psit_nG[0] # scaling for OFDFT (see paper Lehtomaeki)
    dE_drho_ref = kinetic_energy_op_ref + vt_G_ref
    
    
    # atomic forces
    W_aL = Calc_ref.hamiltonian.calculate_atomic_hamiltonians(Calc_ref.density)
    atomic_energies = Calc_ref.hamiltonian.update_corrections(Calc_ref.density, W_aL)
    Calc_ref.wfs.eigensolver.subspace_diagonalize(Calc_ref.hamiltonian, Calc_ref.wfs, Calc_ref.wfs.kpt_u[0]) 
    forces_ref = calculate_forces(Calc_ref.wfs, Calc_ref.density, Calc_ref.hamiltonian)
    
    #def TEST_intialize_Calc_electronic_():
    # initialize
    kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
    coords = [(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
    gpts = (32, 32, 32)
    xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
    maxiter = 500
    lambda_coeff = 1.0
    eigensolver = CG(tw_coeff=lambda_coeff)
    mixer = Mixer()
    setups = 'lambda_' + str(lambda_coeff)
    txt = 'output_test.txt'
    kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}
    CPMD_obj = CPMD()
    
    # test: calculate dE_drho
    CPMD_obj.calculate_forces_el_nuc(kwargs_Calc, kwargs_mol, coords, Calc_ref.wfs.kpt_u[0].psit_nG[0], Calc_ref.wfs.kpt_u[0].f_n)
    
    # assert    
    hard = np.alltrue(CPMD_obj.atomic_forces == forces_ref)
    soft = np.allclose(CPMD_obj.atomic_forces, forces_ref)
    
    if hard:
        print('Forces are identical!')
    else:
        print('Forces are NOT identical!')
    
    if soft:
        print('calculate_forces_el_nuc() returns correct atomic forces!')
    else:
        print('calculate_forces_el_nuc() DOES NOT return correct atomic forces')
    
    if hard:
        return('Passed')
    else:
        return('Failed')
