#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:39:17 2019

@author: misa
"""

from CPMD_algo import CPMD
from gpaw.projections import Projections
from gpaw import GPAW
from ase import Atoms
from gpaw.eigensolvers import CG
from gpaw.mixer import Mixer
from gpaw import setup_paths
setup_paths.insert(0, '/home/misa/APDFT/prototyping/gpaw/OFDFT/setups')
from matplotlib import pyplot as plt
import numpy as np
import math
from gpaw.forces import calculate_forces
import os
import cProfile

def create_ref_Calc(d):
    from gpaw.eigensolvers import CG
    from gpaw.mixer import Mixer
    from gpaw import setup_paths
    setup_paths.insert(0, '/home/misa/APDFT/prototyping/gpaw/OFDFT/setups')
    a = 12
    c = a/2

    d_list = np.linspace(d, d, 1) # bond distance
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
                    setups=name, txt='Calc_ref.out')
        
            
        molecule.set_calculator(Calc_ref)
        energy_arr[idx] = molecule.get_total_energy()
        Calc_ref.write('Calc_ref_32_gpts_d_'+ str(d) +'.gpw', mode='all')
    return(Calc_ref)

def create_ref_Calc_64_gpt(d):
    from gpaw.eigensolvers import CG
    from gpaw.mixer import Mixer
    from gpaw import setup_paths
    setup_paths.insert(0, '/home/misa/APDFT/prototyping/gpaw/OFDFT/setups')
    a = 12
    c = a/2

    d_list = np.linspace(d, d, 1) # bond distance
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
        
        Calc_ref = GPAW(gpts=(64, 64, 64),
                    xc=xcname,
                    maxiter=500,
                    eigensolver=eigensolver,
                    mixer=mixer,
                    setups=name, txt='Calc_ref.out')
            
        molecule.set_calculator(Calc_ref)
        energy_arr[idx] = molecule.get_total_energy()
        Calc_ref.write('Calc_ref_64_gpts_d_'+ str(d) +'.gpw', mode='all')
    return(Calc_ref)
    
def reference(d, gpts):
    """ load gpw file for initial wavefunction or calculate if gpw-file does not exist
        d: H-H bond distance
        gpts: number of grid points
    """
    current_dir = os.getcwd()
    if gpts == 32:
        if os.path.isfile( os.path.join(current_dir, 'Calc_ref_32_gpts_d_'+ str(d) +'.gpw') ):
            Calc_ref = GPAW( os.path.join(current_dir, 'Calc_ref_32_gpts_d_'+ str(d) +'.gpw') )
        else:
            Calc_ref = create_ref_Calc(d)
    elif gpts == 64:
        if os.path.isfile( os.path.join(current_dir, 'Calc_ref_64_gpts_d_'+ str(d) +'.gpw') ):
            Calc_ref = GPAW( os.path.join(current_dir, 'Calc_ref_64_gpts_d_'+ str(d) +'.gpw') )
        else:
            Calc_ref = create_ref_Calc_64_gpt(d)
    print('Reference intialized!')
    return Calc_ref

Calc_ref = reference(2.0, 32)

# initialize
kwargs_mol = {'symbols':Calc_ref.atoms.symbols.get_chemical_formula(), 'cell':Calc_ref.atoms.cell.diagonal(), 'pbc':Calc_ref.atoms.pbc }
coords = Calc_ref.atoms.get_positions()
gpts = Calc_ref.get_number_of_grid_points()
xc = Calc_ref.get_xc_functional()
maxiter = 500
lambda_coeff = float(Calc_ref.parameters['setups'][Calc_ref.parameters['setups'].find('lambda_')+7:])
eigensolver = CG(tw_coeff=lambda_coeff)
mixer = Mixer()
setups = 'lambda_' + str(lambda_coeff)
txt = 'output_test.txt'
kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}

occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
mu = 1000
dt = 0.1
niter_max = 10

# Create target directory & all intermediate directories if don't exists
path = '/home/misa/APDFT/prototyping/gpaw/CPMD/results/'+'test_dt_'+str(dt)+'-mu_'+str(mu)
if not os.path.exists(path):
    os.makedirs(path)

CPMD_obj = CPMD(kwargs_Calc, kwargs_mol, occupation_numbers, mu, dt, niter_max, pseudo_wf, coords, path)

# test run
CPMD_obj.run()
#CPMD_obj.save_all()

#cProfile.run('CPMD_obj.run()', 'stats')
#CPMD_obj.save_all()
#import pstats
#p = pstats.Stats('stats')
#p.strip_dirs()
#p.sort_stats('cumulative').print_stats(20)

#Calc_ref = create_ref_Calc(2.0)
#forces_ref_before_update = calculate_forces(Calc_ref.wfs, Calc_ref.density, Calc_ref.hamiltonian)
#nproj_a = [setup.ni for setup in Calc_ref.wfs.setups] # atomic density matirx
#for kpt in Calc_ref.wfs.kpt_u:
#
#    kpt.P = Projections(
#                Calc_ref.wfs.bd.nbands, nproj_a,
#                kpt.P.atom_partition,
#                Calc_ref.wfs.bd.comm,
#                collinear=True, spin=0, dtype=Calc_ref.wfs.dtype)
#
#kpt.psit.matrix_elements(Calc_ref.wfs.pt, out=kpt.P)    
#Calc_ref.wfs.calculate_atomic_density_matrices(Calc_ref.density.D_asp)
#Calc_ref.density.calculate_pseudo_density(Calc_ref.wfs) # electron density
#Calc_ref.density.interpolate_pseudo_density()
#Calc_ref.density.calculate_pseudo_charge() # charge density
#Calc_ref.hamiltonian.update_pseudo_potential(Calc_ref.density) # calculate effective potential
#Calc_ref.hamiltonian.restrict_and_collect(Calc_ref.hamiltonian.vt_sg, Calc_ref.hamiltonian.vt_sG) # restrict to coarse grid
#vt_G_ref = Calc_ref.hamiltonian.gd.collect(Calc_ref.hamiltonian.vt_sG[0], broadcast=True) # get effective potential           
#kinetic_energy_op_ref = np.zeros(kpt.psit_nG[0].shape, dtype = float) # calculate dT/drho
#Calc_ref.wfs.kin.apply(kpt.psit_nG[0], kinetic_energy_op_ref, phase_cd=None)
#kinetic_energy_op_ref = kinetic_energy_op_ref/kpt.psit_nG[0] # scaling for OFDFT (see paper Lehtomaeki)
#dE_drho_ref = kinetic_energy_op_ref + vt_G_ref
#
#
## atomic forces
##W_aL = Calc_ref.hamiltonian.calculate_atomic_hamiltonians(Calc_ref.density)
##atomic_energies = Calc_ref.hamiltonian.update_corrections(Calc_ref.density, W_aL)
#
#Calc_ref.hamiltonian.update(Calc_ref.density)
#magmom_a=Calc_ref.atoms.get_initial_magnetic_moments()        
#magmom_av = np.zeros((len(Calc_ref.atoms), 3))
#magmom_av[:, 2] = magmom_a
#Calc_ref.create_occupations(magmom_av[:, 2].sum(), True)
#print(Calc_ref.hamiltonian.get_energy(Calc_ref.occupations))