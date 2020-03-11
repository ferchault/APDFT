#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:12:02 2019

@author: misa
"""

from CPMD_algo_NEW import CPMD_new
from gpaw.projections import Projections
from gpaw import GPAW
from ase import Atoms
from gpaw.eigensolvers import CG
from gpaw.mixer import Mixer
from gpaw import setup_paths
setup_paths.insert(0, '/home/misa/APDFT/prototyping/gpaw/OFDFT/setups')

from gpaw.forces import calculate_forces
import numpy as np

from functools import partial
from gpaw.matrix import matrix_matrix_multiply as mmm

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


Calc_ref = create_ref_Calc()

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

W_aL = Calc_ref.hamiltonian.calculate_atomic_hamiltonians(Calc_ref.density)
atomic_energies = Calc_ref.hamiltonian.update_corrections(Calc_ref.density, W_aL)

#Calc_ref.wfs.kpt_u[0].P.spin=0
Calc_ref.wfs.eigensolver.subspace_diagonalize(Calc_ref.hamiltonian, Calc_ref.wfs, Calc_ref.wfs.kpt_u[0])

forces_ref = calculate_forces(Calc_ref.wfs, Calc_ref.density, Calc_ref.hamiltonian)
#debug = Calc_ref

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
CPMD_obj = CPMD_new()

# test: calculate dE_drho
CPMD_obj.calculate_forces_el_nuc(kwargs_Calc, kwargs_mol, coords, Calc_ref.wfs.kpt_u[0].psit_nG[0], Calc_ref.wfs.kpt_u[0].f_n)

#CPMD_obj.Calc_obj.wfs.kpt_u[0].eps_n = np.array([-0.11987959630161966])
#CPMD_obj.Calc_obj.wfs.kpt_u[0].P.spin=0
#W_aL = CPMD_obj.Calc_obj.hamiltonian.calculate_atomic_hamiltonians(CPMD_obj.Calc_obj.density)
#atomic_energies = CPMD_obj.Calc_obj.hamiltonian.update_corrections(CPMD_obj.Calc_obj.density, W_aL)
#CPMD_obj.Calc_obj.wfs.eigensolver.initialize(CPMD_obj.Calc_obj.wfs)
#CPMD_obj.Calc_obj.wfs.eigensolver.subspace_diagonalize(CPMD_obj.Calc_obj.hamiltonian, CPMD_obj.Calc_obj.wfs, CPMD_obj.Calc_obj.wfs.kpt_u[0])
#
#
#forces_new = calculate_forces(CPMD_obj.Calc_obj.wfs, CPMD_obj.Calc_obj.density, CPMD_obj.Calc_obj.hamiltonian)

#CPMD_obj.calculate_forces_on_nuclei()
#forces_new = CPMD_obj.atomic_forces


