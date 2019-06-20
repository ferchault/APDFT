#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 08:18:02 2019

@author: misa
"""

from gpaw.projections import Projections
from ase import Atoms
from gpaw import GPAW
from gpaw.mixer import Mixer
from gpaw.eigensolvers import CG
from gpaw.poisson import PoissonSolver
import numpy as np
from gpaw import setup_paths
setup_paths.insert(0, '/home/misa/APDFT/prototyping/gpaw/OFDFT/setups')


# XC functional + kinetic functional (minus the Tw contribution) to be used
xcname = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
# Fraction of Tw
lambda_coeff = 1.0
name = 'lambda_{0}'.format(lambda_coeff)
elements = 'H2'
mixer = Mixer()
eigensolver = CG(tw_coeff=lambda_coeff)
poissonsolver=PoissonSolver()



def initialize_everything(Calc, Molecule):
    Calc.initialize(Molecule)
    Calc.initialize_positions(Molecule)
    nlcao, nrand = Calc.wfs.initialize(Calc.density, Calc.hamiltonian,
                                               Calc.spos_ac)

def update_wf(Calc, pseudo_wf, f_n):
    for kpt in Calc.wfs.kpt_u:
        kpt.psit_nG[0] = pseudo_wf
        kpt.f_n = f_n
        
def update_projectors(Calc):        
    nproj_a = [setup.ni for setup in Calc.wfs.setups]
    for kpt in Calc.wfs.kpt_u:

        kpt.P = Projections(
                    Calc.wfs.bd.nbands, nproj_a,
                    kpt.P.atom_partition,
                    Calc.wfs.bd.comm,
                    collinear=True, spin=Calc.wfs.nspins, dtype=Calc.wfs.dtype)

    kpt.psit.matrix_elements(Calc.wfs.pt, out=kpt.P)

def update_mykpts(Calc): # mykpts will maybe accessed somewhere in gpaw without this mykpts are random numbers
    Calc.wfs.mykpts = Calc.wfs.kpt_u # update mykpts
    
def update_atomic_density_matrix(Calc):
    Calc.wfs.calculate_atomic_density_matrices(Calc.density.D_asp)
    
def calculate_density(Calc):
    # calculate pseudo dens on coarse grid from wavefunction and core density
    Calc.density.calculate_pseudo_density(Calc.wfs) # coarse grid
    # interpolate to fine grid
    Calc.density.interpolate_pseudo_density() # should give the same result as:
    # calc_NEW.density.nt_sg = calc_NEW.density.distribute_and_interpolate(calc_NEW.density.nt_sG, calc_NEW.density.nt_sg)
    
def calculate_effective_potential(Calc):
    # calculate potential
    Calc.hamiltonian.update_pseudo_potential(Calc_NEW.density)
    # restrict to coarse grid
    Calc.hamiltonian.vt_sG.fill(0.0)
    Calc.hamiltonian.restrict_and_collect(Calc.hamiltonian.vt_sg, Calc.hamiltonian.vt_sG)
    
### APPLIES \hat{T} to pseudo wavefunction, not to pseudo density; core density not included
### WORKS ONLY IF ONE K-POINT IS USED
def calculate_kinetic_en_gradient(Calc): 
    kinetic_energy_op = None
    for kpt in Calc.wfs.kpt_u:
        kinetic_energy_op = np.zeros(kpt.psit_nG[0].shape, dtype = float)
        Calc.wfs.kin.apply(kpt.psit_nG[0], kinetic_energy_op, phase_cd=None)
        kinetic_energy_op = kinetic_energy_op/kpt.psit_nG[0] # scaling for OFDFT (see paper)
    return(kinetic_energy_op)
    
    
    

###############################################################################
##                      ref calc                                             ##
###############################################################################
a = 12.0
c = a/2
d = 1.3
d_list = np.linspace(1.3, 1.3, 1) # bond distance

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

debug=Calc_ref


###############################################################################
##      initialize empty GPAW object with updated nuclei coords              ##
###############################################################################

Calc_NEW = GPAW(gpts=(32, 32, 32),
                xc=xcname,
                maxiter=500,
                eigensolver=eigensolver,
                mixer=mixer,
                setups=name, txt='/home/misa/APDFT/prototyping/gpaw/CPMD/new_gradient.txt')
# molecule with correct cell and geometry
a = 12.0
c = a/2
d = 1.3
Molecule_updated_pos = Atoms(elements,
                     positions=[(c,c,c-d/2), (c, c, c+d/2)] ,
                     cell=(a,a,a), pbc=True)

###############################################################################
##   initialize objects (wfs, density,hamiltonian                            ##
###############################################################################
initialize_everything(Calc_NEW, Molecule_updated_pos)

###############################################################################
##   add pseudo wavefunction from CPMD to empty gpaw object                  ##
###############################################################################
pseudo_wf=Calc_ref.wfs.kpt_u[0].psit_nG[0] # get this after SCF calculation and do the propagation with this object (or scaling by occupation?)
f_n = Calc_ref.wfs.kpt_u[0].f_n # get this after SCF calculation
update_wf(Calc_NEW, pseudo_wf, f_n)

###############################################################################
##   calculate new atomic density matrix for pseudo charge density           ##
############################################################################### 
update_projectors(Calc_NEW)    
update_mykpts(Calc_NEW)
update_atomic_density_matrix(Calc_NEW)

###############################################################################
##                         calculate density                                 ##
###############################################################################
calculate_density(Calc_NEW)

###############################################################################
##                         calculate charge density                          ##
###############################################################################
Calc_NEW.density.calculate_pseudo_charge()

###############################################################################
##                         calculate hamiltonian                             ##
###############################################################################
calculate_effective_potential(Calc_NEW) # effective potential
hat_T = calculate_kinetic_en_gradient(Calc_NEW) # scaled kinetic energy gradient

###############################################################################
##                         calculate dE/dRho                                 ##
###############################################################################
dE_dRho_new = hat_T + Calc_NEW.hamiltonian.vt_sG[0]

###############################################################################
##                          TESTING                                          ##
###############################################################################


###                     atomic density matrix                               ###
error0_tot = abs(Calc_NEW.density.D_asp.data[0]-Calc_ref.density.D_asp.data[0])
error1_tot = abs(Calc_NEW.density.D_asp.data[1]-Calc_ref.density.D_asp.data[1])
error_tot_max = max([np.amax(error0_tot), np.amax(error1_tot)])

comp_D_asp0 = np.allclose(Calc_NEW.density.D_asp.data[0], Calc_ref.density.D_asp.data[0])
comp_D_asp0 = np.allclose(Calc_NEW.density.D_asp.data[1], Calc_ref.density.D_asp.data[1])
 # different; due to updating/density mixing in SCF-cycle?
update_projectors(Calc_ref)    
update_mykpts(Calc_ref)
update_atomic_density_matrix(Calc_ref) 

error0_tot_updated = abs(Calc_NEW.density.D_asp.data[0]-Calc_ref.density.D_asp.data[0])
error1_tot_updated = abs(Calc_NEW.density.D_asp.data[1]-Calc_ref.density.D_asp.data[1])
error_tot_max_updated = max([np.amax(error0_tot_updated), np.amax(error1_tot_updated)])

comp_D_asp0_updated = np.allclose(Calc_NEW.density.D_asp.data[0], Calc_ref.density.D_asp.data[0])
comp_D_asp0_updated = np.allclose(Calc_NEW.density.D_asp.data[1], Calc_ref.density.D_asp.data[1])

###############################################################################

###                              density                                    ###
calculate_density(Calc_ref) # for comparison with calc_NEW
comp_nt_sG = np.allclose(Calc_NEW.density.nt_sG, Calc_ref.density.nt_sG)
comp_nt_sg=np.allclose(Calc_NEW.density.nt_sg, Calc_ref.density.nt_sg)
non_zero_diff_nt_sg = np.where(Calc_NEW.density.nt_sg-Calc_ref.density.nt_sg!=0)

###############################################################################

###                             charge density                              ###
Calc_ref.density.calculate_pseudo_charge()
comp_rhot_g = np.allclose(Calc_NEW.density.rhot_g, Calc_ref.density.rhot_g)
non_zero_diff_rhot_g = np.where(Calc_NEW.density.rhot_g-Calc_ref.density.rhot_g!=0)

###############################################################################

###                              hamiltonian, dE/drho                       ###
calculate_effective_potential(Calc_ref)
hat_T_ref = calculate_kinetic_en_gradient(Calc_ref)
dE_dRho_ref = hat_T_ref + Calc_ref.hamiltonian.vt_sG[0]

### kinetic energy gradient
comp_kin = np.allclose(hat_T, hat_T_ref)
hard_comp_kin = np.where(hat_T != hat_T_ref)

### vbar correction
comp_vbar = np.allclose(Calc_NEW.hamiltonian.vbar_g, Calc_ref.hamiltonian.vbar_g)
hard_comp_vbar = np.where(Calc_NEW.hamiltonian.vbar_g != Calc_ref.hamiltonian.vbar_g)

### electrostatic potential
comp_vHt_g = np.allclose(Calc_NEW.hamiltonian.vHt_g, Calc_ref.hamiltonian.vHt_g)
hard_comp_vHt_g = np.where(Calc_NEW.hamiltonian.vHt_g != Calc_ref.hamiltonian.vHt_g)

### effective potential (vbar + vHtg + vxc) on coarse grid
comp_eff_pot = np.allclose(Calc_NEW.get_effective_potential(), Calc_ref.get_effective_potential())
hard_comp_eff_pot = np.where(Calc_NEW.get_effective_potential() != Calc_ref.get_effective_potential())

### Energy-density gradient dE/drho on the coarse grid
comp_dE_drho = np.allclose(dE_dRho_new, dE_dRho_ref)
hard_comp_dE_drho = np.where(dE_dRho_new != dE_dRho_ref)



### exchange correlation potential
Calc_ref.hamiltonian.vt_sg.fill(0.0)
Calc_ref.hamiltonian.xc.calculate(Calc_ref.hamiltonian.finegd, Calc_ref.density.nt_sg, Calc_ref.hamiltonian.vt_sg)
Calc_NEW.hamiltonian.vt_sg.fill(0.0)
Calc_NEW.hamiltonian.xc.calculate(Calc_NEW.hamiltonian.finegd, Calc_NEW.density.nt_sg, Calc_NEW.hamiltonian.vt_sg)
comp_vxc = np.allclose(Calc_NEW.hamiltonian.vt_sg, Calc_ref.hamiltonian.vt_sg)
hard_comp_vxc = np.where(Calc_NEW.hamiltonian.vt_sg!= Calc_ref.hamiltonian.vt_sg)











