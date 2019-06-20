#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:53:33 2019

@author: misa
"""

from gpaw.projections import Projections
from ase import Atoms
from gpaw import GPAW, PW
from gpaw.mixer import Mixer
from gpaw.eigensolvers import CG
from gpaw.poisson import PoissonSolver
import numpy as np
from gpaw import setup_paths
setup_paths.insert(0, '/home/misa/APDFT/prototyping/gpaw/OFDFT/setups')
from ase.units import Bohr, Hartree, create_units

###############################################################################
##      initialize empty GPAW object with updated nuclei coords              ##
###############################################################################

h = 0.2
a = 12.0
c = a/2
d = 1.3

# XC functional + kinetic functional (minus the Tw contribution) to be used
xcname = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
# Fraction of Tw
lambda_coeff = 1.0
name = 'lambda_{0}'.format(lambda_coeff)

elements = 'H2'

mixer = Mixer()
eigensolver = CG(tw_coeff=lambda_coeff)
poissonsolver=PoissonSolver()



molecule_updated_pos = Atoms(elements,
                     positions=[(c,c,c-d/2), (c, c, c+d/2)] ,
                     cell=(a,a,a), pbc=True)

calc_NEW = GPAW(gpts=(32, 32, 32),
                xc=xcname,
                maxiter=500,
                eigensolver=eigensolver,
                mixer=mixer,
                setups=name, txt='/home/misa/APDFT/prototyping/gpaw/CPMD/new_gradient.txt')


calc_NEW.initialize(molecule_updated_pos)
calc_NEW.initialize_positions(molecule_updated_pos)
nlcao, nrand = calc_NEW.wfs.initialize(calc_NEW.density, calc_NEW.hamiltonian,
                                           calc_NEW.spos_ac)

###############################################################################
##                      ref calc                                             ##
###############################################################################

d_list = np.linspace(1.3, 1.3, 1) # bond distance

energy_arr = np.empty(len(d_list))
calc1 = None
for idx, d in enumerate(d_list):
    molecule = Atoms(elements,
                     positions=[(c,c,c-d/2), (c, c, c+d/2)] ,
                     cell=(a,a,a), pbc=True)
    
    calc1 = GPAW(gpts=(32, 32, 32),
                xc=xcname,
                maxiter=500,
                eigensolver=eigensolver,
                mixer=mixer,
                setups=name, txt='test.txt')
        
    molecule.set_calculator(calc1)
    
    energy_arr[idx] = molecule.get_total_energy()


debug=calc_NEW
debug=calc1


###############################################################################
##   add pseudo wavefunction from CPMD to empty gpaw object                  ##
###############################################################################

pseudo_wf=calc1.wfs.kpt_u[0].psit_nG[0] # get this after SCF calculation and do the propagation with this object (or scaling by occupation?)

# if we try calc_NEW.wfs.kpt_u[0].psit_nG = pseudo_wf=calc1.wfs.kpt_u[0].psit_nG we get an attribute error
# this also sets calc_NEW.wfs.kpt_u[0].psit.array to the correct value
calc_NEW.wfs.kpt_u[0].psit_nG[0] = pseudo_wf

# also calculation of occupation numbers necessary
calc_NEW.wfs.kpt_u[0].f_n = np.array([2.0])

###############################################################################
##   calculate new atomic density matrix for pseudo charge density           ##
###############################################################################

# GPAW projectors multiplied by wavefunction already?
nproj_a = [setup.ni for setup in calc_NEW.wfs.setups]
kpt=calc_NEW.wfs.kpt_u[0]
calc_NEW.wfs.kpt_u[0].P = Projections(
                    calc_NEW.wfs.bd.nbands, nproj_a,
                    calc_NEW.wfs.kpt_u[0].P.atom_partition,
                    calc_NEW.wfs.bd.comm,
                    collinear=True, spin=calc_NEW.wfs.nspins, dtype=calc_NEW.wfs.dtype)

calc_NEW.wfs.kpt_u[0].psit.matrix_elements(calc_NEW.wfs.pt, out=calc_NEW.wfs.kpt_u[0].P)


calc_NEW.wfs.mykpts = calc_NEW.wfs.kpt_u # update mykpts

debug=calc_NEW

# updating of D_asp
calc_NEW.wfs.calculate_atomic_density_matrices(calc_NEW.density.D_asp)

###############################################################################
##                         calculate density                                 ##
###############################################################################

# calculate pseudo dens on coarse grid from wavefunction and core density
calc_NEW.density.calculate_pseudo_density(calc_NEW.wfs) # coarse grid
calc1.density.calculate_pseudo_density(calc1.wfs) # for comparison with calc_NEW
comp_nt_sG = np.allclose(calc_NEW.density.nt_sG, calc1.density.nt_sG)

# interpolate to fine grid
calc_NEW.density.interpolate_pseudo_density() # should give the same result as:
# calc_NEW.density.nt_sg = calc_NEW.density.distribute_and_interpolate(calc_NEW.density.nt_sG, calc_NEW.density.nt_sg)
calc1.density.interpolate_pseudo_density() # comparison
comp_nt_sg=np.allclose(calc_NEW.density.nt_sg, calc1.density.nt_sg) # comparison

###############################################################################
##                         calculate charge density                          ##
###############################################################################

# calculate charge density rhot_g
calc_NEW.density.calculate_pseudo_charge()
calc1.density.calculate_pseudo_charge()
comp_rhot_g = np.allclose(calc_NEW.density.rhot_g, calc1.density.rhot_g)

###############################################################################
##                         calculate hamiltonian                             ##
###############################################################################

calc_NEW.hamiltonian.update_pseudo_potential(calc_NEW.density)
calc1.hamiltonian.update_pseudo_potential(calc1.density)

comp_vbar_g=np.allclose(calc_NEW.hamiltonian.vbar_g, calc1.hamiltonian.vbar_g)
comp_vHt_g=np.allclose(calc_NEW.hamiltonian.vHt_g, calc1.hamiltonian.vHt_g)

compare_xc = np.allclose(calc1.hamiltonian.vt_sg, calc_NEW.hamiltonian.vt_sg)

debug=calc_NEW

# get projections and calculate density matrix
# wfs.calculate_atomic_density_matrices(Density.D_asp)
#
# D_sp = 1
# ni = 5
# D_sii = np.zeros((len(D_sp), ni, ni))
# f_n = 2
# P_ni0=calc1.wfs.kpt_u[0].P[0]
# D_sii[0] += np.dot(P_ni0.T.conj() * f_n, P_ni0).real
# D_sp[:] = [struct.pack(D_ii) for D_ii in D_sii]

#calculate_atomic_density_matrices_k_point(self, D_sii, kpt, a, f_n)


#calc.density.D_asp = update
#calc.density.nt_sg = from_CPMD
#calc.density.calculate_pseudo_charge()

#calc.density.nt_sG = np.empty((1, 64, 64, 64), dtype=float)
#calc.set_positions(calc.atoms) #maybe only this one?
#calc.density.nt_sg = np.empty((1, 128, 128, 128), dtype=float)
#calc.density.nt_sg = calc.density.distribute_and_interpolate(calc.density.nt_sG, calc.density.nt_sg)

#calc.density.D_asp=calc1.density.D_asp
#calc.density.nt_sg=calc1.density.nt_sg
#calc.density.calculate_pseudo_charge()




