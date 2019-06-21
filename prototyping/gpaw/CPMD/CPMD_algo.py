#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:46:51 2019

@author: misa

handles CPMD simulation

- calculates dE_drho from new nuclear coordinates and pseudo wavefunction from previous CPMD run

"""

from gpaw import GPAW
from ase import Atoms
from gpaw.projections import Projections
import numpy as np
from gpaw import setup_paths
setup_paths.insert(0, '/home/misa/APDFT/prototyping/gpaw/OFDFT/setups')

class CPMD():
    Calc_obj = None
    kinetic_energy_gradient = None
    effective_potential = None
    dE_drho = None
    
    def __init__(self):
        self.Calc_obj = None
        self.kinetic_energy_gradient = None
        self.effective_potential = None
        self.dE_drho = None

###############################################################################
###                             get dE/drho                                 ###
###############################################################################
        
    # create Calculator with correct nuclei position, DFT functional and wavefunction, density and hamiltonian objects
    def initialize_Calc_basics(self, kwargs_calc, kwargs_mol, coord_nuclei):
        Molecule = Atoms(positions = coord_nuclei, **kwargs_mol) # molecule with new positions of nuclei
        self.Calc_obj = GPAW(**kwargs_calc) # Calc object with calculation parameters
        
        # initialize everything (especially wfs, density, hamiltonian)
        self.Calc_obj.initialize(Molecule)
        self.Calc_obj.initialize_positions(Molecule)
        nlcao, nrand = self.Calc_obj.wfs.initialize(self.Calc_obj.density, self.Calc_obj.hamiltonian,
                                                   self.Calc_obj.spos_ac)
        
    # set correct wave function, and calculate density and pseudo charge density from wave function
    def intialize_Calc_electronic(self, pseudo_wf, occupation_numbers):
        self.update_wf(pseudo_wf, occupation_numbers) # add new wave function, update partial wave coefficients
        self.Calc_obj.wfs.calculate_atomic_density_matrices(self.Calc_obj.density.D_asp) # get correct atomic density matrix
        self.calculate_density()
        self.Calc_obj.density.calculate_pseudo_charge()
        
    # assign correct wave function from previous CPMD calulation to the wave function object
    # and then recalculate integral of projector functions with pseudo wavefunction; needed for update of atomic density matrix
    def update_wf(self, pseudo_wf, occupation_numbers):
        for kpt in self.Calc_obj.wfs.kpt_u: # add wavefunction array
            kpt.psit_nG[0] = pseudo_wf
            kpt.f_n = occupation_numbers
            
        self.update_projectors(self.Calc_obj)
        self.Calc_obj.wfs.mykpts = self.Calc_obj.wfs.kpt_u # just in case mykpts will be accessed somewhere later
            
    # calculates \braket{projector}{pseudo wave function}
    def update_projectors(self, Calc):        
        nproj_a = [setup.ni for setup in Calc.wfs.setups]
        for kpt in Calc.wfs.kpt_u:
    
            kpt.P = Projections(
                        Calc.wfs.bd.nbands, nproj_a,
                        kpt.P.atom_partition,
                        Calc.wfs.bd.comm,
                        collinear=True, spin=Calc.wfs.nspins, dtype=Calc.wfs.dtype)
    
            kpt.psit.matrix_elements(Calc.wfs.pt, out=kpt.P)
            
    # calculate electron density on coarse and fine grid
    def calculate_density(self):
        # calculate pseudo dens on coarse grid from wavefunction and core density
        self.Calc_obj.density.calculate_pseudo_density(self.Calc_obj.wfs) # coarse grid
        # interpolate to fine grid
        self.Calc_obj.density.interpolate_pseudo_density() # should give the same result as:
        # calc_NEW.density.nt_sg = calc_NEW.density.distribute_and_interpolate(calc_NEW.density.nt_sG, calc_NEW.density.nt_sg)
     
        
    def calculate_effective_potential(self):
        # calculate potential
        self.Calc_obj.hamiltonian.update_pseudo_potential(self.Calc_obj.density)
        # restrict to coarse grid
        self.Calc_obj.hamiltonian.restrict_and_collect(self.Calc_obj.hamiltonian.vt_sg, self.Calc_obj.hamiltonian.vt_sG)
        
    def get_effective_potential(self):
        vt_G = self.Calc_obj.hamiltonian.gd.collect(self.Calc_obj.hamiltonian.vt_sG[0], broadcast=True)
        return(vt_G)
        
    ### APPLIES \hat{T} to pseudo wavefunction, not to pseudo density; core density not included
    ### WORKS ONLY IF ONE K-POINT IS USED
    ################################# SCALING BY LAMBDA!!!!!!!!!!!!!!!!!!!!!!!!!!
    def calculate_kinetic_en_gradient(self): 
        kinetic_energy_op = None
        for kpt in self.Calc_obj.wfs.kpt_u:
            kinetic_energy_op = np.zeros(kpt.psit_nG[0].shape, dtype = float)
            self.Calc_obj.wfs.kin.apply(kpt.psit_nG[0], kinetic_energy_op, phase_cd=None)
            kinetic_energy_op = kinetic_energy_op/kpt.psit_nG[0] # scaling for OFDFT (see paper Lehtomaeki)
        return(kinetic_energy_op)
        
    # initializes everything and then calculates energy-density gradient dE/drho
    def calculate_dE_drho(self, kwargs_calc, kwargs_mol, coord_nuclei, pseudo_wf, occupation_numbers):
        self.initialize_Calc_basics(kwargs_calc, kwargs_mol, coord_nuclei) # ini step 1
        self.intialize_Calc_electronic(pseudo_wf, occupation_numbers) # ini step 2
        self.calculate_effective_potential() # calculate effective potential
        self.effective_potential = self.get_effective_potential()  
        self.kinetic_energy_gradient = self.calculate_kinetic_en_gradient()
        self.dE_drho = self.kinetic_energy_gradient + self.effective_potential


###############################################################################
        
        