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
from gpaw.forces import calculate_forces
import math

class CPMD():
    Calc_obj = None
    kinetic_energy_gradient = None
    effective_potential = None
    dE_drho = None
    atomic_forces = None
    
    kwargs_calc = None
    kwargs_mol = None
    occupation_numbers = None
    
    # CPMD stuff
    mu = None
    dt = None
    niter_max = None
    
    pseudo_wf = None
    pseudo_wf_previous = None
    coords = None
    coords_previous = None
    
    
    def __init__(self, kwargs_calc=None, kwargs_mol=None, occupation_numbers=None, mu=None, dt=None, niter_max=None, pseudo_wf=None, coords_nuclei=None):
        self.Calc_obj = None
        self.kinetic_energy_gradient = None
        self.effective_potential = None
        self.dE_drho = None
        self.atomic_forces = None
        # set
        self.kwargs_calc = kwargs_calc
        self.kwargs_mol = kwargs_mol
        self.occupation_numbers = occupation_numbers
        self.mu = mu
        self.dt = dt
        self.niter_max = niter_max
        
        self.pseudo_wf = pseudo_wf
        self.pseudo_wf_previous = None
        self.coords = coords_nuclei
        self.coords_previous = None

    def initialize_GPAW_calculator(self, kwargs_calc, kwargs_mol, coord_nuclei, pseudo_wf, occupation_numbers):
        self.initialize_Calc_basics(kwargs_calc, kwargs_mol, coord_nuclei) # ini step 1
        self.intialize_Calc_electronic(pseudo_wf, occupation_numbers) # ini step 2
        
    # create Calculator with correct nuclei position, DFT functional and wavefunction, density and hamiltonian objects
    def initialize_Calc_basics(self, kwargs_calc, kwargs_mol, coord_nuclei):
        self.Calc_obj = None
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
                        collinear=True, spin=0, dtype=Calc.wfs.dtype)
    
            kpt.psit.matrix_elements(Calc.wfs.pt, out=kpt.P)
            
    # calculate electron density on coarse and fine grid
    def calculate_density(self):
        # calculate pseudo dens on coarse grid from wavefunction and core density
        self.Calc_obj.density.calculate_pseudo_density(self.Calc_obj.wfs) # coarse grid
        # interpolate to fine grid
        self.Calc_obj.density.interpolate_pseudo_density() # should give the same result as:
        # calc_NEW.density.nt_sg = calc_NEW.density.distribute_and_interpolate(calc_NEW.density.nt_sG, calc_NEW.density.nt_sg)
        
        
    def get_forces(self, kwargs_calc, kwargs_mol, coord_nuclei, pseudo_wf, occupation_numbers):
        self.initialize_GPAW_calculator(kwargs_calc, kwargs_mol, coord_nuclei, pseudo_wf, occupation_numbers)
        self.calculate_dE_drho()
        self.calculate_forces_on_nuclei()
###############################################################################
###                             get dE/drho                                 ###
###############################################################################
        
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
        

    def calculate_forces_el_nuc(self, kwargs_calc, kwargs_mol, coord_nuclei, pseudo_wf, occupation_numbers):
        """
        this function calculates forces on electron density and nuclei
        I have to put everything in one function, otherwise updating of 
        atomic hamiltonian does not work and I get the wrong values 
        for the forces on the nuclei??! 
        """
        self.initialize_Calc_basics(kwargs_calc, kwargs_mol, coord_nuclei) # ini step 1
        self.intialize_Calc_electronic(pseudo_wf, occupation_numbers) # ini step 2
        self.calculate_effective_potential() # calculate effective potential
        self.effective_potential = self.get_effective_potential()  
        self.kinetic_energy_gradient = self.calculate_kinetic_en_gradient()
        self.dE_drho = self.kinetic_energy_gradient + self.effective_potential
        # calculate forces on nuclei
        W_aL = self.Calc_obj.hamiltonian.calculate_atomic_hamiltonians(self.Calc_obj.density)
        atomic_energies = self.Calc_obj.hamiltonian.update_corrections(self.Calc_obj.density, W_aL)
        self.Calc_obj.wfs.eigensolver.initialize(self.Calc_obj.wfs)
        self.Calc_obj.wfs.eigensolver.subspace_diagonalize(self.Calc_obj.hamiltonian, self.Calc_obj.wfs, self.Calc_obj.wfs.kpt_u[0])
        
        self.atomic_forces = calculate_forces(self.Calc_obj.wfs, self.Calc_obj.density, self.Calc_obj.hamiltonian)

###############################################################################
###                             get dE/dR                                   ###
###############################################################################
    def initialize_eigensolver(self):
        self.Calc_obj.wfs.eigensolver.initialize(self.Calc_obj.wfs)
        self.Calc_obj.wfs.eigensolver.subspace_diagonalize(self.Calc_obj.hamiltonian, self.Calc_obj.wfs, self.Calc_obj.wfs.kpt_u[0])
        
    def calculate_forces_on_nuclei(self):
        self.initialize_eigensolver()
        self.forces = calculate_forces(self.Calc_obj.wfs, self.Calc_obj.density, self.Calc_obj.hamiltonian)
        
###############################################################################
###                             update density                              ###
###############################################################################




    # calculation of the new pseudo density in the CPMD cycle
    def update_density(self, niter):
        
        # scale wave function to get pseudo valence density (sqrt_ps_dens)
        # we propagate the square root of the pseudo density \sqrt{\tilde{n}} without core contribution?, but
        # we get only the pseudo wave function \tilde{\psi}, that is related to \tilde{n} as
        # \tilde{n} = f_n |\tilde{\psi}|^2, where f_n is the number of valence electrons
        # therefore \sqrt{\tilde{n}} = \sqrt{f_n} \tilde{\psi}
        # f_n is an array, but in OFDFT we have only one f_n value therefore access f_n[0]
        sqrt_ps_dens0 = np.sqrt(self.occupation_numbers[0])*self.pseudo_wf
        sqrt_ps_dens1_unconstrained = np.zeros(sqrt_ps_dens0.shape)
        
        if niter > 0: # do verlet
            do = 'some stuff'
        else: # propagation for first step   
            sqrt_ps_dens1_unconstrained = sqrt_ps_dens0 + 0.5*self.dt**2/self.mu*(-self.dE_drho)

        # add constraint                       
        volume_gpt = self.Calc_obj.density.gd.dv
        tau = self.calculate_lambda_constraint(sqrt_ps_dens0, sqrt_ps_dens1_unconstrained, volume_gpt)
        sqrt_ps_dens1 = sqrt_ps_dens1_unconstrained + tau*sqrt_ps_dens0
        
        self.pseudo_wf_previous = self.pseudo_wf.copy()
        self.pseudo_wf = (sqrt_ps_dens1/np.sqrt(self.occupation_numbers[0])).copy() # undo scaling to get pseudo valence density without occupation
    
    # calculation of the langrange multiplier necessary to ensure that number of electrons is conserved
    def calculate_lambda_constraint(self, sqrt_ps_dens, sqrt_ps_dens1_unconstrained, volume_gpt):
        int_phi0_squared = np.sum( np.power(sqrt_ps_dens, 2)*volume_gpt )
        int_phi1_tilde_squared = np.sum( np.power(sqrt_ps_dens1_unconstrained, 2)*volume_gpt )
        int_mixed = np.sum( sqrt_ps_dens1_unconstrained*sqrt_ps_dens*volume_gpt )
        
        p = 2*int_mixed/int_phi0_squared
        q = ( int_phi1_tilde_squared - int_phi0_squared )/int_phi0_squared
        tau_pos = -p/2 + math.sqrt( (-p/2)**2 - q )
        tau_neg = -p/2 - math.sqrt( (-p/2)**2 - q )
        tau = None
        if ( abs(tau_pos) < abs(tau_neg) ):
            tau = tau_pos
        else:
            tau = tau_neg
           
        if tau > 1:
            print('Warning: tau > 1 ')
        return(tau)
            
            
            
            
            
            
            
            

        