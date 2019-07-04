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
from matplotlib import pyplot as plt
import os

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
    
    def __init__(self, kwargs_calc=None, kwargs_mol=None, occupation_numbers=None, mu=None, dt=None, niter_max=None, pseudo_wf=None, coords_nuclei=None, main_path = None):
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
        self.coords_new = None
        
        # storage path
        self.main_path = main_path

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
            
            
    # we need this for the calculation of the total energy
    def update_occupations(self):
        magmom_a=self.Calc_obj.atoms.get_initial_magnetic_moments()        
        magmom_av = np.zeros((len(self.Calc_obj.atoms), 3))
        magmom_av[:, 2] = magmom_a
        self.Calc_obj.create_occupations(magmom_av[:, 2].sum(), True)
        
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
        potential_energies = self.Calc_obj.hamiltonian.update_pseudo_potential(self.Calc_obj.density)
        # restrict to coarse grid
        self.Calc_obj.hamiltonian.restrict_and_collect(self.Calc_obj.hamiltonian.vt_sg, self.Calc_obj.hamiltonian.vt_sG)
        return(potential_energies)
        
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
        
        pseudo_potential_energies = self.calculate_effective_potential() # calculate effective potential, and collect potential energies of pseudo-density before update
        pseudo_kinetic_energy = self.Calc_obj.hamiltonian.calculate_kinetic_energy(self.Calc_obj.density)
        self.effective_potential = self.get_effective_potential()  
        self.kinetic_energy_gradient = self.calculate_kinetic_en_gradient()
        self.dE_drho = self.kinetic_energy_gradient + self.effective_potential
        # calculate forces on nuclei
        W_aL = self.Calc_obj.hamiltonian.calculate_atomic_hamiltonians(self.Calc_obj.density)
        atomic_energies = self.Calc_obj.hamiltonian.update_corrections(self.Calc_obj.density, W_aL)    
        self.Calc_obj.wfs.eigensolver.initialize(self.Calc_obj.wfs)
        self.Calc_obj.wfs.eigensolver.subspace_diagonalize(self.Calc_obj.hamiltonian, self.Calc_obj.wfs, self.Calc_obj.wfs.kpt_u[0])
        self.atomic_forces = calculate_forces(self.Calc_obj.wfs, self.Calc_obj.density, self.Calc_obj.hamiltonian)
        
        # add pseudo energies and PAW correction energies together
        pseudo_energies = np.zeros(len(pseudo_potential_energies)+1)
        pseudo_energies[0] = pseudo_kinetic_energy
        pseudo_energies[1:] = pseudo_potential_energies # coulomb, zero, external, xc
        energies = pseudo_energies+atomic_energies
       
        self.Calc_obj.hamiltonian.world.sum(energies)
        (self.Calc_obj.hamiltonian.e_kinetic0, self.Calc_obj.hamiltonian.e_coulomb, self.Calc_obj.hamiltonian.e_zero, self.Calc_obj.hamiltonian.e_external, self.Calc_obj.hamiltonian.e_xc) = energies
        return(pseudo_energies, atomic_energies)
        
        
    # returns total energy; energy from SCF + eigenvalues
    def get_total_energy_static(self):
        # necessary to get eigenvalues of bands
        self.update_occupations()
        total_free_energy = self.Calc_obj.hamiltonian.get_energy(self.Calc_obj.occupations)
        return(total_free_energy)
        
    def kinetic_energy_dynamics(self):
        # kinetic energy of nuclei
        v_nuclei = np.linalg.norm(self.coords_new - self.coords_previous)/(2.0*self.dt) # velocities of nuclei
        en_nuc = 0.5*np.sum(self.Calc_obj.atoms.get_masses()*np.power(np.sum(v_nuclei), 2))
        
        # kinetic energy of electron density
        v_dens = np.sqrt(self.occupation_numbers[0])*(self.pseudo_wf_new - self.pseudo_wf_previous)/(2.0*self.dt) # velocity of electron density
        en_dens = self.Calc_obj.density.gd.dv*(self.mu/np.linalg.det(self.Calc_obj.atoms.cell) )*np.sum(np.power(v_dens, 2)) # sum of kinetic energy at every gpt
        kin_dynamics = en_nuc + en_dens
        return(kin_dynamics, en_nuc, en_dens)
        
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
###                      update density, nuclei                             ###
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
            sqrt_ps_dens1_unconstrained = self.verlet_algorithm_density(sqrt_ps_dens0, sqrt_ps_dens1_unconstrained)
        else: # propagation for first step   
            sqrt_ps_dens1_unconstrained = sqrt_ps_dens0 + 0.5*self.dt**2/self.mu*(-self.dE_drho)

        # add constraint                       
        volume_gpt = self.Calc_obj.density.gd.dv
        tau = self.calculate_lambda_constraint(sqrt_ps_dens0, sqrt_ps_dens1_unconstrained, volume_gpt)
        sqrt_ps_dens1 = sqrt_ps_dens1_unconstrained + tau*sqrt_ps_dens0
        # save tau, change; change tells how big the change in wavefunction is during CPMD
        self.tau[niter] = tau
        self.change[niter] = np.sum(sqrt_ps_dens1_unconstrained / (tau*sqrt_ps_dens0))
        
        self.pseudo_wf_previous = self.pseudo_wf.copy()
        self.pseudo_wf = (sqrt_ps_dens1/np.sqrt(self.occupation_numbers[0])).copy() # undo scaling to get pseudo valence density without occupation
        
    # calulation of new nuclei positions
    def update_nuclei(self, niter):
        self.coords_new = np.zeros(self.coords.shape)
        if niter > 0:
            for idx in range( 0, len(self.coords) ):
                self.coords_new[idx] = self.verlet_algorithm_nucleus(idx)    
        else:
            for idx in range( 0, len(self.coords) ):
                # get force on and mass of nucleus
                atomic_force_nucleus = self.atomic_forces[idx]
                mass_nucleus = self.Calc_obj.atoms.get_masses()[idx]
                # calculate shift for nucleus
                prefactor = 0.5*self.dt**2/mass_nucleus           
                shift_nucleus = prefactor*atomic_force_nucleus
                # update coordinate
                self.coords_new[idx] = self.coords[idx] + shift_nucleus
        
        self.coords_previous = self.coords.copy()
        self.coords = self.coords_new.copy()
                        
    def verlet_algorithm_density(self, sqrt_ps_dens0, sqrt_ps_dens1_unconstrained):
        sqrt_ps_dens_previous = np.sqrt(self.occupation_numbers[0])*self.pseudo_wf_previous
        prefactor = self.dt**2/self.mu
        
        sqrt_ps_dens1_unconstrained = 2*sqrt_ps_dens0 - sqrt_ps_dens_previous + prefactor*self.dE_drho
        return(sqrt_ps_dens1_unconstrained)
    # applies verlet algorithm to single nucleus at position self.coords[idx]
    def verlet_algorithm_nucleus(self, idx):
        # get force on and mass of nucleus
        atomic_force_nucleus = self.atomic_forces[idx]
        mass_nucleus = self.Calc_obj.atoms.get_masses()[idx]
        coord_nucleus = self.coords[idx]
        coord_nucleus_previous = self.coords_previous[idx]
        
        coord_new = 2*coord_nucleus - coord_nucleus_previous + (self.dt**2/mass_nucleus)*atomic_force_nucleus
        return(coord_new)
                    
    # calculation of the langrange multiplier necessary to ensure that number of electrons is conserved
    def calculate_lambda_constraint(self, sqrt_ps_dens, sqrt_ps_dens1_unconstrained, volume_gpt):
        int_phi0_squared = np.sum( np.power(sqrt_ps_dens, 2)*volume_gpt )
        int_phi1_tilde_squared = np.sum( np.power(sqrt_ps_dens1_unconstrained, 2)*volume_gpt )
        int_mixed = np.sum( sqrt_ps_dens1_unconstrained*sqrt_ps_dens*volume_gpt )
        
        p = 2*int_mixed/int_phi0_squared
        q = ( int_phi1_tilde_squared - int_phi0_squared )/int_phi0_squared
#        if (-p/2)**2 - q < 0:
#            print('Discriminant below zero!')
        try:
            tau_pos = -p/2 + math.sqrt( (-p/2)**2 - q )
            tau_neg = -p/2 - math.sqrt( (-p/2)**2 - q )
        except ValueError:
            self.save_all()
            raise Exception('Negative Discriminant in the calculation of tau!')
            
        tau = None
        if ( abs(tau_pos) < abs(tau_neg) ):
            tau = tau_pos
        else:
            tau = tau_neg
        return(tau_pos)
            
    def run(self):
        shape_dens_store = tuple( [self.niter_max+1] ) + self.pseudo_wf.shape
        shape_coord_nuclei_store = tuple( [self.niter_max+1] ) + self.coords.shape
        self.store_dens = np.zeros(shape_dens_store)
        self.store_nuclei = np.zeros( shape_coord_nuclei_store )
        
        self.store_dens[0] = self.pseudo_wf
        self.store_nuclei[0] = self.coords
        
        # intialize storage for energies
        self.pseudo_energies = np.zeros((self.niter_max+1, 5)) # e_coloumb, e_zero, e_external, e_xc
        self.atomic_energies = np.zeros((self.niter_max+1, 5))
        self.total_energies_static = np.zeros(self.niter_max+1)

        #storage for tau, corrections
        self.tau = np.zeros(self.niter_max+1)
        self.change = np.zeros(self.niter_max+1)
        
        for niter in range(0, self.niter_max):
            print('Start iteration: '+ str(niter))
            # forces and energy
            pseudo_energies, atomic_energies = self.calculate_forces_el_nuc(self.kwargs_calc, self.kwargs_mol, self.coords, self.pseudo_wf, self.occupation_numbers)
            self.pseudo_energies[niter] = pseudo_energies.copy()
            self.atomic_energies[niter] = atomic_energies.copy()
            self.total_energies_static[niter] = self.get_total_energy_static()
            
            # position of density and nuclei
            self.update_density(niter)
            self.update_nuclei(niter)
            self.store_dens[niter+1] = self.pseudo_wf
            self.store_nuclei[niter+1] = self.coords
            
        # energies for last CPMD step
        pseudo_energies, atomic_energies = self.calculate_forces_el_nuc(self.kwargs_calc, self.kwargs_mol, self.coords, self.pseudo_wf, self.occupation_numbers)
        self.pseudo_energies[self.niter_max] = pseudo_energies.copy()
        self.atomic_energies[self.niter_max] = atomic_energies.copy()
        self.total_energies_static[self.niter_max] = self.get_total_energy_static()
        
    def save_all(self):
        # save distance along z-axis
        path_dist = os.path.join(self.main_path, 'nuclei_dist.npy')
        movement_H1 = self.store_nuclei[:, 0, 2]
        movement_H2 = self.store_nuclei[:, 1, 2]
        dist=abs(movement_H1-movement_H2)
        dist_plot=dist[np.where(dist>0)]
        np.save(path_dist, dist_plot)
        # save position of nuclei
        path_nuclei = os.path.join(self.main_path, 'nuclei_pos.npy')
        np.save(path_nuclei, self.store_nuclei[0:len(dist_plot)])
        # save time
        path_time = os.path.join(self.main_path, 'time.npy')
        time=np.linspace(0, self.dt*(len(dist_plot)-1), len(dist_plot))
        np.save(path_time, time)
        # save density
        path_density = os.path.join(self.main_path, 'density.npy')
        np.save(path_density, self.store_dens[0:len(dist_plot)])
        # save pseudo energy
        path_total_en_static = os.path.join(self.main_path, 'total_energies_static.npy')
        path_pseudo_en = os.path.join(self.main_path, 'pseudo_energies.npy')
        path_atomic_en = os.path.join(self.main_path, 'atomic_energies.npy')
        np.save(path_total_en_static, self.total_energies_static[0:len(dist_plot)])
        np.save(path_pseudo_en, self.pseudo_energies[0:len(dist_plot)])       
        np.save(path_atomic_en, self.atomic_energies[0:len(dist_plot)])
        # save tau and change of wf
        path_tau = os.path.join(self.main_path, 'tau.npy')
        path_change = os.path.join(self.main_path, 'change.npy')
        np.save(path_tau, self.tau)
        np.save(path_change, self.change)