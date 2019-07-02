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
                    setups=name, txt='test.txt')
        
            
        molecule.set_calculator(Calc_ref)
        energy_arr[idx] = molecule.get_total_energy()
        kinetic_energy = Calc_ref.hamiltonian.calculate_kinetic_energy(Calc_ref.density)
        potential_energy = np.sum(Calc_ref.hamiltonian.update_pseudo_potential(Calc_ref.density))
        print('Total pseudo energy = ' +str(kinetic_energy+potential_energy))
        print('Total energy = ' + str(energy_arr[0]))
        #Calc_ref.write('result_32_gpts_d_'+str(d)+'.gpw', mode='all')
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
                    setups=name, txt='test.txt')
            
        molecule.set_calculator(Calc_ref)
        energy_arr[idx] = molecule.get_total_energy()
        kinetic_energy = Calc_ref.hamiltonian.calculate_kinetic_energy(Calc_ref.density)
        potential_energy = np.sum(Calc_ref.hamiltonian.update_pseudo_potential(Calc_ref.density))
        print('Total pseudo energy = ' +str(kinetic_energy+potential_energy))
        print('Total energy = ' + str(energy_arr[0]))
        #Calc_ref.write('result_32_gpts_d_'+str(d)+'.gpw', mode='all')
    return(Calc_ref)
        
#def test_CPMD_run():
# create reference calc object
Calc_ref = create_ref_Calc(2.0)
#Calc_ref = create_ref_Calc_64_gpt(2.0)
print('Reference intialized!')
# initialize
kwargs_mol = {'symbols':'H2', 'cell':(12,12,12), 'pbc':True }
coords = Calc_ref.atoms.get_positions()#[(6.0, 6.0, 5.35), (6.0, 6.0, 6.65)]
gpts = (32, 32, 32)
#gpts = (64, 64, 64)
xc = '1.0_LDA_K_TF+1.0_LDA_X+1.0_LDA_C_PW'
maxiter = 500
lambda_coeff = 1.0
eigensolver = CG(tw_coeff=lambda_coeff)
mixer = Mixer()
setups = 'lambda_' + str(lambda_coeff)
txt = 'output_test.txt'
kwargs_Calc = { 'gpts':gpts , 'xc':xc, 'maxiter':maxiter, 'eigensolver':eigensolver, 'mixer':mixer, 'setups':setups, 'txt':txt}

occupation_numbers = Calc_ref.wfs.kpt_u[0].f_n
pseudo_wf = Calc_ref.wfs.kpt_u[0].psit_nG[0]
mu = 1000
dt = 0.05
niter_max = 50

# Create target directory & all intermediate directories if don't exists
path = '/home/misa/APDFT/prototyping/gpaw/CPMD/results/'+'dt_'+str(dt)+'-mu_'+str(mu)
if not os.path.exists(path):
    os.makedirs(path)

CPMD_obj = CPMD(kwargs_Calc, kwargs_mol, occupation_numbers, mu, dt, niter_max, pseudo_wf, coords, path)

# test run
#CPMD_obj.run()

cProfile.run('CPMD_obj.run()', 'stats')
CPMD_obj.save_all()

import pstats
p = pstats.Stats('stats')
p.strip_dirs()
p.sort_stats('cumulative').print_stats(20)

