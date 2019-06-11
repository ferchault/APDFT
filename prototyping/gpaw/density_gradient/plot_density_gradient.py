#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:48:52 2019

@author: misa

calculate and plot gradient dE/drho to check if values are reasonable

"""

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/gpaw/tools')
from gpaw import setup_paths
setup_paths.insert(0, '/home/misa/APDFT/prototyping/gpaw/OFDFT')

import numpy as np
from gpaw import GPAW
import processing as pr
from matplotlib import pyplot as plt
from ase.units import Bohr, Hartree

calc_obj = GPAW(r'/home/misa/APDFT/prototyping/gpaw/OFDFT/result_64_gpts.gpw') # load data from SCF calculation
cell_param = calc_obj.atoms.cell
cell_dim = [ cell_param[0][0], cell_param[1][1], cell_param[2][2] ]

sqrt_pseudo_dens = np.sqrt(calc_obj.get_pseudo_density()) # take sqrt of pseudo density to get "orbital" \sqrt{\tilde{\rho}}
sqrt_pseudo_dens = np.sqrt(calc_obj.density.nt_sG[0]) # take sqrt of pseudo density to get "orbital" \sqrt{\tilde{\rho}}

### kinetic energy
kin_operator = np.zeros(sqrt_pseudo_dens.shape, dtype=float) # create grid for kinetic energy operator applied to orbital
calc_obj.wfs.kin.apply(sqrt_pseudo_dens, kin_operator, phase_cd=None) # calculate \hat{T} \tilde{\rho}

### effective potential
eff_potential = calc_obj.get_effective_potential() # effective potential of pseudo density (including kinetic energy functional thomas-fermi...)

### density gradient \partial E / \partial \rho  ( see J. Chem. Phys. 141, 234102 (2014); eq.(3) )
dens_grad = kin_operator/sqrt_pseudo_dens + eff_potential

# plot kin_operator, kin_operator/sqrt_pseudo_dens, eff_potential and kin_operator/sqrt_pseudo_dens + eff_potential
# to check if results are reasonable

def create_func_obj(func_value, cell_dim): # wrapper to create func objects
    kwargs_func = {'func_value':func_value, 'length_cell' : cell_dim, 'bohr':False} # \hat{T} \sqrt{\tilde{\rho}}
    Func_obj = pr.Func_3var(**kwargs_func)
    return(Func_obj)
    
def get_slice(Func_obj): # returns x,y coordinates and func value of plane through bond axis of H-H
    plane = Func_obj.select_plane('x', cell_dim[0]/2)
    return(Func_obj.coordinates[0], Func_obj.coordinates[1], plane)

### comparison of kinetic part before and after scaling with 1/\sqrt{\tilde{\rho}}
#func_value_list = [kin_operator, kin_operator/sqrt_pseudo_dens]
#titles = [r'$ -\frac{1}{2}\nabla^2 \sqrt{\tilde{\rho}} $', r'$ -\frac{1}{2 \sqrt{\tilde{\rho}} } \nabla^2 \sqrt{\tilde{\rho}} $ ']
#xlims = [ [4, 8], [4 ,8] ]
#ylims = [ [4.5, 7.5], [4.5, 7.5] ]
#xlabel = r'$a_x$'
#ylabel = r'$a_y$'
#pos = np.array( [ [6.0, 6.0], [5.625, 6.375] ] ) # position of H nuclei

### plot of kinetic part, effective potential and sum of both
func_value_list = [kin_operator/sqrt_pseudo_dens, eff_potential, dens_grad]
titles = [r'$ -\frac{1}{2 \sqrt{\tilde{\rho}} } \nabla^2 \sqrt{\tilde{\rho}} $ ', r'$\upsilon_{\text{eff}}$', r'$-\frac{1}{2 \sqrt{\tilde{\rho}} } \nabla^2 \sqrt{\tilde{\rho}}+\upsilon_{\text{eff}}$']
xlims = [ [4, 8], [4 ,8], [4 ,8] ]
ylims = [ [4.5, 7.5], [4.5, 7.5], [4.5, 7.5] ]
xlabel = r'$a_x$'
ylabel = r'$a_y$'
pos = np.array( [ [6.0, 6.0], [5.625, 6.375] ] ) # position of H nuclei

Obj_list = []

for func_value in func_value_list: # create func objects
    Obj_list.append(create_func_obj(func_value, cell_dim))

plane_list = []
for Func_Obj in Obj_list: # create plane lying in H-H bond
    sl = get_slice(Func_Obj)
    plane_list.append(sl)

fig, ax = plt.subplots(1, len(plane_list)) # plot planes
for idx, plane in enumerate(plane_list):
    #ax[idx].set_title(titles[idx])
    ax[idx].set_xlim( xlims[idx][0], xlims[idx][1] )
    ax[idx].set_ylim( ylims[idx][0], ylims[idx][1] )
    ax[idx].set_xlabel(xlabel)
    ax[idx].set_ylabel(ylabel)
    ax[idx].contour(plane[0], plane[1], plane[2])
    ax[idx].scatter(pos[1], pos[0], c='black') # position of H nuclei














