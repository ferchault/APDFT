#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:49:38 2019

@author: misa

Goal: apply kinetic energy operator to density/sqrt of density (\hat{T} \psi to obtain kinetic energy gradient with respect to density

This script compares (\hat{T} \psi using my own formulation of \hat{T} with the kinetic energy operator from gpaw.wavefunctions and
a grid object that is used to calculate the kinetic energy during the SCF cycle by looking at slices through the corresponding grids



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


################################################################################
# contour plots along and orthogonal to bond axis in H2

def get_plots(func_obj, title=None):
    x_plane_kin_par_bond = func_obj.select_plane('x', 6.0/Bohr)
    y_plane_kin_par_bond = func_obj.select_plane('y', 6.0/Bohr)
    

    
    fig, ax = plt.subplots(2, 2)
    
    if(title!=None):
        fig = plt.gcf()
        fig.canvas.set_window_title(title)
    
    # plot along bond axis orthogonal to x-axis
    pos = np.array( [[6.0/Bohr,6/Bohr], [5.625/Bohr,6.375/Bohr]] )
    x1, x2 = 4.7/Bohr, 7.3/Bohr
    y1, y2 = 5.12/Bohr, 6.88/Bohr
    ax[0][0].set_title(r'Plane of $ -\frac{1}{2} \nabla^2 \tilde{\rho}^{1/2} $ parallel to bond axis and orthogonal to x-axis')
    ax[0][0].set_xlabel(r'cell axis z')
    ax[0][0].set_ylabel(r'cell axis y')
    ax[0][0].set_xlim(x1, x2)
    ax[0][0].set_ylim(y1, y2)
    ax[0][0].contour(func_obj.coordinates[1], func_obj.coordinates[2], x_plane_kin_par_bond)
    ax[0][0].scatter(pos[1], pos[0], c='red')
    
    # plot along bond axis orthogonal to y-axis
    ax[0][1].set_title(r'Plane of $ -\frac{1}{2} \nabla^2 \rho^{1/2} $ parallel to bond axis and orthogonal to y-axis')
    ax[0][1].set_xlabel(r'cell axis x')
    ax[0][1].set_ylabel(r'cell axis z')
    ax[0][1].set_xlim(x1, x2)
    ax[0][1].set_ylim(y1, y2)
    ax[0][1].contour(func_obj.coordinates[0], func_obj.coordinates[2], y_plane_kin_par_bond)
    ax[0][1].scatter(pos[1], pos[0], c='red')
    
    z_plane_kin_o_bond_h1 = func_obj.select_plane('z', 5.625/Bohr)
    z_plane_kin_o_bond_h2 = func_obj.select_plane('z', 6.375/Bohr)
    
    # plot orthogonal to bond axis through H1
    pos_z = np.array( [[6.0/Bohr, 6.0/Bohr], [6.0/Bohr, 6.0/Bohr]] )
    x1z, x2z = 5.25/Bohr, 6.73/Bohr
    y1z, y2z = 5.25/Bohr, 6.73/Bohr
    ax[1][0].set_title(r'Plane of $ -\frac{1}{2} \nabla^2 \rho^{1/2} $ orthogonal to bond axis at H1')
    ax[1][0].set_xlabel(r'cell axis x')
    ax[1][0].set_ylabel(r'cell axis y')
    ax[1][0].set_xlim(x1z, x2z)
    ax[1][0].set_ylim(y1z, y2z)
    ax[1][0].contour(func_obj.coordinates[0], func_obj.coordinates[1], z_plane_kin_o_bond_h1)
    ax[1][0].scatter(pos_z[1], pos_z[0], c='red')
    
    # plot orthogonal to bond axis through H2
    ax[1][1].set_title(r'Plane of $ -\frac{1}{2} \nabla^2 \rho^{1/2} $ orthogonal to bond axis at H2')
    ax[1][1].set_xlabel(r'cell axis x')
    ax[1][1].set_ylabel(r'cell axis y')
    ax[1][1].set_xlim(x1z, x2z)
    ax[1][1].set_ylim(y1z, y2z)
    ax[1][1].contour(func_obj.coordinates[0], func_obj.coordinates[1], z_plane_kin_o_bond_h2)
    ax[1][1].scatter(pos_z[1], pos_z[0], c='red')
    
    
# load results from GPAW calculation
calc2 = GPAW('/home/misa/APDFT/prototyping/gpaw/OFDFT/result_64_gpts.gpw')

# cell width in Bohr
cell_vectors = calc2.atoms.cell
a_x = cell_vectors[0][0]
a_y = cell_vectors[1][1]
a_z = cell_vectors[2][2]

###############################################################################

# create func object for sqrt of coarse pseudo density and apply own kinetic energy operator to sqrt(density)
sqrt_pseudo_dens = np.sqrt(calc2.density.nt_sG[0]) # sqrt of pseudo density on coarse grid
kwargs_pseuod_dens = {'func_value': sqrt_pseudo_dens, 'length_cell':[a_x, a_y, a_z]}
Pseudo_Wf = pr.Func_3var(**kwargs_pseuod_dens) 
kin_op = Pseudo_Wf.get_kin_en_op()

# create func object for \hat{T} sqrt_ps_density
kwargs_kin_op = kwargs_pseuod_dens = {'func_value': kin_op, 'length_cell':[a_x, a_y, a_z]}
Kinetic = pr.Func_3var(**kwargs_kin_op) 

# plot planes through \hat{T} sqrt_ps_density
get_plots(Kinetic, 'own function for kinetic energy')


###############################################################################

# use gpaw function to apply \hat{T} to sqrt ps dens
kin_op_gpaw = np.zeros(sqrt_pseudo_dens.shape, dtype=float)
calc2.wfs.kin.apply(sqrt_pseudo_dens, kin_op_gpaw, phase_cd=None)

# create func object for gpaw \hat{T} sqrt_ps_density
kwargs_kin_gpaw = {'func_value': kin_op_gpaw, 'length_cell':[a_x, a_y, a_z]}
Kinetic_gpaw = pr.Func_3var(**kwargs_kin_gpaw)

# plot planes through gpaw \hat{T} sqrt_ps_density
get_plots(Kinetic_gpaw, 'gpaw function for kinetic energy operator')

# plot line along bond axis
z_line_along_bond_gpaw = Kinetic_gpaw.select_line(['x', 'y'], [6.0, 6.0])

################################################################################

# use gpaw grid also used for calculation of kinetic energy
vt_G = -calc2.hamiltonian.vt_sG[0]

# create func object for gpaw \hat{T} sqrt_ps_density
kwargs_vt_G = {'func_value': vt_G, 'length_cell':[a_x, a_y, a_z]}
Kinetic_vt_G = pr.Func_3var(**kwargs_vt_G)

# plot planes through gpaw \hat{T} sqrt_ps_density
get_plots(Kinetic_vt_G, 'gpaw grid used for calculation of kinetic energy')

# plot line along bond axis
z_line_along_bond_vtg = Kinetic_vt_G.select_line(['x', 'y'], [6.0, 6.0])

