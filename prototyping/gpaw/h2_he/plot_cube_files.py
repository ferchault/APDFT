#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:21:54 2019

@author: misa

extracts density along z-coordinate along the bond for H2 (or other diatomic molecules) from cube file

parameters must be specified in cube_info
"""

import math
import numpy as np
from matplotlib import pyplot as plt

# get gpt along a coordinate q closest to the position of atom specified in cube_info
# e.g. x-coordinate and atom position = 3.0 in cell for electronic structure calculation
# get_gpt() returns the index of the gridpoint along the x-coordinate that is closest to 3.0

def get_gpt_atom(cube_info, coord='x'):
    
    # number of gpt along coord
    num_gpt=cube_info['num_gpt_' + coord]
    
    # calculate intervals between origin of cell and atom position along coordinate q
    
    length_cell=cube_info['length_cell_q']
    
    dist_between_gpt = length_cell/(num_gpt-1)
    
    dist_atom_origin = cube_info['atom_pos_'+coord] # assuming orign at 0
    
    # index of gpt that is closest to the atom position (rounding and conversion to integer necessary because gpt is in general not exactly at atom position)
    
    idx_gpt_atom = int( math.floor( dist_atom_origin / dist_between_gpt + 0.5) )
    
    return(idx_gpt_atom)
    
# make array with position of gpt
def make_gpt_coords(cube_info, coord_in):
    
    num_gpt=cube_info['num_gpt_' + coord_in]
    length_cell=cube_info['length_cell_q']
    
    dist_between_gpt = length_cell/(num_gpt-1)
    
    gpt_list = np.arange(0, length_cell+dist_between_gpt, dist_between_gpt)
    
    return(gpt_list)
    
# returns slice of density in plane specified by coord_in (e.g. coord_in='x' gives y-z-plane)
# the plane is orthogonal to the axis coord_in and goes through the gpt on the axis coord_in that is closest to the coordinate coord_in of the atom in cube_info

# works so far only with y-z plane
    
def get_2d_slice(dens_3d, cube_info, coord_in):
    
    coord = coord_in
    idx_gpt_atom=get_gpt_atom(cube_info, coord=coord)
    
    # define stop index from number of gpt in one slice along y-z coordinate
    size_plane=cube_info['num_gpt_y']*cube_info['num_gpt_z']
    
    idx_start=idx_gpt_atom*size_plane
    idx_end = idx_start + size_plane
    
    gpt_list_y = make_gpt_coords(cube_info, 'y')
    gpt_list_z = make_gpt_coords(cube_info, 'z')
    
    return(gpt_list_y, gpt_list_z, dens_3d[idx_start:idx_end:1])

# returns 1d slice of density from a 2d slice of density starting at gpt specified by idx_gpt_atom
# works so far only with z vs density
    
def get_slice_1d_slice_from2d(dens_slice_2d, cube_info, coord_in):
    
    coord = coord_in
    idx_gpt_atom=get_gpt_atom(cube_info, coord=coord)
    
    
    size_line=cube_info['num_gpt_z']
    idx_start=idx_gpt_atom*size_line
    idx_end = idx_start + size_line
    
    gpt_list_z = make_gpt_coords(cube_info, 'z')
    
    return(gpt_list_z, dens_slice_2d[idx_start:idx_end:1])
    
# gets 1d slice directly from density
    
def get_1d_slice_along_z(dens,cube_info):
    
    y2d,z2d,slice_2d=get_2d_slice(dens, cube_info, 'x')
    z1d,slice_1d=get_slice_1d_slice_from2d(slice_2d, cube_info, 'y')
    return(z1d, slice_1d)

#################################################################
    

# Different fractional charges added to H2
#pathlist=['density_0_0.cube', 'density_0_2.cube', 'density_0_4.cube', 'density_0_6.cube', 'density_0_8.cube', 'density_1_0.cube']
#lamb_list=['$\lambda=0.0$', '$\lambda=0.2$', '$\lambda=0.4$', '$\lambda=0.6$', '$\lambda=0.8$', '$\lambda=1.0$']
    
## Parameter for point charge
#pathlist=['par_pc/dens_rc_-1.cube', 'par_pc/dens_rc_-0_1.cube', 'par_pc/dens_rc_-0_01.cube', 'par_pc/dens_rc_0_01.cube', 'par_pc/dens_rc_0_1.cube', 'par_pc/dens_rc_1.cube']    
#lamb_list=['rc=-1', 'rc=-0.1', 'rc=-0.01', 'rc=0.01', 'rc=0.1', 'rc=1']

# Comparison of normal calculation with calculation using external potential
pathlist=['alch_vs_normal/density_He.cube', 'alch_vs_normal/density_He_expot_H2.cube']
lamb_list=['He', 'H2 + external potential']

i=0


for postpath in pathlist:
    
    path='/home/misa/APDFT/prototyping/gpaw/h2_he/output/'+postpath
    dens = np.loadtxt(path, dtype=float, skiprows=8)
    
        
    cube_info = {'num_gpt_x':112, 'num_gpt_y':112, 'num_gpt_z':112, 'length_cell_q':6.0, 'atom_pos_x':3.0, 'atom_pos_y':3.0, 'atom_pos_z':2.63}
    
    z, dens_z=get_1d_slice_along_z(dens,cube_info)
    
    plt.xlabel('Cell coordinate')
    plt.ylabel('Density along H-H bond')
    
    plt.plot(z, dens_z, label=lamb_list[i])
    plt.legend()
    i=i+1
#y2d,z2d,slice_2d=get_2d_slice(dens, cube_info, 'x')
#z1d,slice_1d=get_slice_1d_slice(slice_2d, cube_info, 'y')

