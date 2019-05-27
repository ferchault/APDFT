#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:12:38 2019

@author: misa
"""

import math
import numpy as np
from matplotlib import pyplot as plt

class Func_3D:
    
    func_value = None
    shape = None
    axes = {'x':0, 'y':1, 'z':2}
    
#    def __init__(self, func_value):
#        self.func_value = func_value
#        self.shape = np.shape(self.func_value)
        
    def __init__(self, func_value):
        self.func_value = func_value
        self.shape = np.shape(self.func_value)

    
    # get gridpoint along the coordinate q closest to the position of atom on axis q (coord_atom)
    # input: number of gridpoints along axis, length of cell along axis and atom coordinate on axis
    # e.g. x-coordinate and atom position = 3.0 in cell for electronic structure calculation
    # get_gpt() returns the index of the gridpoint along the x-coordinate that is closest to 3.0
    
    def get_gpt_atom(self, axis, coord_atom, length_cell):   
        # calculate intervals between origin of cell and atom position along coordinate q   
        dist_between_gpt = length_cell/(self.shape[self.axes[axis]]-1)
        # index of gpt that is closest to the atom position (rounding and conversion to integer necessary because gpt is in general not exactly at atom position)
        idx_gpt_atom = int( math.floor( coord_atom / dist_between_gpt + 0.5) )
        return(idx_gpt_atom)


        
    # select plane along coordinates q2, q3 closest to a certain atom coordinate along axis q1
     
    def select_plane(self, axis_q1, coord_atom, length_cell):
        idx_closest_coord_atom = self.get_gpt_atom(axis_q1, coord_atom, length_cell)
        plane = self.func_value[idx_closest_coord_atom]
        return(plane)
    
class Plot_func_3D(Func_3D):
    length_cell = None
    coordinates = None
    
    def __init__(self, func_value, length_cell):
        self.func_value = func_value
        self.shape = np.shape(self.func_value)
        self.length_cell = length_cell
        x_coord = np.linspace(0, length_cell, self.shape[0])
        y_coord = np.linspace(0, length_cell, self.shape[1])
        z_coord = np.linspace(0, length_cell, self.shape[2])
        self.coordinates = np.array( (x_coord, y_coord, z_coord) )
    
    def get_gpt_atom(self, axis, coord_atom):
    
        # calculate intervals between origin of cell and atom position along coordinate q   
        dist_between_gpt = self.length_cell/(self.shape[self.axes[axis]]-1)

        # index of gpt that is closest to the atom position (rounding and conversion to integer necessary because gpt is in general not exactly at atom position)
        idx_gpt_atom = int( math.floor( coord_atom / dist_between_gpt + 0.5) )
        
        return(idx_gpt_atom)
        
    def select_plane(self, axis_q1, coord_atom):
        idx_closest_coord_atom = self.get_gpt_atom(axis_q1, coord_atom)
        plane = self.func_value[idx_closest_coord_atom]
        return(plane)
