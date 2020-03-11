#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:12:38 2019

@author: misa
"""

import math
import numpy as np
from ase.units import Bohr, Hartree
from matplotlib import pyplot as plt


    
class Func_3var():
    func_value = None
    shape = None
    axes = {'x':0, 'y':1, 'z':2}
    length_cell = None
    dim = None
    coordinates = None
    spacing = None
    dv = None # volume per gridpoint
    
    bohr = 1.0 # used for conversion from Angstrom to Bohr
    
    # derivatives
    gradient = None
    hessian = None
    laplace = None
    kin_en_op = None
    
    
    def __init__(self, calc_obj=None, **kwargs):
        
        if kwargs.get('bohr', False):
            self.bohr = Bohr
            
        if type(calc_obj) == type(None): # initialization without calc_obj 
            self.func_value = kwargs['func_value'] # values of function
            self.shape = np.shape(self.func_value) # number of gridpoints along each axis
            self.dim = len(self.shape) # number of axes 
            
            self.length_cell = [None] * self.dim # length of cell along each axis
            for i in range(0, self.dim):
                self.length_cell[i] = kwargs['length_cell'][i]/self.bohr
                
            self.spacing = [None] * self.dim # spacing between gridpoints along each axis
            for i in range(0, self.dim):
                self.spacing[i] = self.length_cell[i]/self.shape[i]
                
            self.coordinates = [None] * self.dim # value of coordinate at each gridpoint along the axes
            for i in range(0, self.dim):
                self.coordinates[i] = np.linspace(0, self.length_cell[i]-self.spacing[i], self.shape[i])
             
            self.dv = 1.0 # calculate volume per grid point
            for i in self.spacing:
                self.dv *= i
            
        else:
            self.func_value = calc_obj.density.nt_sG[0]
            self.shape = self.func_value.shape
            self.dim = len(self.shape)
            
            self.length_cell = [None] * self.dim # length of cell along each axis
            for idx, val in enumerate(calc_obj.atoms.cell):
                self.length_cell[idx] = val[idx]/self.bohr # conversion from Angstrom to Bohr
                
            self.spacing = calc_obj.density.gd.get_grid_spacings() # get spacing in Bohr from calc object
            
            self.coordinates = [None] * self.dim # value of coordinate at each gridpoint along the axes
            for i in range(0, self.dim):
                self.coordinates[i] = np.linspace(0, self.length_cell[i]-self.spacing[i], self.shape[i])

    # get gridpoint along the coordinate q closest to the position of atom on axis q (coord_atom)
    # input: number of gridpoints along axis, length of cell along axis and atom coordinate on axis
    # e.g. x-coordinate and atom position = 3.0 in cell for electronic structure calculation
    # get_gpt() returns the index of the gridpoint along the x-coordinate that is closest to 3.0
    def get_gpt_atom(self, axis, coord_atom):
        
        axis = self.axes[axis]
        idx_gpt = np.searchsorted(self.coordinates[axis], coord_atom, side='left')
        
        gpt_val = self.coordinates[axis][idx_gpt]
        
        return(idx_gpt, gpt_val)
        
    # select plane q2, q3, with axis_q1 as input at the coordinate coord_atom along axis_q1    
    def select_plane(self, axis_q1, coord_atom):

        idx_closest_coord_atom, gpt_val = self.get_gpt_atom(axis_q1, coord_atom)
        
        if(axis_q1=='x'):
            plane = self.func_value[idx_closest_coord_atom,:,:]
        elif(axis_q1=='y'):
            plane = self.func_value[:,idx_closest_coord_atom,:]
        elif(axis_q1=='z'):
            plane = self.func_value[:,:,idx_closest_coord_atom]
        else:
            print('Axis does not exist, returning None')
            plane=None
       
        return(plane)
    
    # select line along axis q3, where axes=[q1, q2] at the coordinates coords_atom along q1, q2
    def select_line(self, axes, coords_atom):
        idx_closest_coord_atom_ax0, gpt_val = self.get_gpt_atom(axes[0], coords_atom[0])
        idx_closest_coord_atom_ax1, gpt_val = self.get_gpt_atom(axes[1], coords_atom[1])
        
        if(axes==['x', 'y']):
            line = self.func_value[idx_closest_coord_atom_ax0, idx_closest_coord_atom_ax1, :]
        elif(axes==['x', 'z']):
            line = self.func_value[idx_closest_coord_atom_ax0, :, idx_closest_coord_atom_ax1]
        elif(axes==['y', 'z']):
            line = self.func_value[:, idx_closest_coord_atom_ax0, idx_closest_coord_atom_ax1]
        else:
            print('Axis do not exist, returning None')
            line = None
        return(line)
        
    def calculate_gradient(self):
        assert self.spacing[0]==self.spacing[1] and self.spacing[0]==self.spacing[2], "Different spacings along different axes"
        self.gradient = np.array(np.gradient(self.func_value, self.spacing[0], edge_order=2))
        
    def get_gradient(self):
        self.calculate_gradient()
        return(self.gradient)
            
    def calculate_hessian(self):     
        if type(self.gradient) == type(None):
            self.calculate_gradient()   
        # matrix for second derivatives each element of matrix is a nxnxn grid
        # with the value of the second derivative at each gridpoint
        self.hessian = np.empty((self.dim, self.dim)+self.shape, dtype=float)
        for grad_idx in range(0, self.dim):
            assert self.spacing[0]==self.spacing[1] and self.spacing[0]==self.spacing[2], "Different spacings along different axes"
            row_hessian = np.gradient(self.gradient[grad_idx], self.spacing[0], edge_order=2) # row in hessian matrix
            self.hessian[grad_idx,:] = row_hessian
        
    def get_hessian(self):
        self.calculate_hessian()
        return(self.hessian)
        
    def calculate_laplace(self):
        if type(self.hessian)==type(None):
            self.calculate_hessian()
        # get trace of hessian
        self.laplace = np.trace(self.hessian, axis1=0, axis2=1)
        
    def get_laplace(self):
        if type(self.laplace)==type(None):
            self.calculate_laplace()
        return(self.laplace)
        
    def calculate_applied_kin_en_op(self):
        if type(self.laplace)==type(None):
            self.calculate_laplace()
        self.kin_en_op = -0.5*self.laplace
        
    def get_kin_en_op(self):
        if type(self.kin_en_op)==type(None):
            self.calculate_applied_kin_en_op()
        return(self.kin_en_op)
     
    # multiply value by volume assigned to each gridpoint
    def scale_volume(self):
        self.func_value = self.func_value*self.dv
        
    def integrate_left(self, op_wf):
        function_to_integrate = self.func_value*op_wf
        return( np.sum(function_to_integrate) )
            
class Func_3var_no_pbc(Func_3var):
    
    def __init__(self, calc_obj=None, **kwargs):
        
        if type(calc_obj) == type(None): # initialization without calc_obj
            self.func_value = kwargs['func_value'] # values of function
            self.shape = np.shape(self.func_value) # number of gridpoints along each axis
            self.dim = len(self.shape) # number of axes 
            
            self.length_cell = [None] * self.dim # length of cell along each axis
            for i in range(0, self.dim):
                self.length_cell[i] = kwargs['length_cell'][i]
                
            self.spacing = [None] * self.dim # spacing between gridpoints along each axis
            for i in range(0, self.dim):
                self.spacing[i] = self.length_cell[i]/self.shape[i]  
                
            self.coordinates = [None] * self.dim # value of coordinate at each gridpoint along the axes
            for i in range(0, self.dim):
                self.coordinates[i] = np.linspace(self.spacing[i], self.length_cell[i]-self.spacing[i], self.shape[i])
             
            self.dv = 1.0 # calculate volume per grid point
            for i in self.spacing:
                self.dv *= i
                
        else:
            self.func_value = calc_obj.density.nt_sG[0]
            self.shape = self.func_value.shape
            self.dim = len(self.shape)
            
            self.length_cell = [None] * self.dim # length of cell along each axis
            for idx, val in enumerate(calc_obj.atoms.cell):
                self.length_cell[idx] = val[idx]/self.bohr # conversion from Angstrom to Bohr
                
            self.spacing = calc_obj.density.gd.get_grid_spacings() # get spacing in Bohr from calc object
            
            self.coordinates = [None] * self.dim # value of coordinate at each gridpoint along the axes
            for i in range(0, self.dim):
                self.coordinates[i] = np.linspace(self.spacing[i], self.length_cell[i]-self.spacing[i], self.shape[i])
        
        # returns gpt coordinate and corresponding gpt idx, that is closest to the input coordinate @in[coord_atom] along the specified axis
        #def get_gpt_atom(self, axis, coord_atom):
            
        
class Func_1var(Func_3var):
    def calculate_gradient(self):
        self.gradient = np.array( np.gradient(self.func_value, self.spacing[0], edge_order=2) )
    
    def calculate_hessian(self):     
        if type(self.gradient) == type(None):
            self.calculate_gradient()   
        # matrix for second derivatives each element of matrix is a nxnxn grid
        # with the value of the second derivative at each gridpoint
        self.hessian = np.empty((1, self.shape[0]), dtype=float)
        self.hessian = np.gradient(self.gradient,self.spacing[0], edge_order=2)
        
class Func_2var(Func_3var):
    
    def calculate_gradient(self):
        assert self.spacing[0]==self.spacing[1], "Different spacings along different axes"
        self.gradient = np.array(np.gradient(self.func_value, self.spacing[0], edge_order=2))
    
    def calculate_hessian(self):     
        if type(self.gradient) == type(None):
            self.calculate_gradient()   
        # matrix for second derivatives each element of matrix is a nxnxn grid
        # with the value of the second derivative at each gridpoint
        self.hessian = np.empty((self.dim, self.dim)+self.shape, dtype=float)
        for grad_idx in range(0, self.dim):
            assert self.spacing[0]==self.spacing[1], "Different spacings along different axes"
            row_hessian = np.gradient(self.gradient[grad_idx], self.spacing[0], edge_order=2) # row in hessian matrix
            self.hessian[grad_idx,:] = row_hessian

##test 1d case
#length_interval = 5.0
#number_points = 100
#x=np.linspace(0, length_interval-length_interval/number_points, number_points)
#y=x*x
#kwargs={'func_value':y, 'length_cell':[5.0]}
#parabola = Func_1var(**kwargs)
#x=parabola.coordinates[0]
#f=parabola.func_value
#f_1st=parabola.get_gradient()
#f_2nd=parabola.get_hessian()
#real_2nd= np.zeros(100)
#real_2nd.fill(2)
#plt.plot(x, f, '.', x, f_1st, '.', x, f_2nd, '.', x, 2*x, x, real_2nd)

## test 2d case
#length_interval = 5.0
#number_points = 100
#x_cor=np.linspace(0, length_interval-length_interval/number_points, number_points)
#y_cor=np.linspace(0, length_interval-length_interval/number_points, number_points)
#xx, yy = np.meshgrid(x_cor,y_cor)
#
#def par2d(x_cor,y_cor):
#    return(x_cor**2+y_cor**2)
#
#Z=par2d(xx,yy)
#kwargs={'func_value':Z, 'length_cell':[5.0, 5.0]}
#
#parabola_2d = Func_2var(**kwargs)
#
#x = parabola_2d.coordinates[0]
#f = parabola_2d.func_value
#f_grad = parabola_2d.get_gradient()
#f_hessian = parabola_2d.get_hessian()
#
#
#fig_2d, ax_2d = plt.subplots(3, 2)
#
## function
#ax_2d[0,0].set_xlabel('x axis')
#ax_2d[0,0].set_ylabel('y axis')
#
#ax_2d[0,0].contourf(parabola_2d.coordinates[0], parabola_2d.coordinates[1], parabola_2d.func_value)
#
## first derivative in x
#ax_2d[1,0].set_xlabel('x axis')
#ax_2d[1,0].set_ylabel('y axis')
#ax_2d[1,0].contourf(parabola_2d.coordinates[0], parabola_2d.coordinates[1], f_grad[0])
#
## first derivative in y
#ax_2d[1,1].set_xlabel('x axis')
#ax_2d[1,1].set_ylabel('y axis')
#ax_2d[1,1].contourf(parabola_2d.coordinates[0], parabola_2d.coordinates[1], f_grad[1])
#
## second derivative in x
#ax_2d[2,0].contourf(parabola_2d.coordinates[0], parabola_2d.coordinates[1], f_hessian[0][0])
#
#ax_2d[2,1].contourf(parabola_2d.coordinates[0], parabola_2d.coordinates[1], f_hessian[1][1])






