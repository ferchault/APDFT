#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:15:49 2019

@author: misa
get density on grid from cube files
"""

import numpy as np
import matplotlib.pyplot as plt
from ase.units import Bohr

#from ase.calculators.vasp.vasp import VaspChargeDensity

class CUBE(object):
    
    def __init__(self, fname):
        f = open(fname, 'r')
        for i in range(2): f.readline() # echo comment
        tkns = f.readline().split() # number of atoms included in the file followed by the position of the origin of the volumetric data
        self.natoms = int(tkns[0])
        self.origin = np.array([float(tkns[1]),float(tkns[2]),float(tkns[3])])
        tkns = f.readline().split() #
        self.NX = int(tkns[0])
        self.X = np.array([float(tkns[1]),float(tkns[2]),float(tkns[3])])
        tkns = f.readline().split() #
        self.NY = int(tkns[0])
        self.Y = np.array([float(tkns[1]),float(tkns[2]),float(tkns[3])])
        tkns = f.readline().split() #
        self.NZ = int(tkns[0])
        self.Z = np.array([float(tkns[1]),float(tkns[2]),float(tkns[3])])
        
        self.dv = np.linalg.det(np.array([self.X, self.Y, self.Z])) # volume per gridpoint
        
        self.atoms = []
        for i in range(self.natoms):
            tkns = f.readline().split()
            self.atoms.append([float(tkns[0]), float(tkns[2]), float(tkns[3]), float(tkns[4])])
        
        self.atoms = np.array(self.atoms)
        
        self.data = np.zeros((self.NX,self.NY,self.NZ))
        i=0
        for s in f:
            for v in s.split():
                self.data[i//(self.NY*self.NZ), (i//self.NZ)%self.NY, i%self.NZ] = float(v)
                i+=1
        if i != self.NX*self.NY*self.NZ: raise NameError("FSCK!")
        
        self.scale() # values scaled by volume
        
    def project(self, axes):
        """
        scales density by gridvolume and projects density on specified axes (1D or 2D)
        """
        projected_density = np.sum(self.data*self.dv, axis=axes)
        return(projected_density)
        
    def get_axis(self, axis, unit='Ang'):
        """
        returns cell coordinates in Angstrom or Bohr
        """
        if axis == 'x' or axis == 0:
            l_x = self.X[0]*self.NX
            coords = np.linspace(self.origin[0], l_x-self.X[0], self.NX)
            
        if axis == 'y' or axis == 1:
            l_y = self.Y[1]*self.NY
            coords = np.linspace(self.origin[1], l_y-self.Y[1], self.NY)
            
        if axis == 'z' or axis == 2:
            l_z = self.Z[2]*self.NZ
            coords = np.linspace(self.origin[2], l_z-self.Z[2], self.NZ)
            
        if unit=='Ang':
            coords *= Bohr
        elif unit=='Bohr':
            pass
        else:
            raise Exception('Unknown unit')
            
        return(coords)
        
        
    def get_grid(self):
        """
        returns the coordinates of the grid points where the density values are given as a meshgrid
        works so far only for orthogonal coordinate axes
        """
        # length along the axes
        l_x = self.X[0]*self.NX
        l_y = self.Y[1]*self.NY
        l_z = self.Z[2]*self.NZ
        # gpts along every axis
        x_coords = np.linspace(self.origin[0], l_x-self.X[0], self.NX)
        y_coords = np.linspace(self.origin[1], l_y-self.Y[1], self.NY)
        z_coords = np.linspace(self.origin[2], l_z-self.Z[2], self.NZ)
        # create gridpoints
        meshgrid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        #if unit == 'Ang':
        #for i, g in enumerate(meshgrid):
         #   tmp = g*Bohr
          #  meshgrid[i] = tmp
        return(meshgrid)
    
    def get_hmatrix(self):
        h_x = self.X*self.NX
        h_y = self.Y*self.NY
        h_z = self.Z*self.NZ
        return(np.array([h_x, h_y, h_z]))
        
        
    def scale(self):
        """
        calculate density scaled by volume of gridpoints
        """
        self.data_scaled = (self.data*self.dv).copy()
        
    def plot_projection(self, axes):
        """
        plot scaled projection of density along specified projection axis (1D, 2D)
        """
        projected_density = self.project(axes)
        
        if type(axes) == tuple:
            coordinate = np.linspace(self.origin[0], self.X[0]*self.NX*Bohr, self.NX)
            fig, ax = plt.subplots(1,1)
            ax.plot(coordinate, projected_density)
            ax.set_xlabel(r'Cell coordinate $x_0$ (Ang)')
            ax.set_ylabel(r'$\rho (x_0)$')
            
        if type(axes) == int:
            coordinate0 = np.linspace(self.origin[0]+0.5, self.X[0]*self.NX*Bohr-0.5, self.NX)
            coordinate1 = np.linspace(self.origin[0], self.X[0]*self.NX*Bohr, self.NX)
            fig, ax = plt.subplots(1,1)
            ax.contour(coordinate0, coordinate1, projected_density)
            
    
    
# class Vasp_CHG(object):
#     def __init__(self, fname):
#         # get information from CHG file
#         dens_obj = VaspChargeDensity(fname)
#         self.charge_density = np.array(dens_obj.chg[0]) # read CHG file into numpy array
#         self.atoms = dens_obj.atoms[0] # information about cell, atom positions ...
        
#         del(dens_obj) # delete to free up memory
        
#         # scale electron density
#         self.gpts = self.charge_density.shape
#         self.dv = self.atoms.get_volume()/(self.gpts[0]*self.gpts[1]*self.gpts[2]) 
#         self.charge_density *= self.dv
        
#     def project(self, axes):
#         """
#         scales density by gridvolume and projects density on specified axes (1D or 2D)
#         """
#         projected_density = np.sum(self.charge_density, axis=axes)
#         return(projected_density)
    
#     def save_projection(self, fname, pr_axes):
#         """
#         writes projection of density to file
        
#         fname: path to the file where projected density is stored
#         pr_axes: axes over which is integrated to generate projection
#         """
#         np.savetxt(fname, self.project(pr_axes))
        
        
#     def get_grid(self):
#         """
#         returns the coordinates of the grid points where the density values are given as a meshgrid
#         works so far only for orthogonal coordinate axes
#         """
#         # length along the axes
#         l_x = self.atoms.get_cell()[0][0]
#         l_y = self.atoms.get_cell()[1][1]
#         l_z = self.atoms.get_cell()[2][2]
#         # gpts along every axis
#         x_coords = np.linspace(0, l_x-l_x/self.gpts[0], self.gpts[0])
#         y_coords = np.linspace(0, l_y-l_y/self.gpts[1], self.gpts[1])
#         z_coords = np.linspace(0, l_z-l_z/self.gpts[2], self.gpts[2])
#         # create gridpoints
#         meshgrid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
#         return(meshgrid)
        
        
