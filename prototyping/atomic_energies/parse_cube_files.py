#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:15:49 2019

@author: misa
get density on grid from cube files
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial as scs
import scipy.integrate
from scipy.interpolate import CubicSpline
import scipy.spatial.transform as sst
import functools
import copy
from ase.units import Bohr

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
          self.atoms.append([tkns[0], tkns[2], tkns[3], tkns[4]])
        self.data = np.zeros((self.NX,self.NY,self.NZ))
        i=0
        for s in f:
          for v in s.split():
            self.data[i//(self.NY*self.NZ), (i//self.NZ)%self.NY, i%self.NZ] = float(v)
            i+=1
        if i != self.NX*self.NY*self.NZ: raise NameError("FSCK!")
        
    def project(self, axes):
        """
        scales density by gridvolume and projects density on specified axes (1D or 2D)
        """
        projected_density = np.sum(self.data*self.dv, axis=axes)
        return(projected_density)
        
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
            
    
    
    
