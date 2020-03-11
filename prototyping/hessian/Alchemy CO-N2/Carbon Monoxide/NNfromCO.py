#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:22:00 2019

@author: giorgiod
"""
from horton import *
from co_scf_ToNN import b3lyp,uhf,rhf
import matplotlib.pyplot as plt
import numpy as np

def toNN(distance,scf=b3lyp,basis_set='cc-pvtz'):
    numbers=np.array([6,8])
    coordinates=np.array([[-distance/2.,0.,0.],[distance/2.,0.,0.]])
    pseudo_numbers=np.array([6.,8.])
    a=scf(distance,[6.,8.],basis_set)
    CO_energy=a[0]
    rho=a[1]
    grid=BeckeMolGrid(coordinates,numbers,numbers,random_rotate=False)
      
    #V1 e V2 sono funzioni della griglia 
    V1=((grid.points[:, 0]-coordinates[0][0])**2+(grid.points[:, 1]-coordinates[0][1])**2+(grid.points[:, 2]-coordinates[0][2])**2)**-0.5
    V2=((grid.points[:, 0]-coordinates[1][0])**2+(grid.points[:, 1]-coordinates[1][1])**2+(grid.points[:, 2]-coordinates[1][2])**2)**-0.5
    dV=V2-V1
    
    h=.05
    dRho=(scf(distance,[6.+h/2,8.-h/2],basis_set)[1]-scf(distance,[6.-h/2,8.+h/2],basis_set)[1])/h
    
    d1=grid.integrate(rho,dV)
    d2=grid.integrate(dRho,dV)
    
    DRNN=(7*7-6*8)/distance 
    e_ext=CO_energy+d1+d2/2.+DRNN
#    e_N2=scf(distance,[7.,7.],basis_set)[0]
#    print e_N2    
    print CO_energy
    print e_ext
    return e_ext
