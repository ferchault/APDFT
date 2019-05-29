#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:28:43 2019

@author: giorgiod
"""
#Minimal energy coordinates for UHF/STO3G calculation on CHHHH (methane)-units in bohr radii
from horton import *
import numpy as np
from ch4_uhf_opt import uhf,gradient_from_scf
from numpy.linalg import eig,norm
log.set_level(1)
methane=IOData.from_file('Methane.xyz')
 #optimized geometry for methane at UHF/sto-3g level
#min_coordinates=\
#[[-2.86648325e-01 , 5.85359870e+00, -2.23561386e-02],\
# [ 1.22991358e+01  ,5.86642285e+00,  2.87397448e-03],\
# [-4.43819658e+00 ,-4.16726882e+00, -6.31085928e+00],\
# [-4.47547710e+00 , 1.63923519e+01, -5.57717096e+00],\
# [-4.48670245e+00 , 5.55407058e+00,  1.18572150e+01],]
#
#methane.coordinates=np.asarray(min_coordinates)*angstrom


## forward difference for getting derivatives
#h=.001
#a=[]
#a.append(gradient_from_scf(methane))
#for x in methane.coordinates:
#    x[0]+=h
#    a.append(gradient_from_scf(methane))
#    x[0]-=h
#    x[1]+=h
#    a.append(gradient_from_scf(methane))
#    x[1]-=h
#    x[2]+=h
#    a.append(gradient_from_scf(methane))
#    x[2]-=h
hessian=np.ndarray((3*methane.natom,3*methane.natom))
#for i in range(3*methane.natom-2):
#    hessian[i]=(a[i+1]-a[0])/h
#    hessian[i+1]=(a[i+2]-a[0])/h
#    hessian[i+2]=(a[i+3]-a[0])/h
 
 
 
h=.01
en0=uhf(methane,compute_grad=False)

def comp_hess_el(i,j,mol):
    p=[]
    if i==j:
        mol.coordinates[i//3][i-i//3*3]+=h
        p.append(uhf(mol,compute_grad=False))
        mol.coordinates[i//3][i-i//3*3]-=2*h
        p.append(uhf(mol,compute_grad=False))
        mol.coordinates[i//3][i-i//3*3]+=h
        return (p[1]+p[0]-2*en0)/h**2
    if i!=j:
        mol.coordinates[i//3][i-i//3*3]+=h
        mol.coordinates[j//3][j-j//3*3]+=h  #1,1
        p.append(uhf(mol,compute_grad=False))
        mol.coordinates[i//3][i-i//3*3]-=2*h # -1,1
        p.append(uhf(mol,compute_grad=False))
        mol.coordinates[j//3][j-j//3*3]-=2*h  #-1,-1
        p.append(uhf(mol,compute_grad=False))
        mol.coordinates[i//3][i-i//3*3]+=2*h  # 1,-1
        p.append(uhf(mol,compute_grad=False))
        mol.coordinates[i//3][i-i//3*3]-=h
        mol.coordinates[j//3][j-j//3*3]+=h
        return (p[0]+p[2]-p[1]-p[3])/(4*h**2)


for i in range(hessian.shape[0]):
    for j in range(i+1):
        hessian[i][j]=comp_hess_el(i,j,methane)
        if i!=j:
            hessian[j][i]=hessian[i][j]


##  Building up the Hessian Matrix from the derivativesw of the gradient 


#print '************  hessian  *******************  '
#print hessian

##reduced mass Hessian    
red_hessian=np.ndarray((3*methane.natom,3*methane.natom))
for i in range(methane.natom*3):
    m_i=periodic[methane.numbers[i//3]].mass**0.5
    for j in range(methane.natom*3):
        m_j=periodic[methane.numbers[j//3]].mass**0.5   
        red_hessian[i][j]=hessian[i][j]/m_i/m_j
        
  

def to_cm(ev):
    ''' from an eigenvalue of the mass reduced hessian to a wave number (cm-1) '''
    return norm(ev)**0.5 *centimeter/planck/lightspeed

def normalModes(rH=red_hessian):
    rHs=(rH+rH.transpose())/2.
    eigval,eigevec = eig(rHs)
    freq=[]
    for i in eigval:
        freq.append(to_cm(i))
    return freq
        
print normalModes()

for i in eig(red_hessian)[0]:
    print to_cm(i)
