#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:15:44 2019

@author: giorgiod
"""
from horton import *
from co_scf2 import b3lyp,uhf,rhf
import matplotlib.pyplot as plt
import numpy as np

def to_cm(k,Mu):
    return (k/Mu)**0.5*centimeter/planck/lightspeed
def to_k(cm,Mu):
    

Mu_CO=periodic['c'].mass*periodic['O'].mass/(periodic['c'].mass+periodic['O'].mass )
Mu_NN=periodic['N'].mass/2

pes=[]

# scan on the distances------------
#y=2.0
#while y<2.2:
#    pes.append([y,cctz_uhf(y,[6.,8.])[0]])
#    y+=0.02

#-------- NN molecule ----------

y=2.
while y<2.3:
    pes.append([y,b3lyp(y,[6.,8.],'aug-cc-pVQZ')])
    y+=0.1
    
## scan on the nuclear charges------------
#y=0. 
#while y<=2.:
#    pes.append([y,SCFonCO(2.0,[6.+y,8.-y])])
#    y+=0.1

for x in pes:
    plt.scatter(x[0],x[1])

a= np.polyfit([x[0] for x in pes],[x[1] for x in pes],2)

print to_cm(a[0],Mu_CO)

##central finite difference
#CO_energy=SCFonCO(2.0,[6.,8.])
#h=.1
#ad1=(SCFonCO(2.0,[6.+h/2,8.-h/2])-SCFonCO(2.0,[6.-h/2,8.+h/2]))/h
#
#ad2=(SCFonCO(2.0,[6.+h,8.-h])+SCFonCO(2.0,[6.-h,8.+h])-2*CO_energy) /h**2


#forward finite differences
#h=.05
#ad1=(SCFonCO(2.0,[6.+h,8.-h])-CO_energy)/h
#
#ad2=(SCFonCO(2.0,[6.+2*h,8.-2*h])-2*SCFonCO(2.0,[6.+h,8.-h])+CO_energy) /h**2
#
#print SCFonCO(2.0,[7.,7.])
#print CO_energy
#print ad1
#print ad2
#print CO_energy+ad1+1./2*ad2
