#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:15:50 2019

@author: giorgiod
"""

import numpy as np


def optimiz2(scf,r,th=1e-4,h=0.1,maxiter=15):
    energies=[]
    i=0
    dr=h
    while True:
        f0=scf(r)
        p1=scf(r-h)
        p2=scf(r+h)
        f1=(p2-p1)/2./h
        f2=(p2+p1-2*f0)/h**2
        energies.append([r,f0,f1,f2])
        if abs(f1)<th or i>maxiter:
            return energies
        dr=-f1/f2
        r=r+dr
        h=max(dr/2,1e-4)
        
        

def optimiz(scfs,r0,th=1e-4,dr0=0.1):
    de=10.
    r=r0
    dr=dr0
    energies=[]
    e0=scfs(r)
    energies.append([r,e0])
    r=r+dr    
    while abs(de)>th or de>0:
        e1=scfs(r)
        energies.append([r,e1])
        if e1<e0:
            r=r+dr
        else:
            dr=-dr*0.33
            r=r+dr
        de=e1-e0
        e0=e1

    print energies    
    return (energies)

