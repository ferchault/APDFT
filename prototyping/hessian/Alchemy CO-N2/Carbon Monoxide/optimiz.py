#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:15:50 2019

@author: giorgiod
"""

import numpy as np




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

