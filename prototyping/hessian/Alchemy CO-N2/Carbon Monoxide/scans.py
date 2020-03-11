#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:15:44 2019

@author: giorgiod
"""

from co_fun import SCFonCO
import matplotlib.pyplot as plt

pes=[]

## scan on the distances

#y=1.
#while y<2.7:
#    pes.append([y,SCFonCO(y,[6.,8.])])
#    y+=0.1
y=0. 
while y<=2.:
    pes.append([y,SCFonCO(2.0,[6.+y,8.-y])])
    y+=0.1

for x in pes:
    plt.scatter(x[0],x[1])



