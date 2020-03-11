#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:39:51 2019

@author: misa
"""

main = '/scicore/home/lilienfeld/sahre0000/projects/atomic_energies/vasp_tests/dsgdb9nsd_001212/box15.0/'

f0 = main + 'pr0.txt'
f1 = main + 'pr1.txt'
f2 = main + 'pr2.txt'

f01 = main + 'pr01.txt'
f02 = main + 'pr02.txt'
f12 = main + 'pr12.txt'

files = [f0, f1, f2, f01, f02, f12]
axis = [0, 1, 2, (0,1), (0,2), (1,2)]

for s in zip(files, axis):
    dens15.save_projection(s[0], s[1])