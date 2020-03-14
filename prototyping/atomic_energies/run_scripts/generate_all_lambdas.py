#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:09:54 2019

@author: misa
"""

import numpy as np
import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies')
from prepare_initial_run import gd_comp

# generate only lambda values that did not exist already
baseval = [8,15,23,30,38]
start=2
lval = []
for i in baseval:
    l = np.arange(start, i, 2).astype(int)
    l.tolist()
    lval.extend(l)
    start = i + 1
    if start%2 != 0:
        start += 1
    
path = '/home/misa/datasets/qm9/dsgdb9nsd_003712.xyz'

for l in lval:
    gd_comp(path, [l])