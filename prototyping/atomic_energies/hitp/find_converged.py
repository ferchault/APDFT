#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:32:08 2019

@author: misa
"""

import glob
import os
from parse import parse_log_file

def get_tree(path):
    return(glob.glob(path+'/*/*/*/'))
    

def find_pp(dirs, dest):
    ready2cube = []
    broken = []
    for d in dirs:
        status, gemax, num_it = parse_log_file(os.path.join(d,'run.log'))
        if status == 'finished' and gemax < 1e-6:
            ready2cube.append(d)
        elif status == 'broken':
            broken.append(d+'\t'+gemax)
    with open(os.path.join(dest, 'ready2cube'), 'w') as fs:
        for item in ready2cube:
            fs.write("%s\n" % item)
            
    with open(os.path.join(dest, 'broken'), 'w') as fs:
        for item in broken:
            fs.write("%s\n" % item)