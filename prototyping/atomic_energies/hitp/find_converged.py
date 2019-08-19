#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:32:08 2019

@author: misa
"""

import glob
import os
from parse import parse_log_file

def get_tree(path, depth):
    """
    returns the paths to the subdirectories with depth
    """
    if depth==3:
        return(glob.glob(path+'/*/*/*/'))
    elif depth==2:
        return(glob.glob(path+'/*/*/'))
    else:
        raise Exception("Depth not implemented!")
        
def get_status(dirs, dest, num=''):
    """
    find log-files for converged calculations and write their directories to file
    
    also write directories of corrupted log-files
    """
    ready2cube = []
    broken = []
    for d in dirs:
        status, gemax, num_it = parse_log_file(os.path.join(d,'run.log'))
        if status == 'finished' and gemax < 1e-6:
            ready2cube.append(d)
        elif status == 'broken':
            broken.append(d+'\t'+gemax)
    return(ready2cube, broken)

def get_status_and_write(dirs, dest, num=''):
    """
    find log-files for converged calculations and write their directories to file
    
    also write directories of corrupted log-files
    """
    ready2cube = []
    broken = []
    for d in dirs:
        status, gemax, num_it = parse_log_file(os.path.join(d,'run.log'))
        if status == 'finished' and gemax < 1e-6:
            ready2cube.append(d)
        elif status == 'broken':
            broken.append(d+'\t'+gemax)
    with open(os.path.join(dest, 'ready2cube'+num), 'w') as fs:
        for item in ready2cube:
            fs.write("%s\n" % item)
            
    with open(os.path.join(dest, 'broken'+num), 'w') as fs:
        for item in broken:
            fs.write("%s\n" % item)
            
def concatenate_files(file_paths):
    """
    concatenates lines of all the files and returns one list with elements as lines of the files
    """
    output = []
    for file in file_paths:
        with open(file, 'r') as f:
            for line in f:
                line=line.rstrip('\n')
                output.append(line)
    return(output)
            
def find_compounds_finished(dirs):
    """
    dirs: absolute path on scicore to every log-file
    count the number of occurences of each compound and return path if ==5 (all calculations are converged)
    """
    # isolate name of compound from paths and store in list
    comp_names = []
    for idx, el in enumerate(dirs):
        splitted = el.split('/')
        comp_name = splitted[9]
        comp_names.append(comp_name)
    
    # count the number of occurences of each compound and return path if ==5 (all calculations are converged)
    # set returns the names of the compunds (every unique name)
    all_finished = [] # compounds for which all calculations are converged
    for compound in set(comp_names):
        occurences = comp_names.count(compound)
#        assert occurences > 5, "More occurences than lambda-values"
        if occurences >= 5:
            compound = './' + compound + '/' # build relative path
            all_finished.append(compound)
        if occurences > 5:
            print(compound, 'number of occurence = ', occurences)

    return(all_finished)
    
