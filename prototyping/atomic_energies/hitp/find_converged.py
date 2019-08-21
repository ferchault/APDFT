#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:32:08 2019

@author: misa
"""

import glob
import os
from parse import parse_log_file
import pandas as pd

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
        

def get_status(dirs, dest, save_df=False, write=True):
    """
    determine the status of calculation from their log-files, returns a tuple of lists
    every element of the tuple describes one status (broken, converged, restart...)
    each element of the tuple contains all directories with the same status
    
    dirs: list of paths to log-files
    dest: path to destination where the output files are written
    save_df: save pandas Dataframes made from the log-files in the corresponding log-file directories
    write: if True writes the status to files in dest, otherwise the results are returned as a tuple of lists
    """
    
    # get status for every directory in dirs
    ready2cube = []
    restart = []
    broken = []
    other = []
    for d in dirs:
        status, gemax, num_it = parse_log_file(os.path.join(d,'run.log'), save=save_df)
        if status == 'finished' and gemax < 1e-6:
            ready2cube.append(d)
        elif status == 'finished' and gemax > 1e-6:
            restart.append(d)
        elif status == 'broken':
            broken.append(d+'\t'+gemax)
        elif status == "Running, not started or broken":
            other.append(d)
    
    # write results to files or return as lists
    if write:
        with open(os.path.join(dest, 'ready2cube'), 'w') as fs:
            for item in ready2cube:
                fs.write("%s\n" % item)
                
        with open(os.path.join(dest, 'restart'), 'w') as fs:
            for item in restart:
                fs.write("%s\n" % item)
                
        with open(os.path.join(dest, 'broken'), 'w') as fs:
            for item in broken:
                fs.write("%s\n" % item)
        
        with open(os.path.join(dest, 'other'), 'w') as fs:
            for item in other:
                fs.write("%s\n" % item)
    else:
        return(ready2cube, restart, broken, other)
            
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
    
def unified_convergence_lambda(path, lam_ve, save=None):
    """
    return concatenated Dataframes for convergence for one compound for one lambda value over several restarts
    path: path to compound
    lam_ve: lambda value given in valence electrons as a string for which dataframes will be concatenated
    save: path where concatenated fill will be saved
    """
    # sorted paths to all parsed log-files (pandas Dataframe) with the same lambda-value
    direc = path + '/*/' + 've_' + lam_ve + '/parsed_run.log'
    direc = glob.glob(direc)
    direc.sort()
    
    df = pd.read_csv(direc[0], sep='\t')
    # read log-files and append content to list
    for idx, log_f in enumerate(direc):
        if idx!=0:
            tmp = pd.read_csv(log_f, sep='\t')
            tmp['NFI'] = tmp['NFI']+df['NFI'][len(df['NFI'])-1]
            df = pd.concat([df, tmp])
            
    if save:
        df.to_csv(save, sep='\t', header=True, index=False)
    else:
        return(df)
            
            
        
        