#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:55:36 2019

@author: misa

generate modified pp-files for given lambda

input: elements, lambda, path to pp-files
"""

import os
import numpy as np
import qml

def scale_ZV(zv_line, lamb):
    zv=float(zv_line.strip('ZV ='))
    new_zv = zv*lamb
    new_zv_line = '  ZV = %f\n' %new_zv
    return(new_zv_line)
    
def scale_coeffs(coeffs_line, lamb):
    parts = np.array([float(_) for _ in coeffs_line.split('#')[0].strip().split()])
    parts[1:] *= lamb
    formatstring = '%4d' + (len(parts)-1)*' %15.9f' + '   #C  C1 C2\n'
    return(formatstring % (*parts,))

def generate_pp_file(lamb, element, pp_dir='/home/misa/software/PP_LIBRARY/', pp_type='_SG_LDA'):
    name_pp = element + pp_type
    f_pp = os.path.join(pp_dir, name_pp)
    
    new_pp_file = []
    for line in open(f_pp).readlines():
        if 'ZV' in line:
            new_pp_file.append(scale_ZV(line, lamb))
            continue
        if '#C' in line:
            new_pp_file.append(scale_coeffs(line, lamb))
            continue
        new_pp_file.append(line)
    new_pp_file[len(new_pp_file)-1] = new_pp_file[len(new_pp_file)-1].rstrip('\n')
    return(new_pp_file)
    
def write_pp_files_compound(compound, lamb, calc_dir, pp_dir='/home/misa/software/PP_LIBRARY/', pp_type='_SG_LDA'):  
    for k in compound.natypes.keys():
        pp_file = generate_pp_file(lamb, k)
        path_file = os.path.join(calc_dir, k + pp_type)
        with open(path_file, 'w') as f:
            f.writelines(pp_file)

# load compound date
compound = qml.Compound(xyz='/home/misa/datasets/qm9/dsgdb9nsd_014656.xyz')
# define parameters
calc_dir = '/home/misa/APDFT/prototyping/atomic_energies/results/calculations/fractional_charge/dsgdb9nsd_014656/lambda_0p2'
lamb = 0.2
# write pp-files to directory
write_pp_files_compound(compound, lamb, calc_dir)