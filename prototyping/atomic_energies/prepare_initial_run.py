#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:55:23 2019

@author: misa
"""
import qml
import numpy as np
import os

from atom_section import generate_atom_section, shift2center
from explore_qml_data import get_num_val_elec
from pp_files import write_pp_files_compound

def parse_qm9_slice(filepath):
    with open(filepath, 'r') as fs:
        files = []
        [files.append(line.rstrip('\n')) for line in fs]
    return(files)

def gd_slice_qm9(path_list, lambda_ve):
    for path in path_list:
        gd_comp(path, lambda_ve)
    
def gd_comp(compound_path, lambda_ve):
    """
    generate directories with input files and pp for every lambda value of the compound
    compound_path: path to the xyz-qm9-file of the compound
    lambda_ve: list of lambda values given in valence electrons (the lambda value is lambda_ve/total number of ve for the compound)
    """
    # load compound entry in qml.compound
    compound = qml.Compound(xyz=compound_path)

    total_ve = get_num_val_elec(compound.nuclear_charges)
    
    # make directory from compound
    compound_name = os.path.basename(compound_path).split('.')[0]
    base_directory = os.path.join('/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/', compound_name)
    
    # generate parent directory
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    
    for lambda_val in lambda_ve:
        # generate subdirectory for specific lambda value
        directory_lambda = os.path.join(base_directory, 've_' + str(int(lambda_val)) + '/run0/')
        if not os.path.exists(directory_lambda):
            os.makedirs(directory_lambda)
            
        # generate input file for lambda-value
        input_file = gi_file(lambda_val, compound, total_ve)
        # write input-file to directory
        input_path = os.path.join(directory_lambda, 'run.inp')
        with open(input_path, 'w') as f:
            f.writelines(input_file)
        # generate pp-files and write to directory for lambda-value
        write_pp_files_compound(compound, lambda_val/total_ve, directory_lambda)

def gi_file(lambda_ve, compound, total_ve):
    template = '/home/misa/APDFT/prototyping/atomic_energies/input-template/run0.inp'
    
    # specific input section and charge
    charge = lambda_ve - total_ve
    # shift coordinates
    boxsize = np.array([20.0, 20, 20])
    boxcenter = boxsize/2
    compound.coordinates = shift2center(compound.coordinates, boxcenter)
    atom_section = generate_atom_section(compound)
    
    with open(template, 'r') as fs_template:
        input_file = read_and_replace(fs_template, charge, atom_section)
    return(input_file)

def read_and_replace(fs, charge, atom_section):
    """
    reads input file from template and substitutes value of charge and atom section
    fs: file stream of template of input file
    charge: charge that has to be added to compensate for fractional nuclear charges
    atom section: input of the atom section for the compound
    """
    input_file = []
    for line in fs:
        # add correct charge and skip line with charge in template file
        if 'CHARGE' in line:
            input_file.append(line)
            next_line = fs.readline()
            input_file.append('  \t%1.1f\n' %charge)
            continue
        # add atom section to input file and leave for loop (atom section last part of input file)
        elif 'ATOMS' in line:
            input_file.extend(atom_section)
            break
        input_file.append(line)

    return(input_file)
            
        

#compound = qml.Compound(xyz='/home/misa/datasets/qm9/dsgdb9nsd_005349.xyz')
#boxsize = np.array([20.0, 20, 20])
#
#
#
################################################################################
##                           generate atom section                             #
################################################################################
## shift coordinates
#boxcenter = boxsize/2
#compound.coordinates = shift2center(compound.coordinates, boxcenter)
## generate atom section
#at_sec = generate_atom_section(compound)
#
################################################################################
##                           read input template                               #
################################################################################
#template_initial_run = '/home/misa/APDFT/prototyping/atomic_energies/input-template/run.inp'
#input_file = read_till_keyword(template_initial_run, 'ATOMS')
#
################################################################################
##           merge template and atom section and write new file                #
################################################################################
#input_file.extend(at_sec)
#calc_dir = '/home/misa/APDFT/prototyping/atomic_energies/results/calculations/converge_lambda_test/dsgdb9nsd_005349/box20/run.inp'
#with open(calc_dir, 'w') as f:
#    f.writelines(input_file)