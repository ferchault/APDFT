#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:36:56 2019

@author: misa
"""

import qml
import numpy as np
from matplotlib import pyplot as plt
import os
from shutil import copyfile


###############################################################################
#                     copy pp-files to calculation directory                  #
###############################################################################


# shifts set of coordinates so that centroid is at centroid_final
def shift2center(coordinates_initial, centroid_final):
    centroid_initial = np.mean(coordinates_initial, axis=0)
    shift = centroid_final - centroid_initial
    return(coordinates_initial+shift)

# copy pp file for element specified by atomsym from pp-directory into directory for calculation calc_dir
def copy_pp_file(atomsym, filename_dest, calc_dir):
    filename_parent = atomsym + '_SG_LDA' 
    pp_path = os.path.join('/home/misa/software/PP_LIBRARY/', filename_parent)
    calc_path = os.path.join(calc_dir, filename_dest)
    copyfile(pp_path, calc_path)

# creates name of pp-file (atomsymbol followed by index of atom in compound and standard pp-file description)    
def generate_pp_file_name(idx, atomsym):
    return(atomsym + str(idx) + '_SG_LDA')

def generate_pp_file(calc_dir, idx, atomsym):
    pp_file_name_dest = generate_pp_file_name(idx, atomsym)
    copy_pp_file(atomsym, pp_file_name_dest, calc_dir)

def get_pp_files(calc_dir, compound):
    for idx in range(0, len(compound.atomtypes)):
        generate_pp_file(calc_dir, idx, compound.atomtypes[idx])
        
### keep input-file

def read_and_store(f, keyword):
    """
    read lines in input-file that should not be changed 
    till a keyword is reached for which input file must be changed
    return the unchanged lines as a list
    """
    line = ''
    lines = []
    #while (line != keyword+'\n' and line != keyword):
    while not(keyword in line):
        line = f.readline()
        lines.append(line)
    # ensure that new line after keyword
    if lines[len(lines)-1] == keyword:
        lines[len(lines)-1] = keyword+'\n'
    return(lines)

### change input-file

def write_keyword(filestream, keyword, val):
    """
    generate correct input for given keyword
    then skip lines in input-file that shall be replaced by generated input
    """
    result = None
    if keyword == '  CELL ABSOLUTE':
        filestream.readline() # skip next line in input-file, will be replaced with value of result
        result = write_cell_dim(val)
    elif keyword == '&ATOMS':
        result = write_atom_section(val)     
    else:
        assert('Unknown keyword')
    return(result, filestream)

###############################################################################
### set correct cell size CELL ABSOLUTE
   
def write_cell_dim(box_dim):
    """
    input length of cell in x,y,z as np.array; assumes cubic geometry
    returns the correct input line for CELL ABSOLUTE as a list with one element of type str
    """
    line1 = '  \t' + str(box_dim[0]) + ' ' + str(box_dim[1]) + ' ' + str(box_dim[2]) + ' 0.0 0.0 0.0\n'
    return([line1])

###############################################################################
### generate &ATOMS

def write_atom(idx, atomsym, coordinates):
    """
    prepare the input for one atom:
    the name of the pp is 'element_name' + idx of atom in Compound object + '_SG_LDA'
    the coordinates are read from Compund as well (must be shifted to center before)
    """
    line1 = f'*{atomsym}_SG_LDA\n'
    line2 = ' LMAX=S\n'
    line3 = ' 1\n'
    line4 = ' ' + str(coordinates[0]) + ' ' + str(coordinates[1]) + ' ' + str(coordinates[2]) + '\n'
    return( [line1, line2, line3, line4] )
    
def write_atom_section(compound):
    """
    concantenates inputs for individual atoms to one list where each element is one line of the input file
    """
    atom_section = []
    for idx in range(0, len(compound.atomtypes)):
        atom = write_atom(idx, compound.atomtypes[idx], compound.coordinates[idx])
        atom_section.extend(atom)
    atom_section.append('&END')
    return(atom_section)

###############################################################################
    
def generate_new_input(file, key_list):
    # shift molecule to center of box
    box_center = key_list['  CELL ABSOLUTE']/2.0
    key_list['&ATOMS'].coordinates = shift2center(key_list['&ATOMS'].coordinates, box_center)

    new_file = generate_file(file, key_list)
    return(new_file)
    
    
def modify_input(file, key_list):
    # replace input file with correct values
    modified_inp_file = generate_new_input(file, key_list)
    # write modified input to file
    with open(file, 'w') as f:
        f.writelines(modified_inp_file)
   
def initialize(calc_dir, compound, box_size):
    """
    generates pp for every atom and input-file for initial CPMD run
    """
    # generate pp's
    get_pp_files(calc_dir, compound)
    
    # generate input file
    input_file = os.path.join(calc_dir, 'run.inp')
    copyfile(PARENT_INP, input_file)
    key_list ={ '  CELL ABSOLUTE' : box_size, '&ATOMS':compound }
    modify_input(input_file, key_list)
    
###############################################################################
    
def generate_file(file, key_list):
    lines = []
    with open(file, 'r') as f:
        for keyword, val in key_list.items():
            lines.extend(read_and_store(f, keyword))
            new_lines, f = write_keyword(f, keyword, val)
            lines.extend(new_lines)
    return(lines)

def generate_new_pps(compound, lval, calc_dir):
    total_charge = 0
    for idx in range(0, len(compound.atomtypes)):
        pp_name = generate_pp_file_name(idx, compound.atomtypes[idx])        
        charge, pp_file = get_PP_file(compound.nuclear_charges[idx], lval)     
        pp_path = os.path.join(calc_dir, pp_name)       
        total_charge += charge        
        with open(pp_path, 'w') as f:
            f.writelines(pp_file)    
    return(total_charge)

    
def get_PP_file(Z, lval):
    element = lookup[Z]
    fn = '/home/misa/software/PP_LIBRARY/%s_SG_LDA' % element
    ret = []
    for line in open(fn).readlines():
        if 'ZV' in line:
            ZV = float(line.strip().split()[-1])
            ret.append('  ZV = %f' % (ZV*lval))
            continue
        if '#C' in line:
            parts = np.array([float(_) for _ in line.split('#')[0].strip().split()])
            parts[1:] *= lval
            if Z==6:
                formatstring = '%4d' + (len(parts)-1)*' %15.9f' + '   #C  C1 C2'
            elif Z==1:
                formatstring = '%4d' + (len(parts)-1)*' %15.9f' + '   #C  C1 C2 C3 ...' 
            else:
                assert('Error')
            ret.append(formatstring % (*parts,))
            continue
        ret.append(line[:-1])
    return ZV*(1-lval), '\n'.join(ret)
    

PARENT_INP = '/home/misa/APDFT/prototyping/atomic_energies/input-template/run-1/run.inp'
compound = qml.Compound(xyz='/home/misa/datasets/qm9/dsgdb9nsd_003700.xyz')
calc_dir = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/dsgdb9nsd_003700/box30/ve_8/trun'

if not os.path.exists(calc_dir):
    os.makedirs(calc_dir)

box_size = np.array([20.0, 20.0, 20.0])
initialize(calc_dir, compound, box_size)

###############################################################################



lookup = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
rlookup = {'H': 1, 'C': 6, 'O': 8, 'N': 7}

total = generate_new_pps(compound, 0.2, calc_dir)