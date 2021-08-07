#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:23:17 2020

@author: misa
"""

import pickle
import pysmiles

def load_obj(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
    
def save_obj(obj, fname ):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def read_list(fname):
    data = []
    with open(fname, 'r') as f:
        for line in f:
            data.append(line.strip('\n'))
    return(data)

def write_list(fname, ls):
    with open(fname, 'w') as f:
        for line in ls:
            f.write(line+'\n')
            
def ase2pyscf(molecule):
    """
    return mol.atom for pyscf gto.Mole object from ase Atoms object
    """
    elements = molecule.get_chemical_symbols()
    positions = molecule.get_positions()
    atoms = []
    for e, p in zip(elements, positions):
        atoms.append([e, tuple(p)])
    return(atoms)

def fragmentize_smiles(smiles):
    mol = pysmiles.read_smiles(smiles)
    num_atoms = len(mol.nodes.data('element'))
    frags = []
    for i in range(num_atoms):
        el = mol.nodes.data('element')[i]
        h = mol.nodes.data('hcount')[i]
        if el not in ['C', 'N', 'O', 'F', 'S', 'P', 'Cl', 'Br']:
            if h > 1:
                frag = f'[{el}H{h}]'
            elif h == 1:
                frag = f'[{el}H]'
            else:
                frag = f'[{el}]'
        else:
            frag = el
        frags.append(frag)
    return(frags)

def sort_paths(paths, str_pos1, str_pos2, split1, split2):
    """
    sort paths by some numerical value that is hidden in the path
    """
    paths_tuple = []
    for p in paths:
        sorting_string = p.split(split1)[str_pos1]
        value = float(sorting_string.split(split2)[str_pos2])
        paths_tuple.append((value, p))
    paths_tuple.sort()
    paths_sorted = []
    for p in paths_tuple:
        paths_sorted.append(p[1])
    return(paths_sorted)