#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:23:17 2020

@author: misa
"""

import pickle
import pysmiles
import leruli
import numpy as np
import rdkit.Chem
from ase import Atoms

import sys
sys.path.insert(0, '/home/sahre/git_repositories/xyz2mol')
from xyz2mol import xyz2mol

def load_obj(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
    
def save_obj(obj, fname):
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

def graph2geometry(smiles):
    """
    uses leruli to create 3D structure from smiles
    returns list of atom symbols and 2D array with spatial coordinates
    """
    out = leruli.graph_to_geometry(smiles, 'XYZ')
    geo = out['geometry'].split('\n')[2:-1]
    elements = []
    coords = []
    for g in geo:
        elements.append(g.split()[0])
        coords_tmp = []
        for c in g.split()[1:]:
            coords_tmp.append(float(c))
        coords.append(coords_tmp)
    return(elements, coords)

def hcount_smiles(smiles):
    """
    returns list of number of hydrogens bonded to each heavy atom in smiles
    """
    
    mol = pysmiles.read_smiles(smiles)
    num_h = []
    for i in range(len(mol.nodes.data('hcount'))):
        num_h.append(mol.nodes.data('hcount')[i])
    return(num_h)

def parse_QM9(file):
    with open(file, 'r') as f:
        num_atoms = f.readline()
        num_atoms = int(num_atoms.strip('\n')) # get number of atoms
        f.readline() # comment line

        # get elements and coordinates
        elements = []
        coords = []
        for i in range(num_atoms):
            el_i, x_i, y_i, z_i, mulliken_i = f.readline().strip('\n').split()
            elements.append(el_i)
            coords.append([float(x_i), float(y_i), float(z_i)])

        coords = np.array(coords)
    return(elements, coords)

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

def QM9xyz2mol(path2QM9xyz):
    elements, coords = parse_QM9(path2QM9xyz)
    atoms = Atoms(elements, coords)

    numbers = []
    for i in atoms.get_atomic_numbers():
        numbers.append(int(i))
    mol = xyz2mol(numbers, atoms.get_positions())[0]
    # clean up messed up hydrogens in mol
    smiles = rdkit.Chem.MolToSmiles(mol)
    mol = rdkit.Chem.MolFromSmiles(smiles)
    smiles = rdkit.Chem.MolToSmiles(mol)
    return(mol)
