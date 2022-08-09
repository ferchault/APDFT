#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:03:20 2019

@author: misa

Set of functions to 
a) explore datasets where data entries are qml.compounds
b) parse qm9 files for properties
"""

import numpy as np
import qml
from matplotlib import pyplot as plt
import os

from ase import Atoms
from ase.visualize import view

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

###############################################################################
###                                I/O-functions
###############################################################################
def load_compounds(path_list):
    """
    loads qm9 entries given in path_list in Compound objects, returns list with compounds
    """
    compounds = []
    for file in path_list:
        mol = qml.Compound(xyz=file)
        compounds.append(mol)
    return(compounds)

def paths_slice_ve(num_ve, path_list=None):
    """
    input: num_ve:number of valence electrons, paths:paths to xyz files of data
    return: paths to compounds of qm9 with the number of valence electrons specified in the input
    """
    if path_list == None:
    # build path to data
        d = '/home/misa/datasets/qm9/'
    #     d = '/data/sahre/datasets/qm9'
        path_list = [os.path.join(d, f) for f in os.listdir(d)]
        path_list.sort()
    
    #calculate number of valence electrons for every compound
    compounds = []
    num_val = np.empty(len(path_list), dtype=int)
    for idx, file in enumerate(path_list):
        mol = qml.Compound(xyz=file)
        num_val[idx] = get_num_val_elec(mol.nuclear_charges)
        
        if num_val[idx] == num_ve:
            compounds.append(file)
    return(compounds)

def write_paths_slice_ve(num_ve, path):
    """
    Description:
        writes the paths to qm9 entries with specified number of electrons in a txt-file
    input: 
        num_ve: number of valence electrons
        path:   path to file
    """
    compounds = paths_slice_ve(num_ve)
    with open(path, 'w') as f:
        for item in compounds:
            f.write("%s\n" % item)

###############################################################################
###                                size of molecule
###############################################################################

def find_largest_mol(compounds):
    """
    input list with elements of type qml.compound
    returns compound with the largest distance between geometric center and nucleus
    """
    idx_max = idx_largest_comp(compounds)  
    return(compounds[idx_max])
    
def max_dist_center_nuc(com):
    """
    returns the largest distance between geometric center and nucleus for qml.compound
    """
    centroid = np.mean(com.coordinates, axis=0)
    distance2centroid = np.linalg.norm(centroid - com.coordinates, axis = 1)
    max_dist = np.amax(distance2centroid)
    return( max_dist)
    
def idx_largest_comp(compounds):
    """
    input list with elements of type qml.compound
    returns index of compound with the largest distance between geometric center and nucleus
    """
    distances = np.empty(len(compounds))
    for idx, com in enumerate(compounds):
        distances[idx] = max_dist_center_nuc(com)
    idx_max_dist = np.where( distances == np.amax(distances) )
    return(idx_max_dist[0][0])
    
def max_dist_distribution(compounds):
    """
    input: list with elements of type qml.compound
    returns: largest distance between geometric center and nucleus for every compound
    """
    distances = np.empty(len(compounds))
    for idx, com in enumerate(compounds):
        distances[idx] = max_dist_center_nuc(com)
    return(distances)

###############################################################################
###                                show molecules
###############################################################################

    
def get_smiles(file):
    """
    parse smiles string from files in QM9 dataset
    """
    with open(file, 'r') as f:
        content = f.readlines()
        smiles = content[-2].split()[0]
    return(smiles)

def show(compound, viewer='Avogadro'):
    """
    compound: compound of qm9 in qml format
    shows molecular geometry in Avogadro
    """
    com = Atoms(positions=compound.coordinates, symbols=compound.atomtypes)
    view(com)
    
def moltosvg(mol, molSize = (300,300), kekulize = True):
    """
    show structure of rdkit mol object
    """
    
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:','')

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

###############################################################################
###                                other
###############################################################################

def get_charge_neighbours(mol):
    """
    returns the nuclear charge of the bonding partners for all atoms in mol
    if mol has hydrogens added, also their neighbours will be considered
    """
    charge_neighbours = []
    chN = []
    chH = []
    for atom in mol.GetAtoms():
        degree = atom.GetTotalDegree() # number of binding partners
        neighbors = atom.GetNeighbors()
        charges_neighbors = 0
        for n in neighbors:
            charges_neighbors += n.GetAtomicNum()
        chN.append(charges_neighbors)
        charge_H = degree - len(neighbors) # number of hydrogens bonded that are not explicit; is equal to charge of implicit hydrogens
        chH.append(charge_H)
        charge = (charges_neighbors + charge_H)
        charge_neighbours.append(charge)
    return(charge_neighbours)  


def get_num_val_elec(nuclear_charges):
    """
    calculates total number of valence electrons of molecule up to third row of periodic table
    input: list of nuclear charges
    return: total number of valence electrons
    """
    num_val = 0
    for charge in nuclear_charges:
        el = 0
        if charge <=2:
            num_val += charge
        elif charge >= 3 and charge <= 10:
            el = charge - 2
            num_val += el
        elif charge >= 11 and charge <= 18:
            el = charge - 10
            num_val += el
        else:
            assert('Cannot calculate number of valence electrons!')
    return(num_val)

def get_free_atom_data(path='/home/misa/datasets/atomref_qm9.txt'):
    """
    returns a dictionary of shape {key:element} {element_symbol : B3LYP energy of free atom at 0K (from qm9)}
    for elements used in qm9 dataset using atomref.txt from https://doi.org/10.6084/m9.figshare.c.978904.v5
    
    path: path to atomref data 
    """
    f_ref=open(path, 'r')
    lines=f_ref.readlines()
    f_ref.close()
    atoms = lines[5:10]
    
    free_atoms = {}
    for atom in atoms:
        energy = float(atom.split()[2])
        symbol = atom.split()[0]
        free_atoms[symbol] = energy

    return(free_atoms)
    
def get_free_atom_energies(nuc, e_free):
    """
    returns np.array with energy of the free atoms for every element of nuc
    
    nuc: list of nuclear charges
    e_free: energy of free atoms used in qm9 as dict {element_symbol:energy}
    """
    
    energy_free_atom = np.zeros(len(nuc))
    
    for idx, n in enumerate(nuc):
        
        if int(n)==1:
            energy_free_atom[idx] = e_free['H']
        elif int(n) == 6:
            energy_free_atom[idx] = e_free['C']
        elif int(n) == 7:
            energy_free_atom[idx] = e_free['N']
        elif int(n) == 8:
            energy_free_atom[idx] = e_free['O']
        elif int(n) == 9:
            energy_free_atom[idx] = e_free['F']
        else:
            raise Exception("Element not in qm9")
            
    return(energy_free_atom)

def get_property(path, prop):
    """
    returns a property of a compound of qm9 (see doi: 10.1038/sdata.2014.22 (2014).)
    
    path: path to the xyz-file of the compund
    prop: returned property
        U0: Internal energy at 0 K; unit: Ha
    """
    
    f=open(path, 'r')
    props = f.readline()
    props = f.readline()
    f.close()
    if prop == 'U0':
        return(float(props.split('\t')[11]))
    elif prop == 'atomisation_energy':
        total_en = float(props.split('\t')[11])
        energy_free_atoms = get_free_atom_energies(qml.Compound(path).nuclear_charges, get_free_atom_data()).sum()
        atomisation_energy = total_en - energy_free_atoms
        return(atomisation_energy)
    else:
        raise Exception("Unknown Property")

def shift2center(coordinates_initial, centroid_final):
    """
    shifts set of coordinates so that centroid is at centroid_final
    """
    centroid_initial = np.mean(coordinates_initial, axis=0)
    shift = centroid_final - centroid_initial
    return(coordinates_initial+shift)