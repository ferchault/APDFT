#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:45:44 2019

@author: misa
"""

def test_get_property():
    f=open('/home/misa/datasets/qm9/dsgdb9nsd_002626.xyz', 'r')
    props = f.readline()
    props = f.readline()
    f.close()
    total_en = float(props.split('\t')[11])
    
def test_get_free_atom_data():
    # energies of the free atoms
    f_ref=open('/home/misa/datasets/atomref_qm9.txt', 'r')
    lines=f_ref.readlines()
    f_ref.close()
    atoms = lines[5:10]
    
    free_atoms = {}
    for atom in atoms:
        energy = float(atom.split()[2])
        symbol = atom.split()[0]
        free_atoms[symbol] = energy