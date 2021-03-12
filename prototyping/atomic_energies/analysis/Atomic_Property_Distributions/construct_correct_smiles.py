from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import glob
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies')
import explore_qml_data as eqd

import sys
sys.path.insert(0, '/home/misa/git_repositories/xyz2mol')
import xyz2mol

files = glob.glob('/home/misa/datasets/qm9/dsgdb9nsd_*')
files.sort()

correct_smiles = []
for filename in files:
    # read atoms and coordinates. Try to find the charge
    atoms, charge, xyz_coordinates = xyz2mol.read_xyz_file(filename)

    # Get the molobjs
    mols = xyz2mol.xyz2mol(atoms, xyz_coordinates,
        charge=charge,
        use_graph=True,
        allow_charged_fragments=True,
        embed_chiral=False,
        use_huckel=False)
    if mols == 'error':
        correct_smiles.append(['error', filename])
    else:
        for mol in mols:
            # Canonical hack
            isomeric_smiles = True
            smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric_smiles)
            m = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(m, isomericSmiles=isomeric_smiles)
            #print(smiles)
        correct_smiles.append([smiles, filename])
        
with open('/home/misa/datasets/qm9/correct_smiles.txt', 'w') as f:
    for item in correct_smiles:
        f.write(f'{item[0]} {item[1]}\n')