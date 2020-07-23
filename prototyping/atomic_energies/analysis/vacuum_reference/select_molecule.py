import numpy as np
import os
from shutil import copyfile


# get names of molecules
# select 100 names
compounds = []
with open('/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/paths_atomic_energies', 'r') as f:
    for line in f:
        compounds.append(line.strip('/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/').strip('/\n'))

np.random.shuffle(compounds)
selected_compounds = compounds[0:100]

# make directory for every compound
# copy qm9.xyz to directory
# tar, copy to alchemy
basepath = '/home/misa/projects/Atomic-Energies/data/vacuum_reference/qm9_data'
for c in selected_compounds:
    new_dir = os.path.join(basepath, c)
    try:
        os.mkdir(new_dir)    
        print("Directory " , new_dir ,  " Created ")
    except FileExistsError:
        print("Directory " , new_dir ,  " already exists")
    src = '/home/misa/datasets/qm9/'+c+'.xyz'
    copyfile(src, os.path.join(new_dir, c+'.xyz'))

