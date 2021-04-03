from pyscf import gto, dft
import numpy as np
import os
import qml

import sys
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies/')
from alchemy_tools_pyscf import calculate_alchpot
from parse_pyscf import read_input

os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6

# read input file
run_dir = sys.argv[1]
os.chdir(run_dir)
input_parameters = read_input(os.path.join(run_dir, 'input_parameters'))
inputfile = input_parameters['structure_file']
basis =  'def2tzvp'#input_parameters['basis'] 
com = qml.Compound(xyz=inputfile)
# create mol object and set up calculation
mol = gto.Mole()
for ch, coords_atom in zip(com.nuclear_charges, com.coordinates):
    mol.atom.append([ch, coords_atom])
mol.basis = basis
mol.verbose = 0
mol.build()
includeonly = np.arange(len(mol.atom_coords())) # necessary for alchpot function

# run SCF-calculation
mf = dft.RKS(mol)
mf.xc = 'lda,vwn'
mf.kernel()

# calculate the alchemical potentials
dm1_ao = mf.make_rdm1()
alchpots = calculate_alchpot(dm1_ao, includeonly, mol)

np.save(os.path.join(run_dir, 'alchpots.npy'), alchpots)