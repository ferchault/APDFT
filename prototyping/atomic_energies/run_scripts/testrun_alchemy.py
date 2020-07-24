import sys
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies')
import alchemy_tools_pyscf as atp
from parse_pyscf import read_input
import qml
import numpy as np
import os

# get current directory
run_dir = os.getcwd()

print(f'I am in {run_dir}')

print('Initializing')
input_parameters = read_input(os.path.join(run_dir, 'input_parameters'))
inputfile = input_parameters['structure_file']#os.path.join(basepath, com+'.xyz')
intg_meth = input_parameters['intg_meth']
basis = input_parameters['basis'] # 'def2-qzvp'
com = qml.Compound(xyz=inputfile)

lam_vals = np.array([0.5, 1])#np.arange(2, 54, 2)/52
lam_vals = np.concatenate((np.zeros(1), lam_vals))
alchemical_potentials = []
alchemical_potentials.append(np.zeros(len(com.nuclear_charges)).tolist())
mo_energies = []
mo_occs = []
print(inputfile)
print(basis)
print(intg_meth)

np.savetxt(os.path.join(run_dir, 'test.out'), com.nuclear_charges)
