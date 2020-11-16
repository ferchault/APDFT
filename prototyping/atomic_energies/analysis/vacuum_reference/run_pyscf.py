import sys
sys.path.insert(0,'/home/misa/git_repositories/APDFT/prototyping/atomic_energies')
import alchemy_tools_pyscf as atp
from parse_pyscf import read_input
import qml
import numpy as np
import os
import utils_qm as uqm

# get current directory
run_dir = os.getcwd()
print(f'I am in directory {run_dir}')
os.chdir(run_dir)
print('Initializing')
input_parameters = read_input(os.path.join(run_dir, 'input_parameters'))
inputfile = input_parameters['structure_file']#os.path.join(basepath, com+'.xyz')
intg_meth = input_parameters['intg_meth']
basis = input_parameters['basis'] # 'def2-qzvp'
com = qml.Compound(xyz=inputfile)

#lam_vals = #np.array([2, com.nuclear_charges.sum()/2, com.nuclear_charges.sum()])/com.nuclear_charges.sum()  #np.arange(2, com.nuclear_charges.sum(), 2)/com.nuclear_charges.sum()
lam_vals = np.arange(2, com.nuclear_charges.sum()+2, 2)/com.nuclear_charges.sum()
#lam_vals = np.concatenate((np.zeros(1), lam_vals))
alchemical_potentials = []
#alchemical_potentials.append(np.zeros(len(com.nuclear_charges)).tolist())
mo_energies = []
mo_occs = []
dm_restart = None

# print('Starting calculations')
for lam in lam_vals: #np.flip(lam_vals)[:-1]:
    num_elec = atp.get_num_elec(lam, com.nuclear_charges.sum())
    print(f'Doing calculation for lambda = {lam}, number of electrons = {num_elec}')
    print('Preparing input')
    deltaZ, includeonly, mol = atp.prepare_input(com.coordinates, com.nuclear_charges, num_elec, basis)
    print('Doing SCF calculation')
    dm, e_tot, mo_energy, mo_occ = atp.make_apdft_calc(deltaZ, dm_restart, includeonly, mol, method = "HF", **{'max_cycle':300, 'init_guess':'atom'})
#     dm_restart = dm
    mo_energies.append(mo_energy)
    mo_occs.append(mo_occ)
    print('Calculating alchemical potentials')
    alchpots_lambda = atp.calculate_alchpot(dm, includeonly, mol)
    alchemical_potentials.append(alchpots_lambda) 
#   saving results
    uqm.save_obj(alchemical_potentials, os.path.join(run_dir, 'alchemical_potentials_tmp'))
    uqm.save_obj(mo_energies, os.path.join(run_dir, 'mo_energies_tmp'))
    uqm.save_obj(mo_occs, os.path.join(run_dir, 'mo_occupancies_tmp'))
    uqm.save_obj(lam_vals[:np.where(lam_vals==lam)[0][0]+1], os.path.join(run_dir, 'lam_vals_tmp'))
    
alchemical_potentials = np.array(alchemical_potentials)
# average_potentials = atp.calculate_average_alchpots(alchemical_potentials, lam_vals, intg_meth)

print('Saving results')
# save alchemical_potentials
file_raw = os.path.join(run_dir, 'alchpots')
np.save(file_raw, alchemical_potentials, allow_pickle=False)
# save mo energies
file_moen = os.path.join(run_dir, 'mo_energies')
np.save(file_moen, mo_energies, allow_pickle=False)
# save mo occupancies
file_moocc = os.path.join(run_dir, 'mo_occupancies')
np.save(file_moocc, mo_occs, allow_pickle=False)
# save lam_vals
np.savetxt(os.path.join(run_dir, 'lam_vals.txt'),lam_vals)
