import sys
sys.path.insert('/home/misa/git_repositories/APDFT/prototyping/atomic_energies')
import alchemy_tools_pyscf as atp
from parse_pyscf import read_input
import qml
import numpy as np

# get current directory
run_dir = os.getcwd()

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

print('Starting calculations')
for lam in lam_vals[1:]:
    num_elec = atp.get_num_elec(lam, com.nuclear_charges.sum())
    print('Preparing input')
    deltaZ, includeonly, mol = atp.prepare_input(com.coordinates, com.nuclear_charges, num_elec, basis)
    print('Doing SCF calculation')
    dm, e_tot, mo_energy, mo_occ = atp.make_apdft_calc(deltaZ, includeonly, mol, method = "HF")
    mo_energies.append(mo_energy)
    mo_occs.append(mo_occ)
    print('Calculating alchemical potentials')
    alchpots_lambda = atp.calculate_alchpot(dm, includeonly, mol)
    alchemical_potentials.append(alchpots_lambda)
    
alchemical_potentials = np.array(alchemical_potentials)
average_potentials = atp.calculate_average_alchpots(alchemical_potentials, lam_vals, intg_meth)

print('Saving results')
# save alchemical_potentials
file_raw = os.path.join(run_dir, 'alchpots_raw')
np.save(file_raw, alchemical_potentials, allow_pickle=False)
# save mo energies
file_moen = os.path.join(run_dir, 'mo_energies_raw')
np.save(file_moen, mo_energies, allow_pickle=False)
# save mo occupancies
file_moocc = os.path.join(run_dir, 'mo_occupancies_raw')
np.save(file_moocc, mo_occs, allow_pickle=False)
# save lam_vals
np.savetxt(os.path.join(run_dir, 'lam_vals.txt'))
# save average potentials with charges and coords
save = np.array([com.nuclear_charges, com.coordinates[:,0], com.coordinates[:,1], com.coordinates[:,2], average_potentials]).T
header = 'charge\t x_coord\t y_coord\t z_coord\t alchemical_potential'
save_dir = os.path.join(run_dir, f'alchpots_{basis}_{intg_meth}.txt')
np.savetxt(save_dir, save, delimiter='\t', header = header)
