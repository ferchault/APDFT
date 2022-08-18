import os
import numpy as np
import sys
sys.path.insert(0, '/home/sahre/git_repositories/APDFT/prototyping/atomic_energies/hitp')
sys.path.insert(0, '/home/sahre/git_repositories/APDFT/prototyping/atomic_energies')
import explore_qml_data as eqd
import cpmd_io
import glob
import ase.io as aio

def sort_lambda_logfiles(search_dir):
    lambda_dirs = glob.glob(search_dir)
    sorted_lam_dirs = []
    sorted_lam_vals = []
    for l in lambda_dirs:
        lam_val = float(l.split('_')[-1])
        sorted_lam_dirs.append((lam_val, l))
        sorted_lam_vals.append(lam_val)
    sorted_lam_dirs.sort()
    sorted_lam_vals.sort()
    sorted_lam_vals = np.array(sorted_lam_vals)

    logfiles = []
    for lam in sorted_lam_dirs:
        if os.path.isfile(os.path.join(lam[1], 'run.log')):
            logfiles.append(os.path.join(lam[1], 'run.log'))
    return(sorted_lam_vals, logfiles)

def get_energy_contributions(logfiles):
    energy_contributions = {'TOTAL ENERGY =':[], 'KINETIC ENERGY =':[], 'ELECTROSTATIC ENERGY =':[], '(E+I)-(E+I) HARTREE ENERGY =':[], 
                   '(PSEUDO CHARGE I-I) ENERGY =':[], '(E-E) HARTREE ENERGY =':[], 'ESELF =':[], 'ESR =':[], 'LOCAL PSEUDOPOTENTIAL ENERGY =':[],
                   'N-L PSEUDOPOTENTIAL ENERGY =':[], 'EXCHANGE-CORRELATION ENERGY =':[], 'GRADIENT CORRECTION ENERGY =':[]}
    for p in logfiles:
        if os.path.isfile(p):
            with open(p, 'r') as f:
                logfile = f.readlines()

            for k in energy_contributions.keys():
                energy_contributions[k].append(cpmd_io.get_energy_contribution(logfile, k))
        else:
            print(f'Logfile at {p} does not exist')
    for k in energy_contributions.keys():
        energy_contributions[k] = np.array(energy_contributions[k])

    nuc_rep_cpmd = energy_contributions['(PSEUDO CHARGE I-I) ENERGY ='] + energy_contributions['ESR ='] - energy_contributions['ESELF =']
    e_el_cpmd = energy_contributions['TOTAL ENERGY ='] - nuc_rep_cpmd
    return(energy_contributions, e_el_cpmd, nuc_rep_cpmd)


def create_run_dir(path, atom_symbols, nuc_charges, positions, lambda_value, valence_charges = None, integer_electrons = False):

    # create directory if not exists
    os.makedirs(path, exist_ok=True)

    # set parameters independent of lambda value
    num_ve = eqd.get_num_val_elec(nuc_charges) # get number of ve
    boxsize = get_boxsize(num_ve) # get boxsize
    num_gpts_lower, num_gpts_higher = get_gpts(num_ve) # get gridpoints
    num_gpts = num_gpts_higher
    # shift molecule to center of box
    # coords_final = eqd.shift2center(positions, np.array([boxsize, boxsize, boxsize])/2)
    coords_final = positions
    # lambda dependent quantities

    # add correct number of electrons such that system stays isoelectronic to target molecule\
    charge = calculate_charge(lambda_value, num_ve, valence_charges, integer_electrons)

    # generate input file
    input_path = os.path.join(path, 'run.inp')
    
    # start from random wavefunction as initial guess if lambda < 0.5 otherwise the calculation will crash
    if type(lambda_value) == float:
        lam_min = lambda_value
    else:
        lam_min = np.amin(lambda_value)
        
    if lam_min > 0.501:
        write_input(atom_symbols['elIdx'], charge, coords_final, num_gpts, boxsize, input_path, pp_type, template_inp, debug = False)
    else:
        write_input(atom_symbols['elIdx'], charge, coords_final, num_gpts, boxsize, input_path, pp_type, template_inp_small_lambda, debug = False)

    # generate pp-files
    if type(lambda_value) == float:
        write_pp_files_compound(atom_symbols, lambda_value, path, pp_dir, pp_type)
    else:
        write_pp_files_compound_partial(atom_symbols, lambda_value, path, pp_dir, pp_type)
        
def parse_xyz_for_CPMD_input(path2xyz):
    # get structure information from xyz file
    molecule = aio.read(path2xyz)

    atom_symbolsEl = []
    atom_symbolsIdx = []
    for i, el in enumerate(molecule.get_chemical_symbols()):
        atom_symbolsEl.append(el)
        atom_symbolsIdx.append(el + str(i+1))
    atom_symbols = {'el':atom_symbolsEl, 'elIdx':atom_symbolsIdx}
    nuc_charges = molecule.get_atomic_numbers()
    valence_charges = eqd.get_val_charges(nuc_charges)
    positions = molecule.get_positions()
    return(atom_symbols, nuc_charges, positions, valence_charges)