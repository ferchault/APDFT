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

def get_energy_contribution_wrapper(logfile_path, energy_contr):
    """
    prepare the run dirs for finite difference partial derivatives for small molecules
    comp: name of compound
    delta_lambda: delta for finite difference
    lam: lambda_value for which derivatives will be calculated
    """
    energy = None
    with open(logfile_path, 'r') as f:
        logfile = f.readlines()
            # get energy
    if energy_contr == 'total':
        energy = cpmd_io.get_energy_contribution(logfile, 'TOTAL ENERGY =')
    elif energy_contr == 'ion_pseudo':
        energy = cpmd_io.get_energy_contribution(logfile, '(PSEUDO CHARGE I-I) ENERGY =')
    elif energy_contr == 'ion_self':
        energy = cpmd_io.get_energy_contribution(logfile, 'ESELF =')
    elif energy_contr == 'ion_esr':
        energy = cpmd_io.get_energy_contribution(logfile, 'ESR =')
    elif energy_contr == 'kinetic':
        energy = cpmd_io.get_energy_contribution(logfile, 'KINETIC ENERGY =')
    elif energy_contr == 'xc_corr':
        energy = cpmd_io.get_energy_contribution(logfile, 'EXCHANGE-CORRELATION ENERGY =') 
    elif energy_contr == 'local_pp':
        energy = cpmd_io.get_energy_contribution(logfile, 'LOCAL PSEUDOPOTENTIAL ENERGY')
    elif energy_contr == 'nonlocal_pp':
        energy = cpmd_io.get_energy_contribution(logfile, 'N-L PSEUDOPOTENTIAL ENERGY =')
    elif energy_contr == 'electrostatic':
        energy = cpmd_io.get_energy_contribution(logfile, 'ELECTROSTATIC ENERGY =')
    elif energy_contr == 'electronic':
        e_el_parts = []
        for e in ['TOTAL ENERGY =','(PSEUDO CHARGE I-I) ENERGY =', 'ESR =','ESELF =']:
            e_el_parts.append(cpmd_io.get_energy_contribution(logfile, e))
        e_nuc = e_el_parts[1]+e_el_parts[2]-e_el_parts[3]
        e_el = e_el_parts[0]-e_nuc
        energy = e_el
    elif energy_contr == 'potential':
        e_pot_parts = []
        for e in ['TOTAL ENERGY =','(PSEUDO CHARGE I-I) ENERGY =', 'ESR =','ESELF =','KINETIC ENERGY =']:
            e_pot_parts.append(cpmd_io.get_energy_contribution(logfile, e))
        epot = e_pot_parts[0] - (e_pot_parts[1]+e_pot_parts[2]-e_pot_parts[3]) - e_pot_parts[4]
        energy = epot
    elif energy_contr == 'nuclear':
        e_nuc_parts = []
        for e in ['(PSEUDO CHARGE I-I) ENERGY =', 'ESR =','ESELF =']:
            e_nuc_parts.append(cpmd_io.get_energy_contribution(logfile, e))
        e_nuc = e_nuc_parts[0]+e_nuc_parts[1]-e_nuc_parts[2]
        energy = e_nuc
    assert energy != None, f'Could not extract {energy_contr} from {logfile_path}'
    return(energy)

def calculate_partial_derivatives(atom, comp, delta_lambda, energy_contr, lambda_values):
    """
    prepare the run dirs for finite difference partial derivatives for small molecules
    comp: name of compound
    delta_lambda: delta for finite difference
    lam: lambda_value for which derivatives will be calculated
    """

    partial_derivatives = []

    for lam in lambda_values:
        energies = []
        for fd in ['bw', 'fw']:
            logfile_path = f'/data/sahre/projects/finite_differences/small_molecules/{comp}/lam_{np.round(lam, 3)}/{atom}/{fd}/run.log'
            with open(logfile_path, 'r') as f:
                logfile = f.readlines()
            # get energy
            energies.append(get_energy_contribution_wrapper(logfile_path, energy_contr))
        partial_derivatives.append((energies[1]-energies[0])/(2*delta_lambda))
    return(partial_derivatives)