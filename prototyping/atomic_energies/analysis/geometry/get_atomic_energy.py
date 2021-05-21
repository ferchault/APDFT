import numpy as np
import ase
import glob
from matplotlib import pyplot as plt

import pandas as pd
import sys
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies/')
import utils_qm as uqm
from parse_density_files import CUBE
import alchemy_tools2 as at
from ase.units import Bohr
from explore_qml_data import get_num_val_elec
import os
from pyscf import gto, scf, mp, cc, dft

def get_EI_el(lam_vals, alchpots, nuc_charges):
    """
    inetgrate over tilde rho with respect to lambda
    """
    EI_el = []
    for i in range(len(nuc_charges)):
        EI_el.append(np.trapz(alchpots[:, i], lam_vals)*nuc_charges[i])
    return(np.array(EI_el))

def get_e_npbc(nuc_charges, positions):
    atom = []
    for a, c in zip(nuc_charges, positions*Bohr):
        atom.append([int(a), tuple(c)])

    mol = gto.Mole()
    mol.verbose = 0
    #mol.output = 'H2_{}.log'.format(d)
    mol.atom = atom
    mol.basis = 'def2tzvp'
    mol.build()

    # PBE
    mdft = dft.RKS(mol)
    mdft.xc = 'pbe'
    e_pbe = mdft.kernel()
    
    return(e_pbe)

def get_lambda(paths):
    lam_vals = []
    for p in paths:
        if 'DENSITY' in p:
            lam_vals.append(float(p.split('/')[-2][3:]))
        else:
            lam_vals.append(float(p.split('/')[-1].split('.')[0][3:]))
    
    lam_vals = np.array(lam_vals)
    lam_vals = lam_vals/lam_vals[-1]
    return(lam_vals)

def read_cube_data(paths_cubes):
    """
    returns the data necessary to calculate the atomic energies from the cube-files
    for different lambda values
    
    paths_cubes: paths to cubes files
    densities: densities given in different cube files
    lam_vals: lambda value for cube file, obtained by parsing filename
    nuclei: charges and coordinates of the nuclei
    gpts: the gridpoints where density values are given
    """
    
    densities = []
    nuclei = None # nuclear charges and their positions
    gpts = None # gridpoints where density values are given
    h_matrix = np.zeros((3,3)) # needed for the calculation of the distance of the nuclei to the gridpoints with MIC
    
    for idx, path in enumerate(paths_cubes):
        cube = CUBE(path)
        
        densities.append(cube.data_scaled) # density
        
        fname = os.path.split(path)[1] # lambda value
        fname = fname.split('.')[0]
        total_ve = get_num_val_elec(cube.atoms[:, 0])
        # check if integrates to correct number of electrons
        if not np.isclose(cube.data_scaled.sum(), total_ve):
            print(f'{path} does not integrate to correct num_ve')
        
        if idx==len(paths_cubes)-1:
            nuclei = cube.atoms
            gpts = cube.get_grid()
            h_matrix = [cube.X*cube.NX, cube.Y*cube.NY, cube.Z*cube.NZ]
    
    return(densities, nuclei, gpts, h_matrix)

def save_data(alchpots, densities, e_el, e_tot, e_npbc, e_scaled, fname, lam_vals, nuclei):
    """
        alchpots: 2D array, one row contain alchpots at one lambda value for all atoms in same order as specified in nuclei
        the lambda value increases with each row from 0 to 1

        e_el: atomic electronic energy obtained by integration along lambda for every atom in nuclei

        e_tot: addition of nuclear repulsion to e_el

        e_npbc: energy of a single point pbe calculation without pbc for the same configuration as specified in nuclei

        e_scaled: shift of e_tot such that the sum of the atomic energies is equal to the energy of e_nbpc
    """
    data = {'alchpots':alchpots, 'densities':densities, 'e_el':e_el, 'e_tot':e_tot, 'e_nbpc':e_npbc, 'e_scaled':e_scaled, 'lam_vals':lam_vals, 'nuclei':nuclei}
    uqm.save_obj(data, fname)

def scale_EI(e_alch, e_npbc, nuc_charges):
    shift = (e_npbc - e_alch.sum())/nuc_charges.sum()
    EI_scaled = []
    for e, z in zip(e_alch, nuc_charges):
        EI_scaled.append(e + z*shift)
    return(np.array(EI_scaled))

def wrapper_alchpots(densities, nuclei, meshgrid, h_matrix):
    """
    integrate electron density over r for single lambda values
    """
    alchpots = []
    for d in densities:
        out = at.calculate_atomic_energies(d, nuclei, meshgrid, h_matrix, intg_method = 'sum')
        alchpots.append(out[2])
    return(np.array(alchpots))

def wrapper_read_data(paths):
    # load data, parameters

    densities, nuclei, gpts, hmatrix = read_cube_data(paths)
    
    # get density of ueg
    num_ve = get_num_val_elec(nuclei[:, 0])
    ueg = np.zeros(gpts[0].shape)
    ueg[:] = num_ve/gpts[0].shape[0]**3
    assert np.isclose(ueg.sum(), num_ve), 'wrong ueg'
    densities.insert(0, ueg)


    lam_vals = get_lambda(paths)
    lam_vals = np.insert(lam_vals, 0, 0.0)
    
    lamval_copy = lam_vals.copy()
    lam_vals.sort()
    assert np.array_equal(lamval_copy, lam_vals), 'wrong sorting'
    
    return(densities, gpts, hmatrix, lam_vals, nuclei)

def atomic_energy_single_compound(compound_path):
    # make path to cube files
    cubes = ['ve_13.cube', 've_19.cube', 've_26.cube', 've_32.cube']
    paths = []
    for c in cubes:
        paths.append(os.path.join(compound_path, f'cube-files/{c}'))
    
    densities, gpts, hmatrix, lam_vals, nuclei = wrapper_read_data(paths)
    alchpots = wrapper_alchpots(densities, nuclei, gpts, hmatrix)
    e_el = get_EI_el(lam_vals, alchpots, nuclei[:,0])
    e_tot = e_el + at.nuclear_repulsion(nuclei[:,0], nuclei[:,1:])
    e_npbc = get_e_npbc(nuclei[:,0], nuclei[:,1:])
    e_scaled = scale_EI(e_tot, e_npbc, nuclei[:,0])
    
    fname = os.path.join(compound_path, 'results.dict')
    save_data(alchpots, densities, e_el, e_tot, e_npbc, e_scaled, fname, lam_vals, nuclei)
    
    
if __name__ == "__main__":
    compound_path = sys.argv[1]
    compounds = []
    with open(compound_path, 'r') as f:
        for line in f:
            compounds.append(line.strip('\n'))
    for c in compounds:
        atomic_energy_single_compound(c)