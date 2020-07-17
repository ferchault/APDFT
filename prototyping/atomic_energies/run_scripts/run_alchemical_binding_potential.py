import sys
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies/')
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies/hitp')

from ase.units import Bohr
import numpy as np
import qml_interface as qmi
import alchemy_tools2 as at
import utils_qm as uq
from find_converged import concatenate_files
import os

def get_meshgrid(lx, ly, lz, nx, ny, nz):
    """
    returns the coordinates of the grid points where the density values are given as a meshgrid
    works so far only for orthogonal coordinate axes
    """
    # length along the axes
    l_x = lx[0]*nx
    l_y = ly[1]*ny
    l_z = lz[2]*nz
    # gpts along every axis
    x_coords = np.linspace(0, l_x-lx[0], nx)
    y_coords = np.linspace(0, l_y-ly[1], ny)
    z_coords = np.linspace(0, l_z-lz[2], nz)
    # create gridpoints
    meshgrid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    return(meshgrid)

def get_alchpot_free(nuclei, densities_free_atoms, meshgrid, h_matrix, pos_free_atom = np.array([10.0, 10, 10])/Bohr):
    """
    calculate alchemical potential of free atoms at position of every nucleus
    length units should be given in Bohr
    """
    alch_pot_free = []
    for atom_I in nuclei:
        alch_pot_free_I = 0
        for atom_J in nuclei:
            # get density of free atom J
            nuc_charge = atom_J[0]
            density_fa = densities_free_atoms[nuc_charge]
            # calculate distance of R_I to all gridpoints (shift because free J is in center of box)
            s = (pos_free_atom - atom_J[1:4])
            RI_prime = atom_I[1:4] + s
            dist = at.distance_MIC2(RI_prime, meshgrid, h_matrix)
            # integrate
            elec_pot = -(density_fa/dist).sum()
            alch_pot_free_I += elec_pot
        alch_pot_free.append(alch_pot_free_I)  
    return(np.array(alch_pot_free))

# load average density of free atoms
base_path = '/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/free_atoms/'
densities_free_atoms = {1.0:None, 6.0:None, 7.0:None, 8.0:None}
elements = ['H', 'C', 'N', 'O']
for el, k in zip(elements, densities_free_atoms):
    densities_free_atoms[k] = uq.load_obj(base_path + f'av_dens_{el}')

# get meshgrid and h_matrix
#  cell_parameters
nx, ny, nz = densities_free_atoms[1.0].shape
lx, ly, lz = np.array([[20/(Bohr*nx), 0 , 0], [0, 20/(Bohr*ny), 0], [0, 0, 20/(Bohr*nz)]])

h_matrix = [lx*nx, ly*ny, lz*nz]
meshgrid = get_meshgrid(lx, ly, lz, nx, ny, nz)


# paths to the compounds
dirs = concatenate_files(['/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/paths_atomic_energies'])

for compound_path in dirs:
    # load ml data files instead of cuves
    molecule = np.loadtxt(compound_path + 'atomic_energies_with_mic.txt')
    alchpot_free = get_alchpot_free(molecule, densities_free_atoms, meshgrid, h_matrix)

    alchpot_bind = molecule[:, 4] - alchpot_free
    atomic_atomisation_pbc = alchpot_bind*molecule[:,0]

    # write atomic energies and alchemical potentials to file
    store = np.array([molecule[:,0], molecule[:,1], molecule[:,2], molecule[:,3], molecule[:, 4], alchpot_free, alchpot_bind, atomic_atomisation_pbc]).T
    header = 'charge\t x_coord\t y_coord\t z_coord\t alchemical_potential\t alch_pot_free\t alch_pot_bind\t atomic_atomisation_pbc'
    save_dir = os.path.join(compound_path, 'atomic_binding_energies_explicit.txt')
    np.savetxt(save_dir, store, delimiter='\t', header = header)
