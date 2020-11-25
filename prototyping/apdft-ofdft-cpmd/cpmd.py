import numpy as np
from ase import Atoms
from ase import units
import profess_io as pio

def initialize_atoms(pos_file, vel_file, pos_type = 'CART'):
    """
    generate atoms object with:
        - correct positions
        - cell paramters
        - set initial velocities
    """
    cell_param, positions, atom_types = pio.parse_ion_file_complete(pos_file)
#     positions = shift_atoms(positions)

    # velocities from Bohr/aut (PROFESS) -> Ang/ (Ang sqrt(aum/eV)) (ASE)
    # Bohr -> Ang = length_in_Bohr*units.Bohr
    # aut -> Ang sqrt(aum/eV)
    aut2fs = 2.41888433e-2 # go from aut -> fs
    aut2aset = aut2fs*units.fs # from aut -> aset (ase time unit)
    au_vel2ase_vel = units.Bohr/aut2aset # conversion of velocities in a.u. to native ase units
            
    velocities = np.array(pio.parse_velocity_file(vel_file))*au_vel2ase_vel
    if pos_type == 'FRAC':
        atoms = Atoms(atom_types, scaled_positions = positions, cell = cell_param, pbc = True)
    elif pos_type == 'CART':
        atoms = Atoms(atom_types, positions = positions, cell = cell_param, pbc = True)
    atoms.set_velocities(velocities)
    return(atoms)

def initialize_atoms_old(pos_file, vel_file, cell_param, pos_type = 'FRAC'):
    """
    generate atoms object with:
        - correct positions
        - cell paramters
        - set initial velocities
    """
    positions = pio.parse_ion_file(pos_file)
#     positions = shift_atoms(positions)

    # velocities from Bohr/aut (PROFESS) -> Ang/ (Ang sqrt(aum/eV)) (ASE)
    # Bohr -> Ang = length_in_Bohr*units.Bohr
    # aut -> Ang sqrt(aum/eV)
    aut2fs = 2.41888433e-2 # go from aut -> fs
    aut2aset = aut2fs*units.fs # from aut -> aset (ase time unit)
    au_vel2ase_vel = units.Bohr/aut2aset # conversion of velocities in a.u. to native ase units
            
    velocities = np.array(pio.parse_velocity_file(vel_file))*au_vel2ase_vel
    if pos_type == 'FRAC':
        atoms = Atoms('Al4', scaled_positions = positions, cell = cell_param, pbc = True)
    elif pos_type == 'CART':
        atoms = Atoms('Al4', positions = positions, cell = cell_param, pbc = True)
    atoms.set_velocities(velocities)
    return(atoms)

def shift_atoms(positions, length=3.97):
    positions_shifted = []
    for atom in positions:
        atom_shifted = []
        for c in atom:
            if c < 0:
                new_c = c + length
            else:
                new_c = c
            atom_shifted.append(new_c)
        positions_shifted.append(atom_shifted)
    return(positions_shifted)