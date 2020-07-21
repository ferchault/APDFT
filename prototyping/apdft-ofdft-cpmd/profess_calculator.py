from ase import Atoms
from ase.calculators.general import Calculator
from ase import units
import profess_io as pio
import subprocess
import os
import numpy as np

class PROFESS(Calculator):
    """
    calls PROFESS using the parameters given in __init__
    """
    name = 'PROFESS'
    implemented_properties = ['forces']

    def __init__(self, run_dir=None, inpt_name=None, pp_names=None, atoms=None, pos_type = 'CART'):
        self.atoms = atoms
        self.run_dir = run_dir
        self.inpt_name = inpt_name
        self.pp_names = pp_names
        self.energy_zero = 0.0
        
    def initialize(self, run_dir=None, inpt_name=None, pp_names=None, atoms=None, pos_type = 'CART'):
        self.atoms = atoms
        self.run_dir = run_dir
        self.inpt_name = inpt_name
        self.pp_names = pp_names
        self.energy_zero = 0.0
      
    def run_profess(self):
        os.chdir(self.run_dir)
        p = subprocess.run(['/home/misa/git_repositories/PROFESS/PROFESS', self.inpt_name], capture_output = True,  text=True )
        return(p)
    
    def update(self, atoms):
        if atoms != self.atoms:
            self.atoms = atoms
    
    def parse_out_for_en(self, stdout):
        outfile = stdout.split('\n')
        potential_energy = 0
        for line in outfile:
            if 'Final total energy' in line:
                potential_energy = float(line.split()[3])
                break
        potential_energy *= units.Hartree
        return(potential_energy)
    
#     def get_potential_energy(self):
#         print('I am called')
#         # write new .ion file
#         cell_par = self.atoms.get_cell()
#         atom_types = self.atoms.get_chemical_symbols()
#         positions = self.atoms.get_positions()
#         pos_type = 'CART'
#         new_ion = pio.generate_ion_file(cell_par, atom_types, positions, pos_type, self.pp_names)
#         pio.write_file(f'{os.path.join(self.run_dir, self.inpt_name)}.ion', new_ion)
#         # call profess
#         completed_p = self.run_profess()
#         assert completed_p.returncode == 0, 'Calculation of energy failed'
#         # parse output for energy
#         self.energy_zero = self.parse_out_for_en(completed_p.stdout)
#         return(self.energy_zero)
        
    def get_forces(self, atoms=None):
        # write new .ion file
        cell_par = self.atoms.get_cell()
        atom_types = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        pos_type = 'CART'
        new_ion = pio.generate_ion_file(cell_par, atom_types, positions, pos_type, self.pp_names)
        pio.write_file(f'{os.path.join(self.run_dir, self.inpt_name)}.ion', new_ion)
        # call profess
        completed_p = self.run_profess()
        assert completed_p.returncode == 0, 'Calculation of forces failed'
        # update energy
        self.energy_zero = self.parse_out_for_en(completed_p.stdout)
        # read forces
        self.forces = np.array(pio.parse_force_file(f'{os.path.join(self.run_dir, self.inpt_name)}.force.out'))
        # ensures that programm crashes if computation of forces fails in next step
        os.remove(f'{os.path.join(self.run_dir, self.inpt_name)}.force.out')
        return(self.forces)
    
class PROFESS_old(Calculator):
    """
    calls PROFESS using the parameters given in __init__
    """
    name = 'PROFESS'
    implemented_properties = ['forces']

    def __init__(self, run_dir, inpt_name, pp_names, atoms, pos_type = 'CART'):
        self.atoms = atoms.copy()
        self.run_dir = run_dir
        self.inpt_name = inpt_name
        self.pp_names = pp_names
        self.energy_zero = 0.0
        
    def run_profess(self):
        os.chdir(self.run_dir)
        p = subprocess.run(['/home/misa/git_repositories/PROFESS/PROFESS', self.inpt_name])
        return(p)
    
    def update(self, atoms):
        self.atoms = atoms

    def get_forces(self, atoms=None):
        # write new .ion file
        cell_par = self.atoms.get_cell()
        atom_types = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        pos_type = 'CART'
        new_ion = pio.generate_ion_file(cell_par, atom_types, positions, pos_type, self.pp_names)
        pio.write_file(f'{os.path.join(self.run_dir, self.inpt_name)}.ion', new_ion)
        # call profess
        completed_p = self.run_profess()
        assert completed_p.returncode == 0, 'Calculation of forces failed'
        # read forces
        self.forces = np.array(pio.parse_force_file(f'{os.path.join(self.run_dir, self.inpt_name)}.force.out'))
        # ensures that programm crashes if computation of forces fails in next step
        os.remove(f'{os.path.join(self.run_dir, self.inpt_name)}.force.out')
        return(self.forces)