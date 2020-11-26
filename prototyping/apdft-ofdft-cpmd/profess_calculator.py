from ase import Atoms
from ase.calculators.general import Calculator
from ase import units
import profess_io as pio
import subprocess
import os
import numpy as np
from density_calculators import DensityOptimizerCPMD

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
        
    def initialize(self, run_dir=None, inpt_name=None, pp_names=None, atoms=None, profess_path = None, pos_type = 'CART'):
        self.atoms = atoms
        self.run_dir = run_dir
        self.inpt_name = inpt_name
        self.pp_names = pp_names
        self.energy_zero = 0.0
        self.profess_path = profess_path

    def run_profess(self):
        os.chdir(self.run_dir)
        p = subprocess.run([self.profess_path, self.inpt_name], capture_output = True,  text=True )
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
        
    
class PROFESS_CPMD(PROFESS):
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
        
    def initialize(self, atoms=None, dt = None, inpt_name=None, mu = None, pos_type = 'CART',pp_names=None, run_dir=None):
        self.atoms = atoms
        self.run_dir = run_dir
        self.inpt_name = inpt_name
        self.pp_names = pp_names
        self.energy_zero = 0.0
        
        # create and initialize DensityOptimizer
        self.DensOpt = DensityOptimizerCPMD()
        self.DensOpt.initialize(self.atoms, dt, self.inpt_name, mu, self.run_dir)
    
    def get_forces(self, atoms=None):
        # write new .ion file
        cell_par = self.atoms.get_cell()
        atom_types = self.atoms.get_chemical_symbols()
        positions = self.atoms.get_positions()
        pos_type = 'CART'
        new_ion = pio.generate_ion_file(cell_par, atom_types, positions, pos_type, self.pp_names)
        pio.write_file(f'{os.path.join(self.run_dir, self.inpt_name)}.ion', new_ion)
        
        # write density? no but ensure that density file exists
        
        # optimize density for one single step
        # at this step also the forces are calculated, check that exited normally
        self.DensOpt.optimize_vv(1)
        
        # update energy
        self.energy_zero = self.DensOpt.energies[-1]
        
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
        self.completed_p = self.run_profess()
        assert self.completed_p.returncode == 0, 'Calculation failed'
        # read forces
        self.forces = np.array(pio.parse_force_file(f'{os.path.join(self.run_dir, self.inpt_name)}.force.out'))
        # ensures that programm crashes if computation of forces fails in next step
        os.remove(f'{os.path.join(self.run_dir, self.inpt_name)}.force.out')
        return(self.forces)