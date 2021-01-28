import sys
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/apdft-ofdft-cpmd')

from profess_calculator import PROFESS
import profess_io as pio
from cpmd import initialize_atoms
from ase.md.verlet import VelocityVerlet
from ase import units
import os
import numpy as np

from ase import Atoms
from ase.io.trajectory import Trajectory

# initialize atoms object from PROFESS files
pos_file = '/home/misa/projects/APDFT-CPMD/data/BOMD/ase_NVE/Si8_hc/ion.restart_file'
vel_file = '/home/misa/projects/APDFT-CPMD/data/BOMD/ase_NVE/Si8_hc/vel.restart_file'
atoms = initialize_atoms(pos_file, vel_file, pos_type='CART')

# set parameters for PROFESS calculation
profess_path = '/home/misa/opt/PROFESS/PROFESS'
log = '/home/misa/projects/APDFT-CPMD/data/BOMD/ase_NVE/Si8_hc/Si8_nve.log'
traj_path = '/home/misa/projects/APDFT-CPMD/data/BOMD/ase_NVE/Si8_hc/Si8_nve.traj'
run_dir = '/home/misa/projects/APDFT-CPMD/data/BOMD/ase_NVE/Si8_hc'
ini_den = '/home/misa/projects/APDFT-CPMD/data/BOMD/ase_NVE/Si8_hc/initial_density.den'
ini_ion = '/home/misa/projects/APDFT-CPMD/data/BOMD/ase_NVE/Si8_hc/initial_ions.ion'
inpt_name = 'ase_nve'
pp_names = ['si.lda.recpot']
dt = 0.1 # time interval in fs
t = 50 # total time in fs
steps = int(t/dt)
# make empty calculator
calc = PROFESS()
# pass calculator to atoms object
atoms.calc = calc
# initialize calc object with correct parameters
atoms.calc.initialize(run_dir, ini_den, ini_ion, inpt_name, pp_names, atoms, profess_path)

# remove old logfile
try:
    os.remove(log)
except FileNotFoundError:
    print('Already deleted')

dyn = VelocityVerlet(atoms, dt*units.fs, logfile=log)
traj = Trajectory(traj_path, 'w', dyn.atoms)
dyn.attach(traj.write, interval=10)

dyn.run(steps)
traj.close()