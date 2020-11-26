from profess_calculator import PROFESS
import profess_io as pio
from cpmd import initialize_atoms
from ase.md.verlet import VelocityVerlet
from ase import units
import os
import numpy as np
from matplotlib import pyplot as plt

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
inpt_name = 'ase_nve'
pp_names = ['si.lda.recpot']
dt = 0.1 # time interval in fs
t = 3500 # total time in fs
steps = int(t/dt)
# make empty calculator
calc = PROFESS()
# pass calculator to atoms object
atoms.calc = calc
# initialize calc object with correct parameters
atoms.calc.initialize(run_dir, inpt_name, pp_names, atoms, profess_path)

# %%pixie_debugger
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