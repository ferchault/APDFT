from profess_calculator import PROFESS_old
import profess_io as pio
from cpmd import initialize_atoms
from ase.md.verlet import VelocityVerlet
from ase import units
import os
# import pixiedust
import numpy as np

# initialize
pos_file = '/home/misa/git_repositories/PROFESS/test/MD_NVE/saved/ion_step0.dat'
vel_file = '/home/misa/git_repositories/PROFESS/test/MD_NVE/saved/vel_step0.dat'
cell_param = [[3.97, 0, 0], [0, 3.97, 0], [0, 0, 3.97]]
atoms = initialize_atoms(pos_file, vel_file, cell_param, pos_type='FRAC')

# set parameters for PROFESS calculation
run_dir = '/home/misa/git_repositories/PROFESS/test/ase_singlepoint'
inpt_name = 'sp'
pp_names = ['al_HC.lda.recpot']

save_pos = [atoms.get_positions()]
save_vel = [atoms.get_velocities()]
save_forces = []
total_energy = []
e_kin = []
for n in range(1, 515):
    # kinetic energy at step n-1
    e_kin_step = atoms.get_kinetic_energy()
    e_kin.append(e_kin_step)
    atoms.set_calculator(PROFESS_old(run_dir, inpt_name, pp_names, atoms))
    # We want to run MD with constant energy using the VelocityVerlet algorithm.
    dyn = VelocityVerlet(atoms, 1*units.fs)  # 1 fs time step.
    # run only for one step
    f=dyn.run(1)
    
    save_pos.append(atoms.get_positions())
    save_vel.append(atoms.get_velocities())
    # forces and energies belong to step n-1
    e_tot_step = pio.parse_out_file(os.path.join(run_dir, inpt_name+'.out'), 'TOTAL ENERGY')    
    total_energy.append(e_kin_step+e_tot_step)
    save_forces.append(atoms._calc.forces)
total_energy = np.array(total_energy)
e_kin = np.array(e_kin)

# save data
path_etot = '/home/misa/projects/APDFT-CPMD/data/nuclei_md/Al4_NVE_ase_etot_run1step'
path_ekin= '/home/misa/projects/APDFT-CPMD/data/nuclei_md/Al4_NVE_ase_ekin_run1step'
np.savetxt(path_etot, total_energy)
np.savetxt(path_ekin, e_kin)