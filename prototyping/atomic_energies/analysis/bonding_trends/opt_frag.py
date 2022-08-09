import ase.io as aio
from ase.units import Bohr
import sys
sys.path.insert(0, '/home/sahre/git_repositories/APDFT/prototyping/atomic_energies/')
import utils_qm as uqm

from pyscf import gto, dft, scf

# geometric
from pyscf.geomopt.geometric_solver import optimize

# comps = ['CC','CN','CO','CF','NN','NO','NF','OO', 'OF', 'FF']
c = sys.argv[1]
row = 'row_4'
system = 'fragments'
bond_type = 'single'
spin = 1

atoms = aio.read(f'/data/sahre/projects/atomic-energies/data/bonding_trends/pbe0_data/{row}/{system}_{bond_type}/{c}.xyz')

pyscf_atoms = uqm.ase2pyscf(atoms)
basis = 'def2-tzvp'
mol = gto.Mole()
mol.atom = pyscf_atoms
mol.basis = basis
mol.spin = spin
mol.build()

# energy of unrelaxed fragment
mf = dft.ROKS(mol)
mf.xc = 'pbe0'
# if c == 'OO.xyz':
#     mf.level_shift = 0.2
mf.kernel()
with open(f'/data/sahre/projects/atomic-energies/data/bonding_trends/pbe0_data/{row}/{system}_{bond_type}/{c}_energy', 'w') as f:
    f.write(str(mf.e_tot)+'\n')
    
# optimization of fragment
mol_eq = optimize(mf, maxsteps=300)
new_pos = mol_eq.atom_coords()*Bohr
atoms.set_positions(new_pos)
aio.write(f'/data/sahre/projects/atomic-energies/data/bonding_trends/pbe0_data/{row}/{system}_{bond_type}/{c}_opt.xyz', atoms)

# single point of optimized structure
pyscf_atoms = uqm.ase2pyscf(atoms)
basis = 'def2-tzvp'
mol = gto.Mole()
mol.atom = pyscf_atoms
mol.basis = basis
mol.spin = spin
mol.build()

mf = dft.ROKS(mol)
mf.xc = 'pbe0'
# if c == 'OO.xyz':
#     mf.level_shift = 0.2
mf.kernel()

with open(f'/data/sahre/projects/atomic-energies/data/bonding_trends/pbe0_data/{row}/{system}_{bond_type}/{c}_energy_relaxed', 'w') as f:
    f.write(str(mf.e_tot)+'\n')