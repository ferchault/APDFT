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
row = 'mixed_2_3'
system = 'diatomics'
bond_type = 'single'
spin = 0

atoms = aio.read(f'/data/sahre/projects/atomic-energies/data/bonding_trends/pbe0_data/{row}/{system}_{bond_type}/{c}.xyz')

pyscf_atoms = uqm.ase2pyscf(atoms)
basis = 'def2-tzvp'
mol = gto.Mole()
mol.atom = pyscf_atoms
mol.spin = spin
mol.basis = basis
mol.build()

if spin != 0:
    mf = dft.ROKS(mol)
else:
    mf = dft.RKS(mol)
mf.xc = 'pbe0'

if system == 'fragments':
    mf.kernel()
    with open(f'/data/sahre/projects/atomic-energies/data/bonding_trends/pbe0_data/{row}/{system}_{bond_type}/{c}_energy', 'w') as f:
        f.write(str(mf.e_tot)+'\n')

mol_eq = optimize(mf, maxsteps=100)

new_pos = mol_eq.atom_coords()*Bohr
atoms.set_positions(new_pos)

pyscf_atoms = uqm.ase2pyscf(atoms)
basis = 'def2-tzvp'
mol = gto.Mole()
mol.atom = pyscf_atoms
mol.basis = basis
mol.spin = spin
mol.build()

if spin != 0:
    mf = dft.ROKS(mol)
else:
    mf = dft.RKS(mol)
mf.xc = 'pbe0'
mf.kernel()


aio.write(f'/data/sahre/projects/atomic-energies/data/bonding_trends/pbe0_data/{row}/{system}_{bond_type}/{c}_opt.xyz', atoms)

if system == 'fragments':
    with open(f'/data/sahre/projects/atomic-energies/data/bonding_trends/pbe0_data/{row}/{system}_{bond_type}/{c}_energy_relaxed', 'w') as f:
        f.write(str(mf.e_tot)+'\n')
else:
    with open(f'/data/sahre/projects/atomic-energies/data/bonding_trends/pbe0_data/{row}/{system}_{bond_type}/{c}_energy', 'w') as f:
        f.write(str(mf.e_tot)+'\n')