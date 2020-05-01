from diffiqult.Basis import def2TZVP as basis
from diffiqult.Molecule import System_mol
from diffiqult.Task import Tasks
import diffiqult.Energy
import diffiqult.Task
from algopy import UTPM
import numpy as np
np.set_printoptions(precision=20)

# set up meta data
d = 1.
mol = [(1,(0.0,0.0,0.0)),(1,(0.0,0.0,d))]
ne = 2
system = System_mol(mol,basis,ne,shifted=False)
manager = Tasks(system, name='h2_sto_3g',verbose=True)

# calculate SCF energy
#print 'EnergyH2', diffiqult.Energy.rhfenergy(*manager._energy_args())

# calculate derivatives
args = list(manager._energy_args())

D = 3; P = 1
x = UTPM(np.zeros((D, P, 2, 1), dtype=np.float64))
x.data[0, 0] = 0
x.data[1, 0] = 1

args[4] = [1+x, 1-x]
args[-1]= x
args = tuple(args)

y = diffiqult.Energy.rhfenergy(*(args))
print (dir(y))
print 'derivatives', y.data.flatten()

# calculate target energy
heargs = list(manager._energy_args())
heargs[4] = [2, 0]
print 'EnergyTarget', diffiqult.Energy.rhfenergy(*heargs)
