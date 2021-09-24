#%%
import numpy as np
import pyscf
import pyscf.gto
import pyscf.qmmm
import pyscf.scf
import pyscf.dft
import pyscf.lib
import pyscf.mp
from pyscf.data import nist
import matplotlib.pyplot as plt
import findiff
from pyscf.scf.hf import energy_elec
#%%
mol = pyscf.gto.Mole()
mol.atom = "C 0 0 0"
mol.basis = "def2-TZVP"
mol.verbose = 0
mol.multiplicity = 3
mol.build()

def add_qmmm(calc, mol, deltaZ):
    mf = pyscf.qmmm.mm_charge(calc, np.zeros(3)[:, np.newaxis].T, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(float)
        q += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf

def get_energy(lval):
    calc = add_qmmm(pyscf.scf.UHF(mol), mol, [lval])
    hfe = calc.kernel(verbose=0)
    #return pyscf.mp.MP2(calc).kernel()[0] + hfe
    mycc = pyscf.cc.UCCSD(calc).run()
    return mycc.e_tot
    #total_energy = mycc.e_tot
    #Enn = calc.energy_nuc()
    return hfe

#xs = np.linspace(-2, 2, 20)
#es = [get_energy(_) for _ in xs]
# %%
plt.plot(xs, es)
# %%
def get_stencil():
    stencil = [{'coefficients': np.array([1]), "offsets": np.array([0])}]
    for order in range(1, 4):
        stencil.append(findiff.coefficients(deriv=order, acc=2)['center'])
    return stencil
s = get_stencil()

# %%
positions = list(set().union(*[set(_["offsets"]) for _ in s]))
delta = 0.001
fd_energies = [get_energy(_*delta) for _ in positions]
# %%
coeffs = []
for order, stencil in enumerate(s):
    contribution = 0
    for o, c in zip(stencil["offsets"], stencil["coefficients"]):
        contribution += fd_energies[positions.index(o)] * c
    contribution /= delta ** order
    contribution /= np.math.factorial(order)
    coeffs.append(contribution)

# %%
for n in range(1, 10):
    plt.plot(xs, np.polyval(coeffs[:n][::-1], xs)-es, label=f"APDFT{n-1}")
plt.legend()
plt.ylabel("Error [Ha]")
plt.xlabel("lambda")
plt.ylim(-1, 1)
#%%
s
# %%
