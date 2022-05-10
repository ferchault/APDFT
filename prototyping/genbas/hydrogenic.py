#!/usr/bin/env python
#%%
import pyscf.gto
import pyscf.scf
import functools
import pyscf.dft
import pyscf.qmmm
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco


def add_qmmm(calc, mol, deltaZ):
    angstrom = 1 / 0.52917721067
    mf = pyscf.qmmm.mm_charge(calc, mol.atom_coords() / angstrom, deltaZ)

    def energy_nuc(self):
        q = mol.atom_charges().astype(np.float)
        q += deltaZ
        return mol.energy_nuc(q)

    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)

    return mf


@functools.lru_cache()
def get_ref_energy(basis):
    mol = pyscf.gto.Mole()
    mol.atom = "H 0 0 0"
    mol.basis = basis
    mol.verbose = 0
    mol.spin = 1
    mol.build()

    calc = pyscf.scf.RHF(mol)
    nprimitives = sum([(2 * _[1] + 1) * _[2] for _ in mol._bas])
    return nprimitives, calc.kernel()


def get_energy(alphas):
    basis = [[0, [_, 1.0]] for _ in alphas]

    mol = pyscf.gto.Mole()
    mol.atom = "X1 0 0 0"
    mol.basis = {"X1": basis}
    mol.verbose = 0
    mol.spin = 1
    mol.nelectron = 1
    mol.build()

    calc = add_qmmm(pyscf.scf.RHF(mol), mol, np.array([1]))
    try:
        numerical_energy = calc.kernel()
    except:
        return 0, None, None
    return numerical_energy, calc, mol


# cases = [[0.28294231], [1.33249899, 0.20152957], [4.49949138, 0.68125934, 0.15137408], [13.00544413,  1.96166454,  0.44447106,  0.12194243], [34.6069355  , 5.17280704 , 1.17149114  ,0.32836776 , 0.10326863], [100.00961518 , 34.49255866 ,  5.59132154  , 1.23115003  , 0.33855109, 0.10507541]]
# cases = [[100,34.6069355  , 5.17280704 , 1.17149114  ,0.32836776 , 0.10326863]]
# xs = np.linspace(0, 5, 500)
# grid = np.vstack((xs, xs * 0, xs * 0)).T
# Z = 1
# plt.plot(xs, Z ** 3 * np.exp(-2 * Z * xs) / np.pi, label="exact")
# for x0 in cases:
#     result = sco.minimize(lambda _: get_energy(_)[0], x0)
#     print("emin", result.fun)
#     print("solution", result.x)
#     _, calc, mol = get_energy(result.x)
#     dm = calc.make_rdm1()
#     ao_value = pyscf.dft.numint.eval_ao(mol, grid, deriv=0)
#     rho = pyscf.dft.numint.eval_rho(mol, ao_value, dm[0], xctype="LDA")
#     plt.semilogy(xs, rho, label=f"{len(x0)}GTO")

# plt.legend()
# plt.ylim(1e-6, 1)
#%%
cases = [
    [0.28294231],
    [1.33249899, 0.20152957],
    [4.49949138, 0.68125934, 0.15137408],
    [13.00544413, 1.96166454, 0.44447106, 0.12194243],
    [34.6069355, 5.17280704, 1.17149114, 0.32836776, 0.10326863],
    [
        2.05891855e02,
        9.64794977e-02,
        9.66391451e-01,
        2.00511224e01,
        2.91368472e-01,
        3.79660658e00,
    ],
    [
        3.10287922e02,
        2.25387468e-01,
        3.49531523e01,
        7.60750815e00,
        8.28393278e-02,
        2.07817331e00,
        6.50318797e-01,
    ],
]
es = [get_energy(case)[0] for case in cases]
plt.loglog(range(1, len(es) + 1), np.array(es) + 0.5)
for (
    refbasis
) in "STO-3G STO-6G 6-31G def2-SVP def2-TZVP cc-pVDZ cc-pVTZ cc-pVQZ cc-pV5Z".split():
    x, y = get_ref_energy(refbasis)
    plt.scatter((x,), (y + 0.5,), color="C0")
plt.xlabel("# primitives")
plt.ylabel("Energy error [Ha]")
# print(numerical_energy)
#%%
# %%
# from sympy.physics.hydrogen import Psi_nlm
# from sympy import Symbol, conjugate
# r=Symbol("r", positive=True)
# phi=Symbol("phi", positive=True)
# theta=Symbol("theta", positive=True)
# Z=Symbol("Z", positive=True, integer=True, nonzero=True)
# wf = Psi_nlm(1,0,0,r,phi,theta,Z)
# from sympy import integrate, conjugate, pi, oo, sin
# abs_sqrd=wf*conjugate(wf)
# %%
dm
# %%
get_ref_energy("STO-3G")
# %%
-0.5
# %%
for cidx, case in enumerate(cases):
    plt.plot(case, [cidx] * len(case), "o-")
# %%
sco.differential_evolution(
    lambda _: get_energy(np.array(_) ** 2)[0], bounds=[(0.01, 30)] * 7, disp=True
)
# %%
np.array(
    [17.61499139, 0.4747499, 5.9121191, 2.75817116, 0.28781822, 1.44158708, 0.80642346]
) ** 2
# %%
