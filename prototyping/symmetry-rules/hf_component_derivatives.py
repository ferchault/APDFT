#!/usr/bin/env python
# %%
# region imports
import matplotlib.pyplot as plt
import numpy as np
import numpy
import pyscf
import pyscf.gto
import pyscf.qmmm
import pyscf.scf
import pyscf.dft
import pyscf.lib
from pyscf.data import nist
import pandas as pd
import basis_set_exchange as bse

angstrom = 1 / 0.52917721067
#endregion

# region logic
def get_components(molstring, basisset, deltaZ, includeonly, direction):
    mol = pyscf.gto.Mole()
    mol.atom = molstring
    mol.basis = basisset
    mol.verbose = 0
    mol.build()

    def add_qmmm(calc, mol, deltaZ):
        mf = pyscf.qmmm.mm_charge(
            calc, mol.atom_coords()[includeonly] / angstrom, deltaZ
        )

        def energy_nuc(self):
            q = mol.atom_charges().astype(np.float)
            q[includeonly] += deltaZ
            return mol.energy_nuc(q)

        mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)
        return mf

    calc = add_qmmm(pyscf.scf.RHF(mol), mol, deltaZ)
    hfe = calc.kernel(verbose=0)
    dm = calc.make_rdm1()
    total_energy = calc.e_tot

    hkin = mol.intor_symmetric("int1e_kin")
    # workaround to include proper potentials from point charges
    hen = calc.get_hcore() - hkin
    vhf = calc.get_veff(mol, dm)

    epn = 0
    for site in includeonly:
        mol.set_rinv_orig_(mol.atom_coords()[site])
        epn += direction[site]*np.matmul(dm, mol.intor("int1e_rinv")).trace()

    e_nn = calc.energy_nuc()
    kinetic = numpy.einsum("ij,ji->", hkin, dm)
    en = numpy.einsum("ij,ji->", hen, dm)
    hfx = numpy.einsum("ij,ji->", vhf, dm) * 0.5
    return {"total": total_energy, "nn": e_nn, "kin": kinetic, "en": en, "hfx": hfx, "epn": epn}

# H2 test case as obtained from Gaussian
# #HF/6-31G  ExtraLinks=L608
# Kinetic          0.924150
# electron-nuclei -3.117033
# Coulomb          1.137794
# HFX             -0.568897
# Nuclear-nuclear  0.529177
# Same for He
# Kinetic          2.855325
# electron-nuclei -6.737393
# Coulomb          2.053814
# HFX             -1.026907
# Nuclear          0.0

def validate():
    direction = np.array((1, -1))
    results = []
    for lval in np.linspace(0, 1, 10):
        results.append(
            get_components("H 0 0 0; H 0 0 1", "6-31G", direction * lval, [0, 1])
        )
    results = pd.DataFrame(results)

    import matplotlib.pyplot as plt

    for col in results.columns:
        plt.plot(results[col], label=col)
    plt.legend()
    q = get_components(
        "He 0 0 0", bse.get_basis("6-31G", 1, fmt="nwchem"), np.array([0]), [0]
    )
    plt.axhline(q["hfx"])
# endregion

#%%
# region Application alchemical enantiomers
molstring = """C     -0.00000000     1.24690500     1.40549300;
C     -0.00000000     2.43902000     0.70923900;
C      0.00000000     2.43902000    -0.70923900;
C      0.00000000     1.24690500    -1.40549300;
C     -0.00000000    -1.24690500    -1.40549300;
C     -0.00000000    -2.43902000    -0.70923900;
C      0.00000000    -2.43902000     0.70923900;
C     -0.00000000    -1.24690500     1.40549300;
C     -0.00000000     0.00000000     0.71868000;
C     -0.00000000    -0.00000000    -0.71868000;
H     -0.00000000     1.24502900     2.49891700;
H     -0.00000000     3.38894400     1.24943500;
H      0.00000000     3.38894400    -1.24943500;
H      0.00000000     1.24502900    -2.49891700;
H     -0.00000000    -1.24502900    -2.49891700;
H     -0.00000000    -3.38894400    -1.24943500;
H      0.00000000    -3.38894400     1.24943500;
H      0.00000000    -1.24502900     2.49891700"""
direction = np.array([float(_) for _ in str(5775577755)])-6
direction = np.array([float(_) for _ in str(5575757757)])-6
results = []
for lval in np.linspace(-1, 1, 21):
    q = get_components(molstring, "6-31G", direction * lval, range(10), direction)
    q["lval"] = lval
    results.append(q)
results = pd.DataFrame(results)


#%%
centervals = {}
for col in results.columns:
    if col == "lval" or col=="epn":
        continue
    centervals[col] = results.query("lval == 0")[col].values[0]
    label = col
    if label == "hfx":
        label = "ee"
    plt.plot(results.lval, results[col].values-centervals[col], label=label)
plt.plot(results.lval, results.hfx + results.en+results.kin - centervals["hfx"]-centervals["en"]-centervals["kin"], label="ele")

# energy highlight
endval = results.query("lval == 1")["total"].values[0]
centerval = results.query("lval == 0")["total"].values[0]
plt.axhline(endval - centerval, alpha=0.2, color="C0")

# APDFT1 derivative
d1 = results.query("lval > 0.05 & lval < 0.15")["epn"].values[0] - results.query("lval < -0.05 & lval > -0.15")["epn"].values[0]
d1 /= 0.2
print (d1)
q = 0.5
#plt.plot((-q, q), (-q*d1, q*d1), "--", color="C3", label="APDFT1")

# reference energies for 5775577755
# margin = 0.02
# plt.axhline(448.225345- 456.096471, xmin=1-margin, color="grey", label="g09 comparison")
# plt.axhline(462.691890- 456.096471, xmax=margin, color="grey")
# plt.axhline(-1799.781123--1804.338320, xmin=1-margin, color="grey")
# plt.axhline(-1825.901456--1804.338320, xmax=margin, color="grey")
# plt.axhline(378.176841-383.126374, xmin=1-margin, color="grey")
# plt.axhline(378.115117-383.126374, xmax=margin, color="grey")
# eectr = 636.465682+-54.568927
# ccup = 637.054016  -54.657130
# ccdn = 648.740943  -54.610649
# plt.axhline(ccup-eectr, xmin=1-margin, color="grey")
# plt.axhline(ccdn-eectr, xmax=margin, color="grey")

# ET=  383.126374 EV=-1804.338320 EJ=  636.465682 EK=  -54.568927 ENuc=  456.096471 center
# ET=  378.176841 EV=-1799.781123 EJ=  637.054016 EK=  -54.657130 ENuc=  448.225345 up
# ET=  378.115117 EV=-1825.901456 EJ=  648.740943 EK=  -54.610649 ENuc=  462.691890 dn

plt.ylabel("E-E(naphthalene) [Ha]")
plt.xlabel("$\lambda$")
plt.legend()
# endregion
# %%
direction = np.array([float(_) for _ in str(5575757757)])-6
charges = np.array([6]*10) + direction
enn = 0
for i in range(10):
    for j in range(10)
        enn += 
# %%
results
# %%
direction
# %%
