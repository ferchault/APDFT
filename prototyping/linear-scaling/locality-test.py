#%%
from pyscf import gto, dft
mol = gto.M(atom='H  0  0  0; F  0.9  0  0', basis='6-31G')
mf = dft.RKS(mol)
mf.xc = 'pbe,pbe'

mf.kernel()
# %%
mf.mo_coeff
# %%
mf.mo_occ
# %%
import pyscf.tools
pyscf.tools.dump_mat.dump_mo(mol, mf.mo_coeff)
# %%

# %%
import numpy as np
coeff = mf.mo_coeff[:,0]
ao = mol.eval_gto('GTOval', [[0,0,0]])

np.dot(ao, coeff)
# %%
import sys
sys.path.append("../../src")
import apdft
from apdft.calculator.pyscf import PyscfCalculator
import numpy as np
# %%
calculator = PyscfCalculator("HF", "def2-TZVP")
nuclear_numbers, coordinates = apdft.read_xyz("undecane/inp.xyz")
derivatives = apdft.physics.APDFT(2, nuclear_numbers, coordinates, "undecane", calculator, 0, 5, "C", None)
# %%
# assert one order of targets
def _get_stencil_coefficients(self, deltaZ, shift, cutoff):
    # build alphas
    N = len(self._include_atoms)
    nvals = {0: 1, 1: N * 2, 2: N * (N - 1)}
    alphas = np.zeros((sum([nvals[_] for _ in self._orders]), len(self._orders)))

    # test input
    if N != len(deltaZ):
        raise ValueError(
            "Mismatch of array lengths: %d dZ values for %d nuclei."
            % (len(deltaZ), N)
        )

    # order 0
    if 0 in self._orders:
        alphas[0, 0] = 1

    # order 1
    if 1 in self._orders:
        prefactor = 1 / (2 * self._delta) / np.math.factorial(1 + shift)
        for siteidx in range(N):
            alphas[1 + siteidx * 2, 1] += prefactor * deltaZ[siteidx]
            alphas[1 + siteidx * 2 + 1, 1] -= prefactor * deltaZ[siteidx]

    # order 2
    if 2 in self._orders:
        pos = 1 + N * 2 - 2
        for siteidx_i in range(N):
            for siteidx_j in range(siteidx_i, N):
                if siteidx_i != siteidx_j:
                    pos += 2
                if deltaZ[siteidx_j] == 0 or deltaZ[siteidx_i] == 0:
                    continue
                if abs(siteidx_i  - siteidx_j) > cutoff:
                    continue
                if self._include_atoms[siteidx_j] > self._include_atoms[siteidx_i]:
                    prefactor = (1 / (2 * self._delta ** 2)) / np.math.factorial(
                        2 + shift
                    )
                    prefactor *= deltaZ[siteidx_i] * deltaZ[siteidx_j]
                    alphas[pos, 2] += prefactor
                    alphas[pos + 1, 2] += prefactor
                    alphas[0, 2] += 2 * prefactor
                    alphas[1 + siteidx_i * 2, 2] -= prefactor
                    alphas[1 + siteidx_i * 2 + 1, 2] -= prefactor
                    alphas[1 + siteidx_j * 2, 2] -= prefactor
                    alphas[1 + siteidx_j * 2 + 1, 2] -= prefactor
                if self._include_atoms[siteidx_j] == self._include_atoms[siteidx_i]:
                    prefactor = (1 / (self._delta ** 2)) / np.math.factorial(
                        2 + shift
                    )
                    prefactor *= deltaZ[siteidx_i] * deltaZ[siteidx_j]
                    alphas[0, 2] -= 2 * prefactor
                    alphas[1 + siteidx_i * 2, 2] += prefactor
                    alphas[1 + siteidx_j * 2 + 1, 2] += prefactor

    return alphas

def get_energies(cutoff):
    self = derivatives
    targets = self.enumerate_all_targets()
    own_nuc_nuc = apdft.physics.Coulomb.nuclei_nuclei(self._coordinates, self._nuclear_numbers)

    energies = np.zeros((len(targets), len(self._orders)))
    dipoles = np.zeros((len(targets), 3, len(self._orders)))

    # get base information
    refenergy = self.get_energy_from_reference(
        self._nuclear_numbers, is_reference_molecule=True
    )
    epn_matrix = self.get_epn_matrix()
    dipole_matrix = self.get_linear_density_matrix("ELECTRONIC_DIPOLE")

    # get target predictions
    rows = []
    for targetidx, target in enumerate(targets):
        deltaZ = target - self._nuclear_numbers

        deltaZ_included = deltaZ[self._include_atoms]
        alphas = _get_stencil_coefficients(self, deltaZ_included, 1, cutoff)

        # energies
        deltaEnn = apdft.physics.Coulomb.nuclei_nuclei(self._coordinates, target) - own_nuc_nuc
        per_file_contributions = []
        for order in sorted(self._orders):
            contributions = -np.multiply(
                np.outer(alphas[:, order], deltaZ_included), epn_matrix
            )
            per_file_contributions.append(contributions.sum(axis=1))
            contributions = contributions.sum()
            energies[targetidx, order] = contributions
            if order > 0:
                energies[targetidx, order] += energies[targetidx, order - 1]
        energies[targetidx, :] += deltaEnn + refenergy
        rows.append(np.array(per_file_contributions).sum(axis=0))
    return energies

pred_orig = get_energies(100)

# %%
rows = np.array(rows)
import matplotlib.pyplot as plt
for col in range(2, 5):
    plt.hist(rows[:, col], histtype="step")


# %%
energies = {}
for cutoff in range(12):
    energies[cutoff] = get_energies(cutoff)
# %%
ys = []
for cutoff in range(12):
    #plt.hist(np.abs(pred_orig - energies[cutoff]), label=cutoff, histtype="step", range=(0, 0.01))
    ys.append(np.abs(pred_orig - energies[cutoff]).mean())
plt.semilogy(ys)
plt.legend()

