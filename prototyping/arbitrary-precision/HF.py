"""
    Copyright (C) 2015 Rocco Meli

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import mpmath

mpmath.mp.dps = 50

from RHF import *
from matrices import *
from integrals import *
from basis import *
from molecules import *

###########################
###########################
###########################

mol = H2  # Molecule
bs = sto3g_H2  # Basis set
N = 2  # Number of electrons

maxiter = 100  # Maximal number of iteration

verbose = False  # Print each SCF step

###########################
###########################
###########################

# Basis set size
K = bs.K

# print("Computing overlap matrix S...")
S = S_overlap(bs)

if verbose:
    print(S)

# print("Computing orthogonalization matrix X...")
X = X_transform(S)

if verbose:
    print(X)

# print("Computing core Hamiltonian...")
Hc = H_core(bs, mol)

if verbose:
    print(Hc)

# print("Computing two-electron integrals...")
ee = EE_list(bs)

if verbose:
    print_EE_list(bs, ee)

Pnew = mpmath.matrix(K, K)
P = mpmath.matrix(K, K)

converged = False

iter = 1
while not converged:
    # print("\n\n\n#####\nSCF cycle " + str(iter) + ":")
    # print("#####")

    Pnew, F, E = RHF_step(bs, mol, N, Hc, X, P, ee, verbose)  # Perform an SCF step

    # Print results of the SCF step
    e = energy_tot(P, F, Hc, mol)
    # print("   Orbital energies:")
    # print("   ", np.diag(E))

    # Check convergence of the SCF cycle
    dp = delta_P(P, Pnew)
    print(f"{iter:>5} {mpmath.nstr(dp, 5, strip_zeros=False):10} {e}")
    if dp < mpmath.mpf(f"1e-{mpmath.mp.dps-5}"):
        converged = True

        print(
            "\n\n\nTOTAL ENERGY:", energy_tot(P, F, Hc, mol)
        )  # Print final, total energy

    P = Pnew

    iter += 1
