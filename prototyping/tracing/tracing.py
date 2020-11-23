#!/usr/bin/env python
"""Traces MOs across alchemical perturbations."""

# region
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco

# endregion


# region
class FollowMe:
    def __init__(self, filename):
        self._calcs = pd.read_hdf(filename, key="calcs")
        self._sims = pd.read_hdf(filename, key="sims")
        self._maps = pd.read_hdf(filename, key="mappings")
        self._clean()
        self._annotate()
        self._connect()

    def _clean(self):
        self._lvals = np.array(sorted(self._calcs.identifier.values))
        self._nmos = len(self._calcs.mo_energy.values[0])

    def _connect(self):
        labels = np.array([f"MO-{_}" for _ in range(self._nmos)])
        positions = []
        for idx, destination in enumerate(self.lvals):
            if idx == 0:
                ranking = labels
            else:
                origin = self.lvals[idx - 1]
                identifier = (origin, destination)
                sim = self._sims.query("identifier == @identifier").sim.values[0]

                # added energy window
                for i in range(self._nmos):
                    for j in range(self._nmos):
                        dE = abs(self._energies[idx - 1, i] - self._energies[idx, j])
                        if dE > 1 / 27.2114:
                            sim[i, j] = 0
                row, col = sco.linear_sum_assignment(sim, maximize=True)
                ranking = positions[-1][np.argsort(col)]
            positions.append(ranking)
        self._positions = np.array(positions)

    def _annotate(self):
        self._energies = np.zeros((len(self.lvals), self._nmos))
        for idx, lval in enumerate(self._lvals):
            self._energies[idx, :] = self._calcs.query(
                "identifier == @lval"
            ).mo_energy.values[0]

    @property
    def lvals(self):
        return self._lvals


# BCNCCC BCCNCC BNCBNC
# A = FollowMe("0.99-energy-fixed/BCNCCC-BCCNCC.h5")
# B = FollowMe("0.99-energy-fixed/BCCNCC-BNCBNC.h5")
# AB = FollowMe("0.99-energy-fixed/BCNCCC-BNCBNC.h5")
# BCCNCC BNCBNC BNBNCC
# A = FollowMe("0.99-energy-fixed/BCCNCC-BNCBNC.h5")
# B = FollowMe("0.99-energy-fixed/BNCBNC-BNBNCC.h5")
# AB = FollowMe("0.99-energy-fixed/BCCNCC-BNBNCC.h5")

A = FollowMe("fixed2/BNCBCN-BNCNBC.h5")
B = FollowMe("fixed2/BNCNBC-BNCCNB.h5")
AB = FollowMe("fixed2/BNCBCN-BNCCNB.h5")

# endregion


# region
def compareplot(A, B, AB):
    f, axs = plt.subplots(4, 1, figsize=(12, 12))
    consR, consE, ABR, ABE = axs

    positions = np.hstack((A._positions.T, B._positions.T)).T.copy()
    # adjust labels in B
    lookup = dict(zip(B._positions[0], A._positions[-1]))
    positions[A._positions.shape[0] :, :] = np.vectorize(lookup.__getitem__)(
        positions[A._positions.shape[0] :, :]
    )
    energies = np.hstack((A._energies.T, B._energies.T)).T

    # A + B
    consE.axvline(A._positions.shape[0], color="grey")
    consR.axvline(A._positions.shape[0], color="grey")
    for label in positions[0]:
        ranks = np.where(positions == label)[1]
        # if min(ranks) != max(ranks):
        consE.plot(energies[np.arange(len(ranks)), ranks])
        consR.plot(ranks)
        if label == AB._positions[-1][ranks[-1]]:
            color = "green"
        else:
            color = "red"
            print(label, AB._positions[-1][ranks[-1]])
        consR.scatter(len(ranks) + 1, ranks[-1], color=color, s=2)
    consE.set_ylim(-16, -6)

    # AB
    for label in positions[0]:
        ranks = np.where(AB._positions == label)[1]
        # if min(ranks) != max(ranks):
        ABR.plot(ranks)
        ABE.plot(AB._energies[np.arange(len(ranks)), ranks])

    ABE.set_ylim(-16, -6)
    # ABE.set_xlim(110, 150)
    # ABE.axvline(180)
    # ABE.axvline(170)


compareplot(A, B, AB)

# region
# if degenerate and overlap pairwise identnical, no tracking possible, results in random crossing
# visualise orbitals
# avoid paths through molecules with degenerate and maximally operlapping MO
def get_mismatches(A, B, AB):
    positions = np.hstack((A._positions.T, B._positions.T)).T.copy()
    # adjust labels in B
    lookup = dict(zip(B._positions[0], A._positions[-1]))
    positions[A._positions.shape[0] :, :] = np.vectorize(lookup.__getitem__)(
        positions[A._positions.shape[0] :, :]
    )
    energies = np.hstack((A._energies.T, B._energies.T)).T

    # A + B
    mismatch = []
    for label in positions[0]:
        ranks = np.where(positions == label)[1]
        if label != AB._positions[-1][ranks[-1]]:
            mismatch.append(label)
            # print(label, AB._positions[-1][ranks[-1]])
    return mismatch


A = FollowMe("0.99-energy-fixed/BNCBCN-BNCNBC.h5")
B = FollowMe("0.99-energy-fixed/BNCNBC-BNCCNB.h5")
AB = FollowMe("0.99-energy-fixed/BNCBCN-BNCCNB.h5")
get_mismatches(A, B, AB)
# region
changestrs = "BNCCCC BCNCCC BCCNCC BNCBNC BNBNCC BNBCNC BNCBCN BNCNBC BNCCNB BNNBCC BNCNCB BNNCBC BNNCCB BCNNCB NBNBNB BNBNNB BNNNBB CCCCCC".split()
for i in range(len(changestrs)):
    for j in range(i + 1, len(changestrs)):
        try:
            A = FollowMe(f"0.99-energy-fixed/{changestrs[i]}-{changestrs[j]}.h5")
        except:
            continue
        for k in range(j + 1, len(changestrs)):
            try:
                B = FollowMe(f"0.99-energy-fixed/{changestrs[j]}-{changestrs[k]}.h5")
                AB = FollowMe(f"0.99-energy-fixed/{changestrs[i]}-{changestrs[k]}.h5")
            except:
                continue
            print(i, j, k, get_mismatches(A, B, AB))
# region
changestrs = "BNCCCC BCNCCC BCCNCC BNCBNC BNBNCC BNBCNC BNCBCN BNCNBC BNCCNB BNNBCC BNCNCB BNNCBC BNNCCB BCNNCB NBNBNB BNBNNB BNNNBB CCCCCC".split()
origin = changestrs[0]
rows = []

for destination in changestrs[1:]:
    M = FollowMe(f"0.99-energy-fixed/{origin}-{destination}.h5")
    if len(rows) == 0:
        for label, energy in zip(M._positions[-1], M._energies[-1]):
            rows.append(
                {
                    "mol": origin,
                    "label": label,
                    "energy": energy,
                    "ordering": "reordered",
                }
            )
            rows.append(
                {"mol": origin, "label": label, "energy": energy, "ordering": "none"}
            )
    for idx, energy in enumerate(M._energies[-1]):
        rows.append(
            {
                "mol": destination,
                "label": f"MO-{idx}",
                "energy": energy,
                "ordering": "none",
            }
        )
    for label, energy in zip(M._positions[-1], M._energies[-1]):
        rows.append(
            {
                "mol": destination,
                "label": label,
                "energy": energy,
                "ordering": "reordered",
            }
        )

# region
df = pd.DataFrame(rows)
# region
