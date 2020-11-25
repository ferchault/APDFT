#!/usr/bin/env python

#%%
import qml
import pandas as pd
import scipy.optimize as sco
import numpy as np
import sys

sys.path.append("../")
import mlmeta

import matplotlib.pyplot as plt

# %%
class Replay:
    def _connect(self, origin, dest):
        lookup = (origin, dest)
        sim = self._sims.query("identifier == @lookup").sim.values[0]

        row, col = sco.linear_sum_assignment(sim, maximize=True)
        scores = sim[row, col]

        if min(scores) < self._scorethreshold:
            center = (origin + dest) / 2
            mapping_left = self._connect(origin, center)
            mapping_right = self._connect(center, dest)
            return mapping_right[mapping_left]
        else:
            return col

    def __init__(self, filename, scorethrehold=0.9):
        # pd.read_hdf("BCCNCC-BNBCNC.h5", key="mappings")
        self._calcs = pd.read_hdf(filename, key="calcs")
        self._scorethreshold = scorethrehold
        self._sims = pd.read_hdf(filename, key="sims")
        self._maps = pd.read_hdf(filename, key="mappings")

    def connect(self):
        return self._connect(0, 1)

    def get_origin_energies(self):
        return self._calcs.query("identifier == 0.").mo_energy.values[0]

    def get_destination_energies(self):
        return self._calcs.query("identifier == 1.").mo_energy.values[0]


def test_shortcut():
    changestrs = "BNCCCC BCNCCC BCCNCC BNCBNC BNBNCC BNBCNC BNCBCN BNCNBC BNCCNB BNNBCC BNCNCB BNNCBC BNNCCB BCNNCB NBNBNB BNBNNB BNNNBB CCCCCC".split()
    for one in changestrs:
        for other in changestrs:
            try:
                r = Replay(f"{one}-{other}.h5", scorethrehold=0.99)
            except:
                print("missing", one, other)
                continue
            expected = r.connect()
            r = Replay(f"{one}-{other}.h5", scorethrehold=0.9)
            cheaper = r.connect()
            if not np.allclose(expected, cheaper):
                raise ValueError("dang!")


# test_shortcut()


def get_mapping(one, other):
    try:
        r = Replay(f"angle/{one}-{other}.h5", scorethrehold=0.9)
        rightmapping = r.connect()
    except:
        r = Replay(f"angle/{other}-{one}.h5", scorethrehold=0.9)
        rightmapping = np.argsort(r.connect())
    return rightmapping


changestrs = "BNCCCC BCNCCC BCCNCC BNCBNC BNBNCC BNBCNC BNCBCN BNCNBC BNCCNB BNNBCC BNCNCB BNNCBC BNNCCB BCNNCB NBNBNB BNBNNB BNNNBB CCCCCC".split()
xs = np.arange(len(changestrs))
np.random.shuffle(xs)
mapping = np.arange(66)
# A = get_mapping(changestrs[xs[0]], changestrs[xs[1]])
# B = get_mapping(changestrs[xs[1]], changestrs[xs[2]])
# AB = get_mapping(changestrs[xs[0]], changestrs[xs[2]])
# B[A] - AB

offset = 1
A = get_mapping(changestrs[offset], changestrs[offset + 1])
B = get_mapping(changestrs[offset + 1], changestrs[offset + 2])
AB = get_mapping(changestrs[offset], changestrs[offset + 2])
B[A], AB

# off = 1
# r = Replay(f"{changestrs[off]}-{changestrs[off+1]}.h5")
# a = r.connect()
# r = Replay(f"{changestrs[off+1]}-{changestrs[off+2]}.h5")
# b = r.connect()
# r = Replay(f"{changestrs[off]}-{changestrs[off+2]}.h5")
# c = r.connect()

# c - b[a]
# Replay(f"angle/BCNCCC-BCCNCC.h5", scorethrehold=0.9)._calcs.query("identifier==1").mo_energy.values[0]
# %%
def rank_plot(c):
    f, axs = plt.subplots(2, 1, figsize=(8, 12))
    axid, axe = axs

    # get lvals with a change
    lvals = sorted(c._calcs.identifier.unique())
    svals = []
    smaps = []
    sener = [c._calcs.query("identifier == 0").mo_energy.values[0]]
    nmos = len(c._calcs.query("identifier == 0").mo_energy.values[0])

    for origin, destination in zip(lvals[:-1], lvals[1:]):
        idval = (origin, destination)
        if len(svals) == 0:
            svals.append(origin)
        if np.allclose(
            c._maps.query("identifier == @idval").map.values[0], np.arange(nmos)
        ):
            continue
        svals.append(destination)
        smaps.append(c._maps.query("identifier == @idval").map.values[0])
        sener.append(c._calcs.query("identifier == @destination").mo_energy.values[0])
    if svals[-1] < 1:
        svals[-1] = 1

    # filter no change entries
    smaps = np.array(smaps)
    lower = np.min(smaps, axis=0)
    upper = np.max(smaps, axis=0)
    mask = lower != upper
    smaps = smaps[:, mask]
    sener = np.array(sener)[:, mask]

    # plot lvals
    xpos = np.linspace(0, 1, len(svals))
    nmos = smaps.shape[-1]
    sl = 0
    for x, sval in zip(xpos, svals):
        try:
            coloridx = list(sval % (1 / 2 ** np.arange(10))).index(0.0)
        except:
            coloridx = 10
        axid.scatter(np.zeros(nmos) + x, np.arange(nmos), color=f"C{coloridx}")
        q = sener[sl][sener[sl] < 0]
        axe.scatter(np.zeros(len(q)) + x, q, color=f"C{coloridx}")
        sl += 1

    # plot connections
    q = 0
    for idx in range(len(svals) - 1):
        for f, t in enumerate(smaps[idx]):
            t -= sum([1 for _ in mask[:t] if _ == False])
            if f == t:
                alpha = 0.7
            else:
                alpha = 1

            axid.plot(
                xpos[idx : idx + 2], (f, t), color="grey", alpha=alpha, zorder=-10
            )

            if sener[idx, f] < 0 or sener[idx + 1, t] < 0:
                axe.plot(
                    xpos[idx : idx + 2],
                    (sener[idx, f], sener[idx + 1, t]),
                    color="grey",
                    alpha=0.7,
                    zorder=10,
                )

    # label
    plt.xlabel("Mixing parameter $\lambda$ [non-linear spacing]")
    plt.xticks(xpos, svals)


Replay(f"angle/BCNCCC-BNCBNC.h5", scorethrehold=0.9)._calcs.query(
    "identifier==1"
).mo_energy.values[0] - Replay(
    f"angle/BCCNCC-BNCBNC.h5", scorethrehold=0.9
)._calcs.query(
    "identifier==1"
).mo_energy.values[
    0
]
rank_plot(Replay(f"angle/BCCNCC-BNCBNC.h5", scorethrehold=0.9))
#%%
# off = 0
# r = Replay(f"{changestrs[off]}-{changestrs[off+1]}.h5")
# r = Replay(f"{changestrs[off+1]}-{changestrs[off+2]}.h5")
def rank_plot(c):
    f = plt.figure(figsize=(8, 12))
    occupied = sum(c._calcs.query("identifier == 0").mo_occ.values[0] > 0)
    plt.axhline(occupied + 0.5, color="red")

    # get lvals with a change
    lvals = sorted(c._calcs.identifier.unique())
    svals = []
    smaps = []
    nmos = len(c._calcs.query("identifier == 0").mo_energy.values[0])
    for origin, destination in zip(lvals[:-1], lvals[1:]):
        idval = (origin, destination)
        if len(svals) == 0:
            svals.append(origin)
        if np.allclose(
            c._maps.query("identifier == @idval").map.values[0], np.arange(nmos)
        ):
            continue
        svals.append(destination)
        smaps.append(c._maps.query("identifier == @idval").map.values[0])
    if svals[-1] < 1:
        svals[-1] = 1

    # plot lvals
    xpos = np.linspace(0, 1, len(svals))
    for x, sval in zip(xpos, svals):
        try:
            coloridx = list(sval % (1 / 2 ** np.arange(10))).index(0.0)
        except:
            coloridx = 10
        plt.scatter(np.zeros(nmos) + x, np.arange(nmos), color=f"C{coloridx}")

    # plot connections
    for idx in range(len(svals) - 1):
        for f, t in enumerate(smaps[idx]):
            if f == t:
                alpha = 0.7
            else:
                alpha = 1
            plt.plot(xpos[idx : idx + 2], (f, t), color="grey", alpha=alpha, zorder=-10)

    # label
    plt.xlabel("Mixing parameter $\lambda$ [non-linear spacing]")
    plt.xticks(xpos, svals)
    plt.ylabel("MO index")


# rank_plot(r)
#%%
changestrs


#%%
changestrs = "BNCCCC BCNCCC BCCNCC BNCBNC BNBNCC BNBCNC BNCBCN BNCNBC BNCCNB BNNBCC BNCNCB BNNCBC BNNCCB BCNNCB NBNBNB BNBNNB BNNNBB CCCCCC".split()
one = changestrs[0]
Ys_original = []
Ys_resorted = []
for other in changestrs:
    if one == other:
        continue
    r = Replay(f"{one}-{other}.h5")
    mapping = r.connect()
    oener = r.get_origin_energies()
    dener = r.get_destination_energies()
    dener_sorted = dener[np.argsort(mapping)]

    if len(Ys_resorted) == 0:
        Ys_original.append(oener)
        Ys_resorted.append(oener)
    else:
        assert np.allclose(oener, Ys_original[0])
    Ys_original.append(dener)
    Ys_resorted.append(dener_sorted)
Ys_original = np.array(Ys_original)
Ys_resorted = np.array(Ys_resorted)

# %%
def get_mol(changestr):
    lines = f"""12
    
    {changestr[0]}  0.000000000000000  1.391100104090276  0.0
    {changestr[1]}  1.204728031075409  0.695550052045138 -0.0
    {changestr[2]}  1.204728031075409 -0.695550052045138 -0.0
    {changestr[3]} -0.000000000000000 -1.391100104090276  0.0
    {changestr[4]} -1.204728031075409 -0.695550052045138  0.0
    {changestr[5]} -1.204728031075409  0.695550052045138  0.0
    H  0.000000000000000  2.471100189753489  0.0
    H  2.140035536125550  1.235550092230858 -0.0
    H  2.140035536125550 -1.235550092230858 -0.0
    H -0.000000000000000 -2.471100189753489  0.0
    H -2.140035536125550 -1.235550092230858  0.0
    H -2.140035536125550  1.235550092230858  0.0"""
    return qml.Compound(xyz=mlmeta.MockXYZ(lines.split("\n")))


mols = [get_mol(_) for _ in changestrs]
# %%
idx = 30
ns, mae, stddev = mlmeta.get_KRR_learning_curve(mols, "CM", Ys_original[:, idx], k=100)
plt.errorbar(ns, mae, stddev, label="original")
ns, mae, stddev = mlmeta.get_KRR_learning_curve(mols, "CM", Ys_resorted[:, idx], k=100)
plt.errorbar(ns, mae, stddev, label="resorted")
plt.legend()
plt.yscale("log")
plt.xscale("log")

# %%
2 ** -2
# %%

off = 0
# r = Replay(f"{changestrs[off]}-{changestrs[off+1]}.h5")
r = Replay(f"{changestrs[off]}-{changestrs[off+1]}.h5")
a = r.connect()
r = Replay(f"{changestrs[off+1]}-{changestrs[off+2]}.h5")
b = r.connect()
r = Replay(f"{changestrs[off]}-{changestrs[off+2]}.h5")
c = r.connect()
# %%
b[a] - c
# %%
np.array(
    [
        1,
        0,
        3,
        2,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        42,
        41,
        43,
        45,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
    ]
) - b
# %%
r = Replay(f"B.h5.h5")
b2 = r.connect()
# %%
b2 - b
# %%
b
# %%
r._maps
# %%
mapping
# %%
