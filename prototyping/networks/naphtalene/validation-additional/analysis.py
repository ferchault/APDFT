#!/usr/bin/env python

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.stats

# %%
def read_reference_energies():
    folders = glob.glob("../validation-molpro/*/")
    res = []
    for folder in folders:
        this = {}

        basename = folder.split("/")[-2]
        this["label"] = basename.split("-")[-1]
        this["nbn"] = int(basename.split("-")[1])

        try:
            with open(folder + "direct.out") as fh:
                lines = fh.readlines()
            this["energy"] = float(lines[-6].strip().split()[-1])
            this["nn"] = float(
                [_ for _ in lines if "Nuclear energy" in _][0].strip().split()[-1]
            )
        except:
            with open(folder + "run.log") as fh:
                lines = fh.readlines()
            this["energy"] = float(lines[-7].strip().split()[-1])
            this["nn"] = float(
                [_ for _ in lines if "Nuclear repulsion energy" in _][0]
                .strip()
                .split()[-1]
            )

        res.append(this)
    return pd.DataFrame(res)


df = read_reference_energies()

# %%
def read_other():
    blackout = {"b3lyp": [], "pbe": [], "pbe0": [], "xtb": []}
    for method in blackout.keys():
        with open(f"missing_{method}") as fh:
            lines = fh.readlines()
        blackout[method] = [_.strip("/").split("-")[-1] for _ in lines]

    with open("RESULTS") as fh:
        lines = fh.readlines()

    res = []
    for line in lines:
        parts = line.strip().split()
        label = parts[0].split("/")[0].split("-")[-1]
        method = parts[0].split("/")[1].split(".")[0][3:]

        if label in blackout[method.lower()]:
            continue  # did not converge

        if method == "xtb":
            energy = float(parts[-3])
        else:
            energy = float(parts[-2])
        res.append({"method": method, "label": label, "energy": energy})
    return pd.DataFrame(res)


other = read_other()


# %%
# xtb
matching = pd.merge(
    other.query("method=='xtb'"), df, suffixes=("_approx", "_actual"), on="label"
).sort_values("energy_actual")
# plt.scatter(matching.energy_actual.values, matching.energy_approx.values-matching.nbn)
print(
    scipy.stats.spearmanr(
        matching.energy_actual.values, matching.energy_approx.values - matching.nbn
    )
)
xtbranks = matching.energy_approx.values - matching.nbn
actualranks = matching.energy_actual.values
xtb_misrank = np.arange(len(actualranks)) - np.argsort(xtbranks)


# %%
misranks = {}
for method in "PBE PBE0 B3LYP".split():
    matching = pd.merge(
        other.query("method==@method"), df, suffixes=("_approx", "_actual"), on="label"
    ).sort_values("energy_actual")
    print(
        method,
        scipy.stats.spearmanr(
            matching.energy_actual.values, matching.energy_approx.values
        ),
    )
    misranks[method] = np.arange(len(matching.energy_approx.values)) - np.argsort(
        matching.energy_approx.values
    )

# # %%

# # %%
def bond_count(label):
    bonds = {
        "BH": 0,
        "CH": 0,
        "HN": 0,
        "BB": 0,
        "BC": 0,
        "BN": 0,
        "CC": 0,
        "CN": 0,
        "NN": 0,
    }
    infile = [
        (4, 14),
        (3, 13),
        (3, 9),
        (2, 3),
        (4, 9),
        (4, 5),
        (5, 15),
        (2, 12),
        (8, 9),
        (1, 2),
        (5, 6),
        (1, 11),
        (0, 1),
        (6, 16),
        (6, 7),
        (7, 8),
        (0, 8),
        (7, 17),
        (0, 10),
    ]
    for letter in label[:-2]:
        k = "".join(sorted([letter, "H"]))
        bonds[k] += 1

    for a, b in infile:
        if a > 9 or b > 9:
            continue
        k = "".join(sorted([label[_] for _ in (a, b)]))
        bonds[k] += 1
    return bonds


def build_mat(group, bondorder):
    A = []
    for label in group:
        counts = bond_count(label)
        A.append([counts[_] for _ in bondorder])
    return np.array(A)


def bondcounting_figure():
    plt.rc("font", family="serif")
    df["eelec"] = df.energy - df.nn
    for group in (df.sort_values("eelec"),):
        bondorder = ["BB", "BC", "BH", "BN", "CC", "CH", "CN", "HN", "NN"]
        truelabels = [
            _.replace("5", "B").replace("6", "C").replace("7", "N")
            for _ in group.label.values
        ]
        A = build_mat(truelabels, bondorder)
        coeffs = np.linalg.lstsq(A, group.eelec.values, rcond=None)[0]
        energies = np.dot(A, coeffs)
        print(scipy.stats.spearmanr(energies, group.eelec.values))
        return np.argsort(energies)


bc = bondcounting_figure()
bc_misrank = bc - np.arange(len(bc))

# %%
apdft_rank = np.loadtxt("../../naphthalene-standalone/APDFT-ranking.txt")
apdft_misrank = apdft_rank - np.arange(len(apdft_rank))

# %%
plt.style.use("/mnt/c/Users/guido/workcopies/dotfiles/plotstyle/col1.mplstyle")
plt.figure(figsize=(3, 3))
plt.plot(np.arange(len(bc_misrank)), sorted(np.abs(bc_misrank)), label="BC")
plt.plot(np.arange(len(apdft_misrank)), sorted(np.abs(apdft_misrank)), label="Alchemy")
plt.plot(np.arange(len(xtb_misrank)), sorted(np.abs(xtb_misrank)), label="xTB")
for method in "PBE PBE0 B3LYP".split():
    plt.plot(
        np.arange(len(misranks[method])), sorted(np.abs(misranks[method])), label=method
    )
plt.yscale("log")
plt.legend()
plt.savefig("inset2.pdf")
# %%

# PBE: 0.9983
# PBE0: 0.9998
# B3LYP: 0.9998
# xTB: 0.9966
# BC: 0.9562
# APDFT: 0.9899
scipy.stats.spearmanr(actual, np.arange(len(actual)))


# %%
