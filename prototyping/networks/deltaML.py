#!/usr/bin/env python
#%%
import qml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import functools

# %%
def read_reference_energies():
    folders = glob.glob("naphtalene/validation-molpro/*/")
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


def read_other():
    blackout = {"b3lyp": [], "pbe": [], "pbe0": [], "xtb": []}
    for method in blackout.keys():
        with open(f"naphtalene/validation-additional/missing_{method}") as fh:
            lines = fh.readlines()
        blackout[method] = [_.strip("/").split("-")[-1] for _ in lines]

    with open("naphtalene/validation-additional/RESULTS") as fh:
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


df = read_reference_energies()
other = read_other()
# %%
def get_rep(label):
    c = qml.Compound("build_colored_graphs/db-1/inp.xyz")
    c.nuclear_charges[:10] = [int(_) for _ in str(label)]
    c.generate_coulomb_matrix(size=18, sorting="row-norm")
    return c.representation


@functools.lru_cache(maxsize=1)
def get_kernel(sigma):
    X = np.array([get_rep(_) for _ in df.sort_values("energy").label.values])

    K = qml.kernels.gaussian_kernel(X, X, sigma)
    return K
    # self._K[np.diag_indices_from(self._K)] += self._parameters["lambda"]
    #    self._alphas = qml.math.cho_solve(self._K, energies)


K = get_kernel(10)


# %%
def KRR(kernel, lval, tss):
    score = []
    idx = np.arange(kernel.shape[0])
    for k in range(2):
        np.random.shuffle(idx)
        train, test = idx[:tss], idx[tss:]

        subset = kernel[train][:, train]
        subset[np.diag_indices_from(subset)] += lval
        alphas = qml.math.cho_solve(subset, train)

        subset = kernel[test][:, train]
        ranks = np.dot(subset, alphas)
        plt.scatter(test, ranks)
        return ranks


KRR(K, 1e-12, 500)

# %%

# %%
