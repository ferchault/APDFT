#!/usr/bin/env python
#%%
import qml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import functools
import scipy.stats

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
    # c.nuclear_charges[:10] = [int(_) for _ in str(label)]
    # c.generate_coulomb_matrix(size=18, sorting="row-norm")
    # return c.representation
    charges = np.array([int(_) for _ in str(label)])
    # rep = qml.representations.generate_bob(
    #    charges, c.coordinates[:10], "BCN".split(), 10, {"B": 5, "C": 10, "N": 5}
    # )
    rep = qml.representations.generate_fchl_acsf(
        charges, c.coordinates[:10], elements=[5, 6, 7], pad=10, gradients=False
    )
    return rep


@functools.lru_cache(maxsize=10)
def get_kernel(sigma, kind):
    X = np.array([get_rep(_) for _ in df.sort_values("energy").label.values])
    Q = np.array(
        [
            [int(_) for _ in str(label)]
            for label in df.sort_values("energy").label.values
        ]
    )

    # if kind == "gaussian":
    #    K = qml.kernels.gaussian_kernel(X, X, sigma)
    # if kind == "laplacian":
    #    K = qml.kernels.laplacian_kernel(X, X, sigma)
    # return K
    return qml.kernels.get_local_symmetric_kernel(X, Q, sigma)


def get_misranks():
    misranks = {}
    for method in "PBE PBE0 B3LYP".split():
        matching = pd.merge(
            other.query("method==@method"),
            df,
            suffixes=("_approx", "_actual"),
            on="label",
        ).sort_values("energy_actual")

        misranks[method] = np.arange(len(matching.energy_approx.values)) - np.argsort(
            matching.energy_approx.values
        )
    matching = pd.merge(
        other.query("method=='xtb'"), df, suffixes=("_approx", "_actual"), on="label"
    ).sort_values("energy_actual")
    xtbranks = matching.energy_approx.values - matching.nbn
    actualranks = matching.energy_actual.values
    misranks["xtb"] = np.arange(len(actualranks)) - np.argsort(xtbranks)

    apdft_rank = np.loadtxt("naphthalene-standalone/APDFT-ranking.txt")
    misranks["apdft"] = apdft_rank - np.arange(len(apdft_rank))

    misranks["bc"] = np.loadtxt("bc_misrank.txt")
    return misranks


misranks = get_misranks()

# %%
def KRR(kernel, lval, tss, nfold=2, baseline=None):
    scores = []
    idx = np.arange(kernel.shape[0])
    for k in range(nfold):
        np.random.shuffle(idx)
        train, test = idx[:tss], idx[tss:]
        if baseline is None:
            Y_train = train.copy()
            Y_test = test.copy()
        else:
            base = misranks[baseline]  # + np.arange(len(idx))
            towards = misranks["B3LYP"]  # + np.arange(len(idx))
            bmod = base - towards
            Y_train = bmod[train]
            Y_test = base[test]

        subset = kernel[train][:, train]
        subset[np.diag_indices_from(subset)] += lval
        alphas = qml.math.cho_solve(subset, Y_train)

        subset = kernel[test][:, train]
        ranks = np.dot(subset, alphas)
        score = np.abs(Y_test - ranks).mean()
        if baseline == "apdft":
            pred = np.abs(Y_test - ranks).mean()
        # score = scipy.stats.spearmanr(Y_test, ranks).correlation
        scores.append(score)
    return np.array(scores).mean()


sigmas = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
scores = []
for kind in "gaussian laplacian".split():
    for sigma in sigmas:
        K = get_kernel(sigma, kind)
        for lval in (1e-11,):  # , 1e-7, 1e-9:
            scores.append(KRR(K, lval, 200, baseline="apdft"))
            print(kind, sigma, lval, scores[-1])

# %%
sigma = 256
K = get_kernel(sigma, "laplacian")
learningdelta = []
Ns = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
for N in Ns:
    learningdelta.append(KRR(K, 1e-11, N, 10, baseline="apdft"))

# sigma=256
# K = get_kernel(sigma, "laplacian")
# learning = []
# Ns = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
# for N in Ns:
#    learning.append(KRR(K, 1e-11, N, 10, baseline="apdft"))


# %%
plt.loglog(Ns, 1 - np.array(learningdelta))

# %%
# found manually, in seconds
costs = {
    "CCSD": 27 * 60,
    "PBE": 35,
    "PBE0": 40,
    "B3LYP": 40,
    "xtb": 0.065,
    "bc": 27 * 60,  # estimate as ref method
    "apdft": (4 * 60 + 5) / 2286,
}
spearmans = {
    "PBE0": 0.9998,
    "apdft": 0.9899,
    "B3LYP": 0.9998,
    "xtb": 0.9966,
    "bc": 0.9562,
    "PBE": 0.9983,
}
for method in "PBE PBE0 B3LYP xtb bc apdft".split():
    time = 2286 * costs[method]
    score = np.abs(misranks[method]).mean()
    # score = spearmans[method]
    plt.scatter(time, score, label=method)
# plt.plot(np.array(Ns) * costs["CCSD"], 1-np.array(learning), label="Direct ML")
plt.plot(
    np.array(Ns) * costs["B3LYP"] + costs["apdft"],
    np.array(learningdelta),
    "o-",
    label="DeltaML",
)
plt.legend(ncol=2, frameon=False)
plt.yscale("log")
plt.xscale("log")

# %%
np.abs(misranks["apdft"]).mean()
# %%
Ns, learningdelta
# %%
def change_reference(transform, new_base):
    return transform - new_base


testcase = np.array((0, 1, 4, 3, 2, 5)) - np.arange(6)
change_reference(testcase, testcase)
# plt.plot(misranks["apdft"] + np.arange(len(misranks['apdft'])))
# %%

