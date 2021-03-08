#!/usr/bin/env python
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import tqdm
import sklearn.metrics as skm
import qml

# region
def read_one_log(filename):
    with open(filename) as fh:
        lines = fh.readlines()
    try:
        TEWEN = float([_ for _ in lines if "TEWEN" in _][-1].strip().split()[-1])
        TOTEN = float([_ for _ in lines if "TOTEN" in _][-1].strip().split()[-2])
    except IndexError:
        raise

    return TOTEN - TEWEN


logfiles = [f"/data/guido/graphene-BN/64/up/{_}/OUTCAR" for _ in range(1, 201)]
energies = [read_one_log(_) for _ in tqdm.tqdm(logfiles)]
# region
plt.hist(energies)
# region
def build_rep(poscar):
    # assumptions: xy plane graphene
    with open(poscar) as fh:
        poslines = fh.readlines()

    a = np.array(list(map(float, poslines[2].strip().split())))
    b = np.array(list(map(float, poslines[3].strip().split())))

    elements = poslines[5].strip().split()
    if elements != ["B", "N"]:
        raise NotImplementedError()

    coords = []
    for scaledpos in poslines[8:]:
        parts = scaledpos.strip().split()
        coords.append(a * float(parts[0]) + b * float(parts[1]))

    coords = np.array(coords)
    padded = []
    for dx in range(3):
        for dy in range(3):
            padded.append(coords + dx * a + dy * b)
    padded = np.vstack(padded)

    if len(coords) != 128:
        raise NotImplementedError()

    charges = 64 * [5] + 64 * [7]
    charges *= 9
    charges = np.array(charges)

    return qml.representations.generate_atomic_coulomb_matrix(
        charges,
        padded,
        sorting="row-norm",
        central_cutoff=5,
        size=35,
        central_decay=1,
        interaction_cutoff=5,
        interaction_decay=1,
        indices=range(128 * 4, 128 * 5),
    )


molids = range(1, 201)
reps = [
    build_rep(f"/data/guido/graphene-BN/64/up/{_}/POSCAR") for _ in tqdm.tqdm(molids)
]
# reps
# reps = np.concatenate(reps)
# cs, ctrs = build_rep("/data/guido/graphene-BN/64/up/1466/POSCAR")
# plt.scatter(cs[:, 0], cs[:, 1])
# b, e = 128*4, 128*5
# plt.scatter(cs[b:e, 0], cs[b:e, 1])

# region
def get_KRR_learning_curve(reps, Y, k=5):
    ns = [128] * len(reps)
    q = np.concatenate(reps)
    rows = []
    totalidx = np.arange(len(Y), dtype=np.int)
    maxtrainingset = np.floor(np.log(len(Y)) / np.log(2))
    allsigmas = 2.0 ** np.arange(-2, 10)
    for sigmaidx, sigma in enumerate(tqdm.tqdm(allsigmas, desc="Sigmas")):
        Ktotal = qml.kernels.get_local_kernels_gaussian(q, q, ns, ns, [sigma])[0]

        for ntrain in 2 ** np.arange(2, maxtrainingset + 1).astype(np.int):
            maes = {}
            for fold in range(k):
                np.random.shuffle(totalidx)
                train, test = totalidx[:ntrain], totalidx[ntrain:]

                for lexp in (-9,):
                    lval = 10 ** lexp
                    K_subset = Ktotal[np.ix_(train, train)]
                    K_subset[np.diag_indices_from(K_subset)] += lval
                    alphas = qml.math.cho_solve(K_subset, Y[train])

                    K_subset = Ktotal[np.ix_(train, test)]
                    pred = np.dot(K_subset.transpose(), alphas)
                    actual = Y[test]

                    thismae = np.abs(pred - actual).mean()
                    if lexp not in maes:
                        maes[lexp] = []
                    maes[lexp].append(thismae)

            for lexp in maes.keys():
                mae = np.average(maes[lexp])
                std = np.std(maes[lexp])
                rows.append(
                    {
                        "sigma": sigma,
                        "lexp": lexp,
                        "ntrain": ntrain,
                        "mae": mae,
                        "std": std,
                    }
                )

    rows = pd.DataFrame(rows)
    ns, maes, stds = [], [], []
    for name, group in rows.groupby("ntrain"):
        bestcase = group.sort_values("mae")
        ns.append(name)
        maes.append(bestcase["mae"].values[0])
        stds.append(bestcase["std"].values[0])
    ns = np.array(ns)
    maes = np.array(maes)
    stds = np.array(stds)
    order = np.argsort(ns)
    return ns[order], maes[order], stds[order]


ns, maes, stds = get_KRR_learning_curve(reps, np.array(energies), k=10)
# region
plt.loglog(ns, maes, "o-")
# region
