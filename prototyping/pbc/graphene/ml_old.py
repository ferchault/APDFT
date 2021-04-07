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


# logfiles = [f"/data/guido/graphene-BN/64/up/{_}/OUTCAR" for _ in range(1, 2501)]
logfiles = [f"{_.strip()}/OUTCAR" for _ in open("only64_2.5k").readlines()]
energies = [read_one_log(_) for _ in tqdm.tqdm(logfiles)]
# region
plt.hist(energies)
# region
def build_rep(poscar, symmetrize=False):
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

    upcharges = 64 * [5] + 64 * [7]
    upcharges *= 9
    upcharges = np.array(upcharges)

    uprep = qml.representations.generate_atomic_coulomb_matrix(
        upcharges,
        padded,
        sorting="row-norm",
        central_cutoff=5,
        size=35,
        central_decay=1,
        interaction_cutoff=5,
        interaction_decay=1,
        indices=range(128 * 4, 128 * 5),
    )
    if not symmetrize:
        return uprep

    dncharges = 64 * [7] + 64 * [5]
    dncharges *= 9
    dncharges = np.array(dncharges)
    dnrep = qml.representations.generate_atomic_coulomb_matrix(
        dncharges,
        padded,
        sorting="row-norm",
        central_cutoff=5,
        size=35,
        central_decay=1,
        interaction_cutoff=5,
        interaction_decay=1,
        indices=range(128 * 4, 128 * 5),
    )

    return np.vstack((uprep[:64, :], dnrep[64:, :]))


molids = range(1, 2501)
folders = [_.strip() for _ in open("only64_2.5k").readlines()]
reps = [build_rep(f"{_}/POSCAR", symmetrize=True) for _ in tqdm.tqdm(folders)]
# reps2 = [
#    build_rep(f"/data/guido/graphene-BN/64/up/{_}/POSCAR", symmetrize=True)
#    for _ in tqdm.tqdm(molids)
# ]
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
    holdout = 200
    totalidx = np.arange(len(Y) - holdout, dtype=np.int)
    maxtrainingset = np.floor(np.log(len(Y)) / np.log(2))
    allsigmas = 2.0 ** np.arange(-2, 10)
    Kall = qml.kernels.get_local_kernels_gaussian(q, q, ns, ns, allsigmas)
    for sigmaidx, sigma in enumerate(tqdm.tqdm(allsigmas, desc="Sigmas")):
        Ktotal = Kall[sigmaidx]

        for ntrain in 2 ** np.arange(2, maxtrainingset + 1).astype(np.int):
            maes = {}
            hmaes = {}
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

                    # holdout prediction
                    K_subset = Ktotal[
                        np.ix_(train, np.arange(len(Y) - holdout, len(Y)))
                    ]
                    pred = np.dot(K_subset.transpose(), alphas)
                    actual = Y[-holdout:]

                    thishmae = np.abs(pred - actual).mean()
                    if lexp not in hmaes:
                        hmaes[lexp] = []
                    hmaes[lexp].append(thishmae)

            for lexp in maes.keys():
                mae = np.average(maes[lexp])
                hmae = np.average(hmaes[lexp])
                std = np.std(maes[lexp])
                rows.append(
                    {
                        "sigma": sigma,
                        "lexp": lexp,
                        "ntrain": ntrain,
                        "mae": mae,
                        "hmae": hmae,
                        "std": std,
                    }
                )

    rows = pd.DataFrame(rows)
    ns, maes, stds = [], [], []
    for name, group in rows.groupby("ntrain"):
        bestcase = group.sort_values("mae")
        ns.append(name)
        maes.append(bestcase["hmae"].values[0])
        stds.append(bestcase["std"].values[0])
    ns = np.array(ns)
    maes = np.array(maes)
    stds = np.array(stds)
    order = np.argsort(ns)
    return ns[order], maes[order], stds[order]


ns, maes, stds = get_KRR_learning_curve(reps, np.array(energies), k=10)
print(ns, maes, stds)
# ns2, maes2, stds2 = get_KRR_learning_curve(reps2, np.array(energies), k=10)
# print(ns2, maes2, stds2)
# region
# plt.loglog(ns, maes, "o-", label="orig")
# plt.loglog(ns2, maes2, "o-", label="with symmetry")
# plt.legend()
# region

#%%
xs = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
yswithsymm = np.array(
    (
        91.38916025,
        69.94414159,
        56.62984254,
        41.5187096,
        32.99853503,
        25.53158878,
        18.82178703,
        9.95941045,
        6.69857606,
        5.45049415,
    )
)
ysorig = np.array(
    (
        106.71834446,
        83.70406486,
        67.0399434,
        44.90289698,
        33.4512471,
        27.73255691,
        24.38229124,
        14.79122741,
        10.28452817,
        7.93315388,
    )
)
plt.loglog(xs, ysorig / 128 * 1000, "o-", color="C0", label="atomic CM w/cutoff")
plt.loglog(
    xs[:-1],
    ysorig[1:] / 128 * 1000,
    "o-",
    color="C0",
    alpha=0.5,
    label="expected with symmetry",
)
plt.loglog(
    xs, yswithsymm / 128 * 1000, "s-", color="C1", label="same + alchemical symmetry"
)
plt.yticks((40, 60, 80, 100, 200, 400, 800), (40, 60, 80, 100, 200, 400, 800))
plt.xticks(xs, xs)
plt.ylabel("meV / atom")
plt.legend()
plt.xlabel("Training set size")
# region
