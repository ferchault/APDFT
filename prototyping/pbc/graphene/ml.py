#!/usr/bin/env python
#%%
import os

os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
import sklearn.metrics as skm
import qml
import tqdm
import contextshare as shared

#%%

BASEDIR = "/data/guido/graphene-BN/64/up"
LIMIT = 2500


def read_one_log(filename):
    with open(filename) as fh:
        lines = fh.readlines()
    try:
        TEWEN = float([_ for _ in lines if "TEWEN" in _][-1].strip().split()[-1])
        TOTEN = float([_ for _ in lines if "TOTEN" in _][-1].strip().split()[-2])
    except IndexError:
        raise
    return TOTEN - TEWEN


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


def get_KRR_learning_curve_holdout(representations, Y, k, holdoutshare=0.2):
    ndatapoints = len(Y)
    totalidx = np.arange(ndatapoints, dtype=np.int)
    np.random.shuffle(totalidx)
    holdoutstart = int(len(totalidx) * (1-holdoutshare))
    maxtrainingset = np.floor(np.log(holdoutstart) / np.log(2))
    allsigmas = 2.0 ** np.arange(-2, 15)

    print ("Calculate kernel matrices")
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    # D = skm.pairwise_distances(representations, n_jobs=-1)
    ns = np.array([_.shape[0] for _ in representations])
    representations = np.concatenate(representations)
    K = qml.kernels.get_local_kernels_gaussian(
        representations, representations, ns, ns, allsigmas
    )
    os.environ["OMP_NUM_THREADS"] = "1"

    with shared.SharedMemory(
        {"K": K, "Y": Y, "totalidx": totalidx[:holdoutstart]}, nworkers=os.cpu_count()
    ) as sm:

        @sm.register
        def one_case(sigmaidx, lexp, ntrain, nsplit, chunk):
            trainvalid = totalidx[:ntrain]
            Ytrainvalid = Y[trainvalid]

            # split train into segments
            width = int(ntrain / nsplit)
            start = width * chunk
            end = width * (chunk + 1)
            if chunk == nsplit:
                end = ntrain
            train = np.hstack((trainvalid[:start], trainvalid[end:]))
            validation = trainvalid[start:end]
            Y_train = np.hstack((Ytrainvalid[:start], Ytrainvalid[end:]))
            Y_validation = Ytrainvalid[start:end]

            # build model
            lval = 10 ** lexp
            K_subset = K[sigmaidx][np.ix_(train, train)]
            K_subset[np.diag_indices_from(K_subset)] += lval
            alphas = np.linalg.solve(K_subset, Y_train)

            # evaluate model
            K_subset = K[sigmaidx][np.ix_(train, validation)]
            pred = np.dot(K_subset.transpose(), alphas)

            mae = np.abs(pred - Y_validation).mean()
            return {
                "sigmaidx": sigmaidx,
                "lexp": lexp,
                "ntrain": ntrain,
                "chunk": chunk,
                "mae": mae,
            }

        for sigmaidx in range(len(allsigmas)):
            for lexp in (-7, -9, -11, -13):
                for ntrain in 2 ** np.arange(6, maxtrainingset + 1).astype(np.int):
                    for fold in range(k):
                        one_case(sigmaidx, lexp, ntrain, k, fold)
        print ("Hyperparameter optimization")
        foldresults = pd.DataFrame(sm.evaluate(progress=True))
    
    with shared.SharedMemory(
        {"K": K, "Y": Y, "totalidx": totalidx}, nworkers=os.cpu_count()
    ) as sm:
        @sm.register
        def other_case(sigmaidx, lexp, ntrain, holdoutstart):
            train = totalidx[:ntrain]
            holdout = totalidx[holdoutstart:]

            # build model
            lval = 10 ** lexp
            K_subset = K[sigmaidx][np.ix_(train, train)]
            K_subset[np.diag_indices_from(K_subset)] += lval
            alphas = np.linalg.solve(K_subset, Y[train])

            # evaluate model
            K_subset = K[sigmaidx][np.ix_(train, holdout)]
            pred = np.dot(K_subset.transpose(), alphas)

            return np.abs(pred - Y[holdout]).mean()

        params = foldresults.groupby("sigmaidx lexp ntrain".split()).mean()['mae'].reset_index().sort_values("ntrain")
        ns = []
        for ntrain, group in params.groupby("ntrain"):
            loc = group.sort_values("mae").iloc[0]
            ns.append(ntrain)
            other_case(int(loc.sigmaidx), int(loc.lexp), ntrain, holdoutstart)
        print ("Build models")
        maes = sm.evaluate(progress=True)
        return ns, maes

if __name__ == "__main__":
    logfiles = [f"{BASEDIR}/{_}/OUTCAR" for _ in range(1, LIMIT + 1)]
    energies = np.array([read_one_log(_) for _ in logfiles])
    molids = range(1, LIMIT + 1)
    with shared.SharedMemory({}, nworkers=os.cpu_count()) as sm:

        @sm.register
        def load_rep(molid):
            return build_rep(f"{BASEDIR}/{molid}/POSCAR", symmetrize=False)

        for i in molids:
            load_rep(i)

        print ("Build representations")
        reps = sm.evaluate(progress=True)
        reps = np.array(reps)
    # reps2 = [build_rep(f"{BASEDIR}/{_}/POSCAR", symmetrize=True) for _ in molids]
    q = get_KRR_learning_curve_holdout(reps, energies, 10)
    print (q)


# ns, maes, stds = get_KRR_learning_curve(reps, np.array(energies), k=10)
# print(ns, maes, stds)
# ns2, maes2, stds2 = get_KRR_learning_curve(reps2, np.array(energies), k=10)
# print(ns2, maes2, stds2)
# %%
# plt.loglog(ns, maes, "o-", label="orig")
# plt.loglog(ns2, maes2, "o-", label="with symmetry")
# reps.shape
#df = pd.DataFrame(q)
#%%
# xs = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
# yswithsymm = np.array(
#     (
#         91.38916025,
#         69.94414159,
#         56.62984254,
#         41.5187096,
#         32.99853503,
#         25.53158878,
#         18.82178703,
#         9.95941045,
#         6.69857606,
#         5.45049415,
#     )
# )
# ysorig = np.array(
#     (
#         106.71834446,
#         83.70406486,
#         67.0399434,
#         44.90289698,
#         33.4512471,
#         27.73255691,
#         24.38229124,
#         14.79122741,
#         10.28452817,
#         7.93315388,
#     )
# )
# plt.loglog(xs, ysorig / 128 * 1000, "o-", color="C0", label="atomic CM w/cutoff")
# plt.loglog(
#     xs[:-1],
#     ysorig[1:] / 128 * 1000,
#     "o-",
#     color="C0",
#     alpha=0.5,
#     label="expected with symmetry",
# )
# plt.loglog(
#     xs, yswithsymm / 128 * 1000, "s-", color="C1", label="same + alchemical symmetry"
# )
# plt.yticks((40, 60, 80, 100, 200, 400, 800), (40, 60, 80, 100, 200, 400, 800))
# plt.xticks(xs, xs)
# plt.ylabel("meV / atom")
# plt.legend()
# plt.xlabel("Training set size")
# %%
