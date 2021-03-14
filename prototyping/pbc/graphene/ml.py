#!/usr/bin/env python
#%%
import os
import sys

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
        central_cutoff=7,
        size=70,
        central_decay=1,
        interaction_cutoff=7,
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
        central_cutoff=7,
        size=70,
        central_decay=1,
        interaction_cutoff=7,
        interaction_decay=1,
        indices=range(128 * 4, 128 * 5),
    )

    return np.vstack((uprep[:64, :], dnrep[64:, :]))


def distance_pbc(a, b, h_matrix):
    hinv = np.linalg.inv(h_matrix)
    a_t = np.dot(hinv, a)
    b_t = np.dot(hinv, b)
    t_12 = b_t - a_t
    t_12 -= np.round(t_12)
    return np.linalg.norm(np.dot(h_matrix, t_12))


def build_rep_global(poscar, symmetrize=False):
    # input
    with open(poscar) as fh:
        poslines = fh.readlines()

    # coordinates
    hmat = np.zeros((3, 3))
    a = np.array(list(map(float, poslines[2].strip().split())))
    b = np.array(list(map(float, poslines[3].strip().split())))
    hmat[:, 0] = a
    hmat[:, 1] = b
    hmat[:, 2] = (0, 0, 100)

    coords = []
    for scaledpos in poslines[8:]:
        parts = scaledpos.strip().split()
        coords.append(a * float(parts[0]) + b * float(parts[1]))
    coords = np.array(coords)

    # nuclear charges
    elements = poslines[5].strip().split()
    counts = [int(_) for _ in poslines[6].strip().split()]
    Zs = []
    for element, count in zip(elements, counts):
        Zs += [{"B": 5, "C": 6, "N": 7}[element]] * count
    Zs = np.array(Zs)

    # global rep cutout
    def _get_cm(coords, Zs, hmat):
        cm = np.zeros((128, 128))
        for i in range(128):
            for j in range(i, 128):
                if i == j:
                    cm[i, j] = 0.5 * Zs[i] ** 2.4
                else:
                    dij = distance_pbc(coords[i], coords[j], hmat)
                    cm[i, j] = Zs[i] * Zs[j] / dij
                    cm[j, i] = cm[i, j]
        return cm

    def fn(a, b):
        # finds index of the first non matching element
        idx = np.where((a > b) != (a < b))[0][0]

        if a[idx] < b[idx]:
            return a
        else:
            return b

    if symmetrize:
        cm_up = _get_cm(coords, Zs, hmat)
        cm_dn = _get_cm(coords, 12 - Zs, hmat)

        cm = [fn(cm_up[i], cm_dn[i]) for i in range(128)]

        norms = np.array([np.linalg.norm(_) for _ in cm])
        order = np.argsort(norms)
        cm = np.array(cm)[np.ix_(order, order)]
        return cm[np.triu_indices(128)]
    else:
        cm = _get_cm(coords, Zs, hmat)
        norms = np.linalg.norm(cm, axis=1)
        order = np.argsort(norms)
        cm = cm[np.ix_(order, order)]
        return cm[np.triu_indices(128)]


def build_rep_static_order(poscar):
    # input
    with open(poscar) as fh:
        poslines = fh.readlines()

    coords = []
    for scaledpos in poslines[8:]:
        parts = scaledpos.strip().split()
        coords.append(float(parts[0]), float(parts[1]))
    coords = np.array(coords)
    order = np.lexsort((coords[:, 0], coords[:, 1]))

    # nuclear charges
    elements = poslines[5].strip().split()
    counts = [int(_) for _ in poslines[6].strip().split()]
    Zs = []
    for element, count in zip(elements, counts):
        Zs += [{"B": 5, "C": 6, "N": 7}[element]] * count
    Zs = np.array(Zs)[order] - 6

    return np.outer(Zs, Zs)[np.triu_indices(128)]


def get_KRR_learning_curve_holdout(representations, Y, k, kouter=10, holdoutshare=0.2):
    ndatapoints = len(Y)
    totalidx = np.arange(ndatapoints, dtype=np.int)
    holdoutstart = int(len(totalidx) * (1 - holdoutshare))
    maxtrainingset = np.floor(np.log(holdoutstart) / np.log(2))
    allsigmas = 2.0 ** np.arange(-2, 15)

    print("Calculate kernel matrices")
    if len(representations.shape) == 2:
        # global rep
        D = skm.pairwise_distances(representations, n_jobs=os.cpu_count())
        K = np.array([np.exp(-(D ** 2) / (2 * _ * _)) for _ in allsigmas])
    else:
        # local rep
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
        ns = np.array([_.shape[0] for _ in representations])
        representations = np.concatenate(representations)
        qml.kernels.gaussian_kernel_symmetric()
        K = qml.kernels.get_local_kernels_gaussian(
            representations, representations, ns, ns, allsigmas
        )
        os.environ["OMP_NUM_THREADS"] = "1"

    maes = []
    np.random.seed(42)
    for outer in tqdm.tqdm(range(kouter)):
        np.random.shuffle(totalidx)
        with shared.SharedMemory(
            {"K": K, "Y": Y, "totalidx": totalidx[:holdoutstart]},
            nworkers=os.cpu_count(),
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
            foldresults = pd.DataFrame(sm.evaluate(progress=False))

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

            params = (
                foldresults.groupby("sigmaidx lexp ntrain".split())
                .mean()["mae"]
                .reset_index()
                .sort_values("ntrain")
            )
            ns = []
            for ntrain, group in params.groupby("ntrain"):
                loc = group.sort_values("mae").iloc[0]
                ns.append(ntrain)
                other_case(int(loc.sigmaidx), int(loc.lexp), ntrain, holdoutstart)
            maes.append(sm.evaluate(progress=False))
    return ns, np.average(maes, axis=0), np.std(maes, axis=0)


if __name__ == "__main__":
    logfiles = [f"{BASEDIR}/{_}/OUTCAR" for _ in range(1, LIMIT + 1)]
    energies = np.array([read_one_log(_) for _ in logfiles])
    molids = range(1, LIMIT + 1)

    repname = sys.argv[1]
    print(f"running {repname}")

    with shared.SharedMemory({}, nworkers=os.cpu_count()) as sm:

        @sm.register
        def load_rep(molid, repname):
            if repname == "aCM":
                return build_rep(f"{BASEDIR}/{molid}/POSCAR", symmetrize=False)
            if repname == "aCMs":
                return build_rep(f"{BASEDIR}/{molid}/POSCAR", symmetrize=True)
            if repname == "CM":
                return build_rep_global(f"{BASEDIR}/{molid}/POSCAR", symmetrize=False)
            if repname == "CMs":
                return build_rep_global(f"{BASEDIR}/{molid}/POSCAR", symmetrize=True)
            if repname == "dZs":
                return build_rep_static_order(f"{BASEDIR}/{molid}/POSCAR")
            raise NotImplementedError()

        for i in molids:
            load_rep(i, repname)

        print("Build representations")
        reps = sm.evaluate(progress=True)
        reps = np.array(reps)
    q = get_KRR_learning_curve_holdout(reps, energies, 10, kouter=10, holdoutshare=0.15)
    print(q)
