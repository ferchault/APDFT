#!/usr/bin/env python
#%%
# region imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functools
import qml
import requests
import io
import gzip
import tarfile
import tqdm

# endregion

# region load data
class MockXYZ(object):
    """Helper class satisfying the signature QML requires to read XYZ files from memory."""

    def __init__(self, lines):
        self._lines = lines

    def readlines(self):
        return self._lines


@functools.lru_cache(maxsize=1)
def database_naphthalene():
    """ Reads the BN-doped naphthalene database from network, https://arxiv.org/abs/2008.02784."""
    # energies
    df = pd.read_csv("https://zenodo.org/record/3994178/files/reference.csv?download=1")

    # geometry
    res = requests.get("https://zenodo.org/record/3994178/files/inp.xyz?download=1")
    geo = res.content.decode("ascii")

    # molecules
    compounds = []
    for label in df.label.values:
        c = qml.Compound(xyz=MockXYZ(geo.split("\n")))
        c.nuclear_charges = [int(_) for _ in str(label)] + [1] * 8
        compounds.append(c)
    return compounds, df.CCSDenergyinHa.values


@functools.lru_cache(maxsize=1)
def database_qmrxn20():
    """ Reads transitition state geometries from network, https://iopscience.iop.org/article/10.1088/2632-2153/aba822."""
    # energies
    energiesurl = "https://archive.materialscloud.org/record/file?file_id=0eaa6011-b9d7-4c30-b424-2097dd90c77c&filename=energies.txt.gz&record_id=414"
    res = requests.get(energiesurl)
    webfh = io.BytesIO(res.content)
    with gzip.GzipFile(fileobj=webfh) as fh:
        lines = [_.decode("ascii") for _ in fh.readlines()]
    relevant = [
        _ for _ in lines if "transition-states/" in _ and ".xyz" in _ and "lccsd" in _
    ]
    filenames = [line.strip().split(",")[4] for line in relevant]
    energies = np.array([float(line.strip().split(",")[-2]) for line in relevant])

    # geometries
    geometriesurl = "https://archive.materialscloud.org/record/file?file_id=4905b29e-a989-48a3-8429-32e1db989972&filename=geometries.tgz&record_id=414"
    res = requests.get(geometriesurl)
    webfh = io.BytesIO(res.content)
    t = tarfile.open(fileobj=webfh)
    mols = {}
    for fo in t:
        if fo.name in filenames:
            lines = t.extractfile(fo).readlines()
            lines = [_.decode("ascii") for _ in lines]
            mols[fo.name] = qml.Compound(xyz=MockXYZ(lines))
    cs = [mols[_] for _ in filenames]
    return cs, energies


@functools.lru_cache(maxsize=1)
def database_qm9(random_limit=3000):
    """ Reads the QM9 database from network, http://www.nature.com/articles/sdata201422."""
    # exclusion list
    res = requests.get("https://ndownloader.figshare.com/files/3195404")
    exclusion_ids = [
        _.strip().split()[0] for _ in res.content.decode("ascii").split("\n")[9:-2]
    ]

    # geometries and energies
    res = requests.get("https://ndownloader.figshare.com/files/3195389")
    webfh = io.BytesIO(res.content)
    t = tarfile.open(fileobj=webfh)
    energies = []
    contents = []
    for fo in t:
        lines = t.extractfile(fo).read().decode("ascii").split("\n")
        natoms = int(lines[0])
        lines = lines[: 2 + natoms]
        lines = [_.replace("*^", "e") for _ in lines]
        molid = lines[1].strip().split()[0]
        if molid in exclusion_ids:
            continue
        energies.append(float(lines[1].strip().split()[12]))
        contents.append(lines)

    # random subset for development purposes
    idx = np.arange(len(energies))
    np.random.shuffle(idx)
    subset = idx[:random_limit]

    energies = [energies[_] for _ in subset]
    compounds = [qml.Compound(xyz=MockXYZ(contents[_])) for _ in subset]
    return compounds, np.array(energies)


# endregion
# %%
def get_representation(mol, repname, **repkwargs):
    if repname == "CM":
        mol.generate_coulomb_matrix(sorting="row-norm", **repkwargs)
        return mol.representation.copy()
    if repname == "FCHL19":
        rep = qml.representations.generate_fchl_acsf(
            mol.nuclear_charges, mol.coordinates, gradients=False, **repkwargs
        )
        qs = mol.nuclear_charges
        return rep, qs
    raise ValueError("Unknown representation")


def get_KRR_learning_curve(
    molecules, representation, Y, transformation=None, k=5, **repkwargs
):
    if representation == "FCHL19":
        combined = [
            get_representation(_, representation, **repkwargs) for _ in molecules
        ]
        X = np.array([_[0] for _ in combined])
        Q = np.array([_[1] for _ in combined])
    else:
        X = np.array(
            [get_representation(_, representation, **repkwargs) for _ in molecules]
        )

    rows = []
    totalidx = np.arange(len(X), dtype=np.int)
    maxtrainingset = np.floor(np.log(len(X)) / np.log(2))
    allsigmas = 2.0 ** np.arange(-2, 10)
    for sigmaidx, sigma in enumerate(tqdm.tqdm(allsigmas, desc="Sigmas")):
        if representation == "FCHL19":
            if sigmaidx == 0:
                Ktotalcache = qml.kernels.get_local_symmetric_kernels(X, Q, allsigmas)
            Ktotal = Ktotalcache[sigmaidx]
        else:
            Ktotal = qml.kernels.gaussian_kernel_symmetric(X, sigma)

        for ntrain in 2 ** np.arange(2, maxtrainingset + 1).astype(np.int):
            maes = {}
            for fold in range(k):
                np.random.shuffle(totalidx)
                train, test = totalidx[:ntrain], totalidx[ntrain:]
                if transformation is None:
                    btrain, btest = 0, 0
                else:
                    btrain, btest = transformation(train, test, Y)

                for lexp in (-7, -9, -11, -13):
                    lval = 10 ** lexp
                    K_subset = Ktotal[np.ix_(train, train)]
                    K_subset[np.diag_indices_from(K_subset)] += lval
                    alphas = qml.math.cho_solve(K_subset, Y[train] - btrain)

                    K_subset = Ktotal[np.ix_(train, test)]
                    pred = np.dot(K_subset.transpose(), alphas) + btest
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