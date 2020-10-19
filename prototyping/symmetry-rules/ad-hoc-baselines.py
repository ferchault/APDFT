#!/usr/bin/env python
#%%
import sys
import functools
import importlib
import itertools as it

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
import scipy.optimize as sco

sys.path.append("..")
import mlmeta

importlib.reload(mlmeta)

#%%
# region Baselines
class Baseline:
    def __init__(self, mols):
        """ Build fitting cache entries."""
        self._mols = mols
        self._build_cache()

    def _build_cache(self):
        raise NotImplementedError()

    def __call__(self, trainidx, testidx, Y):
        raise NotImplementedError()


class Pipeline:
    def __init__(self, mols, classes):
        self._mols = mols
        self._stages = []
        for classname in classes:
            c = classname(mols)
            self._stages.append(c)

    def __call__(self, trainidx, testidx, Y):
        Ytest = 0
        Ytrain = 0
        Y = np.array(Y).copy()
        for stage in self._stages:
            btrain, btest = stage(trainidx, testidx, Y)
            Y[trainidx] -= btrain
            Ytest += btest
            Ytrain += btrain
        return Ytrain, Ytest


class Identity(Baseline):
    def _build_cache(self):
        pass

    def __call__(self, trainidx, testidx, Y):
        return 0, 0


class DressedAtom(Baseline):
    def _build_cache(self):
        elements = set()
        for mol in self._mols:
            elements = elements | set(mol.nuclear_charges)
        elements = sorted(elements)

        self._A = np.zeros((len(self._mols), len(elements)))
        for molidx, mol in enumerate(self._mols):
            for Z in mol.nuclear_charges:
                self._A[molidx, elements.index(Z)] += 1

    def __call__(self, trainidx, testidx, Y):
        # fit
        A = self._A[trainidx, :]
        coeff = np.linalg.lstsq(A, Y[trainidx])[0]
        trainresiduals = np.dot(A, coeff)

        # transform
        A = self._A[testidx, :]
        testresiduals = np.dot(A, coeff)
        return trainresiduals, testresiduals


class BondCounting(Baseline):
    pass


class LennardJonesLorentzBerthelot(Baseline):
    def _build_cache(self):
        elements = set()
        for mol in self._mols:
            elements = elements | set(mol.nuclear_charges)
        self._elements = sorted(elements)
        combos = it.combinations_with_replacement(self._elements, r=2)
        kinds = ["-".join(map(str, _)) for _ in combos]

        mat6 = np.zeros((len(self._mols), len(kinds)))
        mat12 = np.zeros((len(self._mols), len(kinds)))
        for idx, mol in enumerate(self._mols):
            dm = ssd.squareform(ssd.pdist(mol.coordinates))
            for i in range(mol.natoms):
                for j in range(i + 1, mol.natoms):
                    a, b = sorted((mol.nuclear_charges[i], mol.nuclear_charges[j]))
                    mat6[labelidx, kinds.index(f"{a}-{b}")] += 1 / dm[i, j] ** 6
                    mat12[labelidx, kinds.index(f"{a}-{b}")] += 1 / dm[i, j] ** 12

        self._kinds = kinds
        self._mat6 = mat6
        self._mat12 = mat12

    def _residuals(self, params, trainidx, Y):
        return np.linalg.norm(self._predict(trainidx, params) - Y)

    def _predict(self, trainidx, params):
        order = self._elements
        sigmas = np.zeros(len(self._kinds))
        epsilons = np.zeros(len(self._kinds))
        for kidx, kind in enumerate(self._kinds):
            kind = kind.split("-")
            e1 = order.index(kind[0])
            e2 = order.index(kind[1])
            eps1 = parameters[e1] * parameters[e1]
            eps2 = parameters[e2] * parameters[e2]
            sig1 = parameters[e1 + 4] * parameters[e1 + 4]
            sig2 = parameters[e2 + 4] * parameters[e2 + 4]
            epsilons[kidx] = np.sqrt(eps1 * eps2)
            sigmas[kidx] = (sig1 + sig2) / 2
        pred = np.dot(
            self._mat12[trainidx, :] * sigmas ** 12
            - self._mat6[trainidx, :] * sigmas ** 6,
            epsilons,
        )
        return pred

    def __call__(self, trainidx, testidx, Y):
        result = sco.differential_evolution(
            self._residuals,
            bounds=[(0.5, 2)]*len(self._elements*2),
            workers=-1,
            args={'trainidx': trainidx, 'Y': Y[trainidx]}
        )
        btrain = self._predict(trainidx, result.x)
        btest = self._predict(testidx, result.x)
        return btrain, btest

class D3(Baseline):
    pass


class NuclearNuclear(Baseline):
    pass


# endregion
# %%
@functools.lru_cache(maxsize=1000)
def learning_curve(dataset, repname, transformations):
    # base setup
    dbargs = {}
    if dataset.startswith("qm9"):
        dataset, cutoff = dataset.split(":")
        dbargs["random_limit"] = int(cutoff)

    compounds, energies = getattr(mlmeta, f"database_{dataset}")(**dbargs)
    if repname == "FCHL19":
        repkwargs = {"elements": [1, 5, 6, 7], "pad": 18}
    if repname == "CM":
        repkwargs = {"size": 35}

    # transformation
    transformations = transformations.split("|")
    transformations = [globals()[_] for _ in transformations]
    ts = Pipeline(compounds, transformations)

    return mlmeta.get_KRR_learning_curve(
        compounds, repname, energies, transformation=ts, **repkwargs
    )


# %%
kcal = 627.509474063
flavors = "Identity Identity|DressedAtom".split()
for flavor in flavors:
    xs, maes, stds = learning_curve("qm9:100", "CM", flavor)
    plt.errorbar(
        x=xs,
        y=maes * kcal,
        fmt="o-",
        yerr=stds * kcal,
        label=flavor,
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=3,
    )
plt.xscale("log")
plt.xticks(xs, xs)
plt.minorticks_off()
plt.yscale("log", subsy=range(2, 10))
plt.legend()
plt.xlabel("Training set size")
plt.ylabel("MAE [kcal/mol]")
# %%

# %%

# %%
compounds, energies = mlmeta.database_qm9(random_limit=100)
dir(compounds[0])
# %%

# %%
