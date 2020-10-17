#!/usr/bin/env python
#%%
import sys
import functools
import numpy as np

sys.path.append("..")
import importlib
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
        print(elements)

        self._A = np.zeros((len(self._mols), len(elements)))
        for molidx, mol in enumerate(self._mols):
            for Z in mol.nuclear_charges:
                self._A[molidx, elements.index(Z)] += 1

    def __call__(self, trainidx, testidx, Y):
        A = self._A[trainidx, :]
        coeff = np.linalg.lstsq(A, Y[trainidx])[0]
        trainresiduals = np.dot(A, coeff)

        A = self._A[testidx, :]
        testresiduals = np.dot(A, coeff)
        return trainresiduals, testresiduals


class BondCounting(Baseline):
    pass


class LennardJonesLorentzBerthelot(Baseline):
    pass


class D3(Baseline):
    pass


# endregion
#%%
compounds, energies = mlmeta.database_naphtalene()
# mlmeta.get_KRR_learning_curve(
#    compounds, "FCHL19", energies, k=10, elements=[1, 5, 6, 7], pad=18
# )
b = DressedAtom(compounds)
mlmeta.get_KRR_learning_curve(compounds, "CM", energies, transformation=b)

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
learning_curve("qm9:100", "CM", "Identity|DressedAtom")
# %%
