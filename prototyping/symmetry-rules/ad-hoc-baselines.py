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
import MDAnalysis as mda
import basis_set_exchange as bse

sys.path.append("..")
import mlmeta

# importlib.reload(mlmeta)

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
    def _perceive_bonds(self, compound):
        u = mda.Universe.empty(n_atoms=compound.natoms, trajectory=True)
        labels = [
            bse.lut.element_sym_from_Z(_).capitalize() for _ in compound.nuclear_charges
        ]
        u.add_TopologyAttr("type", labels)
        for atom in range(compound.natoms):
            a = mda.core.groups.Atom(atom, u)
            a.position = compound.coordinates[atom]

        return mda.topology.guessers.guess_bonds(u.atoms, compound.coordinates)

    def _build_cache(self):
        # bond perception
        kinds = set()
        bonds = []
        for mol in self._mols:
            bs = self._perceive_bonds(mol)
            mol_kinds = {}
            for bond in bs:
                a, b = sorted(([mol.nuclear_charges[_] for _ in bond]))
                kind = f"{a}-{b}"
                kinds.add(kind)
                if kind not in mol_kinds:
                    mol_kinds[kind] = 0
                mol_kinds[kind] += 1
            bonds.append(mol_kinds)

        # build cache matrix
        kinds = sorted(kinds)
        self._A = np.zeros((len(self._mols), len(kinds)))
        for molidx, molbonds in enumerate(bonds):
            for b in molbonds:
                self._A[molidx, kinds.index(b)] += 1

    def __call__(self, trainidx, testidx, Y):
        # fit
        A = self._A[trainidx, :]
        A2 = A.T.dot(A) + 1e-7 * np.identity(A.shape[1])
        coeff = np.linalg.lstsq(A2, A.T.dot(Y[trainidx]))[0]
        trainresiduals = np.dot(A, coeff)

        # transform
        A = self._A[testidx, :]
        testresiduals = np.dot(A, coeff)
        return trainresiduals, testresiduals


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
                    mat6[idx, kinds.index(f"{a}-{b}")] += 1 / dm[i, j] ** 6
                    mat12[idx, kinds.index(f"{a}-{b}")] += 1 / dm[i, j] ** 12

        self._kinds = kinds
        self._mat6 = mat6
        self._mat12 = mat12

    def _residuals(self, params, trainidx, Y):
        return np.linalg.norm(self._predict(trainidx, params) - Y)

    def _predict(self, trainidx, parameters):
        order = self._elements
        sigmas = np.zeros(len(self._kinds))
        epsilons = np.zeros(len(self._kinds))
        for kidx, kind in enumerate(self._kinds):
            kind = [int(_) for _ in kind.split("-")]
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
        shift = 10  # necessarily positive, as only negative numbers can be reliably modeled with LJ
        if max(Y) > shift:
            print("WARNING: Possibly positive value, check LJ baseline implementation")
        result = sco.differential_evolution(
            self._residuals,
            bounds=[(0.5, 2)] * len(self._elements) * 2,
            workers=1,
            args=(trainidx, Y[trainidx] - shift),
        )
        self._best_params = result.x
        btrain = self._predict(trainidx, result.x) + shift
        btest = self._predict(testidx, result.x) + shift
        return btrain, btest


class NuclearNuclear(Baseline):
    pass


class D3(Baseline):
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

    # load dataset and determine representation parameters accordigly
    compounds, energies = getattr(mlmeta, f"database_{dataset}")(**dbargs)
    elements = set()
    maxlen = 0
    for mol in compounds:
        elements = elements | set(mol.nuclear_charges)
        maxlen = max(maxlen, mol.natoms)
    elements = sorted(elements)

    if repname == "FCHL19":
        repkwargs = {"elements": elements, "pad": maxlen}
    if repname == "CM":
        repkwargs = {"size": maxlen}

    # transformation
    transformations = transformations.split("|")
    transformations = [globals()[_] for _ in transformations]
    ts = Pipeline(compounds, transformations)

    # determine null model for this transformation
    btrain, _ = ts(np.arange(len(energies)), [], energies)
    residuals = energies - btrain
    nullmodel = np.average(np.abs(np.median(residuals) - residuals))

    res = mlmeta.get_KRR_learning_curve(
        compounds, repname, energies, k=1, transformation=ts, **repkwargs
    )
    return *res, nullmodel


# %%
kcal = 627.509474063
repname = "FCHL19"
dbbname = "naphthalene"
flavors = "Identity DressedAtom DressedAtom|LennardJonesLorentzBerthelot LennardJonesLorentzBerthelot BondCounting".split()
maxnull = 0
for fidx, flavor in enumerate(flavors):
    xs, maes, stds, nullmodel = learning_curve(dbname, repname, flavor)
    maxnull = max(maxnull, nullmodel)
    print(flavor, repname, maxnull)
    label = "".join([_ for _ in flavor if _.isupper() or _ in "|"])
    label = f"{label}@{repname}"
    plt.errorbar(
        x=xs,
        y=maes * kcal,
        fmt="o-",
        yerr=stds * kcal,
        label=label,
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=3,
        color=f"C{fidx}",
    )
plt.title(dbname)
plt.xscale("log")
plt.xticks(xs, xs)
plt.minorticks_off()
plt.yscale("log", subsy=range(2, 10))
plt.xlabel("Training set size")
plt.ylabel("MAE [kcal/mol]")
plt.axhline(maxnull * kcal, label="Null model", color="grey")
plt.legend(frameon=False)
plt.ylim(1, 10 ** np.ceil(np.log(maxnull * kcal) / np.log(10)))
# plt.xlim(64, max(xs))

# %%
# hyperparameters scanned large enough space?
# residual norm for DA|LJLB - since learning seems worse
# larger tss
# cache last DE results
cs, es = mlmeta.database_naphthalene()
xs = np.arange(len(cs)).astype(np.int)
da = DressedAtom(cs)
captured, _ = da(xs, [], es)


# %%
import time

lj = LennardJonesLorentzBerthelot(cs)

# %%
from pyinstrument import Profiler

profiler = Profiler()
profiler.start()
capturedlj, _ = lj(xs, [], es - captured)
profiler.stop()

print(profiler.output_text(unicode=True, color=True))


# %%
lj._best_params
# %%
