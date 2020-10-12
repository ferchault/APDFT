#!/usr/bin/env python

# baseline for total E: 1/r, 1/r^6, bond energies, simple FF
# baseline for electronic E: parabola, dressed atom, alchemy rules
# electronic E: representations CM, M

#%%
# region imports
# ML
import qml
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# AD
# keep jax config first
from jax.config import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp

# Test for double precision import
x = jax.random.uniform(jax.random.PRNGKey(0), (1,), dtype=jnp.float64)
assert x.dtype == jnp.dtype("float64"), "JAX not in double precision mode"

# Helper
import pandas as pd
import matplotlib.pyplot as plt
import functools
import requests
import numpy as np
import itertools as it
import scipy.spatial.distance as ssd
import scipy.optimize as sco

# endregion
# region data preparation
@functools.lru_cache(maxsize=1)
def fetch_energies():
    """Loads CCSD/cc-pVDZ energies.

    Returns
    -------
    DataFrame
        Energies and metadata
    """
    df = pd.read_csv("https://zenodo.org/record/3994178/files/reference.csv?download=1")
    C = -37.69831437
    B = -24.58850790
    N = -54.36533180
    df["totalE"] = df.CCSDenergyinHa.values
    df["nuclearE"] = df.NNinteractioninHa.values
    df["electronicE"] = df.totalE.values - df.nuclearE.values

    # atomisation energy
    atoms = (10 - 2 * df.nBN.values) * C + df.nBN.values * (B + N)
    df["atomicE"] = df.totalE.values - atoms

    df = df["label nBN totalE nuclearE electronicE atomicE".split()].copy()
    return df


@functools.lru_cache(maxsize=1)
def fetch_geometry():
    """Loads the XYZ geometry for naphthalene used in the reference calculations.

    Returns
    -------
    str
        XYZ file contents, ascii.
    """
    res = requests.get("https://zenodo.org/record/3994178/files/inp.xyz?download=1")
    return res.content.decode("ascii")


# endregion
# region cached ML boilerplate
class MockXYZ(object):
    def __init__(self, lines):
        self._lines = lines

    def readlines(self):
        return self._lines


@functools.lru_cache(maxsize=3000)
def get_compound(label):
    c = qml.Compound(xyz=MockXYZ(fetch_geometry().split("\n")))
    c.nuclear_charges = [int(_) for _ in str(label)] + [1] * 8
    return c


# endregion

#%%
# region non-cached ML
kcal = 627.509474063
totalanmhessian = np.loadtxt("totalhessian.txt")
_, totalanmvectors = np.linalg.eig(totalanmhessian)
anmhessian = np.loadtxt("hessian.txt")
_, anmvectors = np.linalg.eig(anmhessian)


def get_learning_curve(df, repname, propname, baselineclass):
    X = np.array(
        [get_representation(get_compound(_), repname) for _ in df.label.values]
    )
    Y = df[propname].values

    rows = []
    baseline = baselineclass(df)
    totalidx = np.arange(len(X), dtype=np.int)
    for sigma in 2.0 ** np.arange(-2, 10):
        print(sigma)
        Ktotal = qml.kernels.gaussian_kernel(X, X, sigma)

        for lval in (1e-7, 1e-9, 1e-11, 1e-13):
            # for ntrain in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
            for ntrain in (8, 32, 128, 512, 2048):
                maes = []
                for k in range(2):
                    np.random.shuffle(totalidx)
                    train, test = totalidx[:ntrain], totalidx[ntrain:]
                    b = baseline.fit(train, propname)

                    K_subset = Ktotal[np.ix_(train, train)]
                    K_subset[np.diag_indices_from(K_subset)] += lval
                    alphas = qml.math.cho_solve(K_subset, Y[train] - b)

                    K_subset = Ktotal[np.ix_(train, test)]
                    pred = np.dot(K_subset.transpose(), alphas) + baseline.transform(
                        test
                    )
                    actual = Y[test]

                    maes.append(np.abs(pred - actual).mean())
                mae = sum(maes) / len(maes)
                rows.append(
                    {"sigma": sigma, "lval": lval, "ntrain": ntrain, "mae": mae}
                )

    rows = pd.DataFrame(rows)
    return rows.groupby("ntrain").min()["mae"]


def canonicalize(dz):
    A = dz
    C = A[[3, 2, 1, 0, 7, 6, 5, 4, 9, 8]]
    E = A[[7, 6, 5, 4, 3, 2, 1, 0, 8, 9]]
    G = E[[3, 2, 1, 0, 7, 6, 5, 4, 9, 8]]
    reps = np.array((A, C, E, G))
    dz = reps[np.lexsort(np.vstack(reps).T)[0]]
    return dz


def get_representation(mol, repname):
    if repname == "CM":
        mol.generate_coulomb_matrix(size=20, sorting="row-norm")
        return mol.representation.copy()
    if repname == "M":
        dz = np.array(mol.nuclear_charges[:10]) - 6
        return np.outer(dz, dz)[np.triu_indices(10)]
    if repname == "MS":
        dz = np.array(mol.nuclear_charges[:10]) - 6
        return np.array(
            (
                sum(dz[[1, 2, 5, 6]] != 0),
                sum(dz[[8, 9]] != 0),
                sum(dz[[0, 3, 4, 7]] != 0),
            )
        )
    if repname == "M2":
        dz = np.array(mol.nuclear_charges[:10]) - 6
        return canonicalize(dz)
    if repname == "M3":
        dz = np.array(mol.nuclear_charges[:10]) - 6
        A = canonicalize(dz)
        if are_strict_alchemical_enantiomers(dz, -dz):
            B = canonicalize(-dz)
            reps = np.array((A, B))
            dz = reps[np.lexsort(np.vstack(reps).T)[0]]
            if np.allclose(dz, B):
                return np.array(list(dz) + [1])
            else:
                return np.array(list(dz) + [0])
        return np.array((list(A) + [1]))
    if repname == "tANM":
        dz = np.array(mol.nuclear_charges[:10]) - 6
        return np.dot(totalanmvectors.T, dz)
    if repname == "ANM":
        dz = np.array(mol.nuclear_charges[:10]) - 6
        return np.dot(anmvectors.T, dz)
    if repname == "cANM":
        dz = np.array(mol.nuclear_charges[:10]) - 6
        A = canonicalize(dz)
        return np.dot(anmvectors.T, A)
    if repname == "PCA":
        rep = get_representation(mol, "M3")
        return pca.transform(np.array([rep]))
    if repname == "M+CM":
        return np.array(
            list(get_representation(mol, "M")) + list(get_representation(mol, "CM"))
        )
    raise ValueError("Unknown representation")


pca = decomposition.PCA(n_components=10)
s = fetch_energies()
A = np.array([get_representation(get_compound(_), "M3") ** 2 for _ in s["label"]])
x_std = StandardScaler().fit_transform(A)
pca.fit(x_std)

try:
    q = pd.read_pickle("lcs.cache")
    lcs = {column: q[column] for column in q.columns}
except:
    lcs = {}
for rep in "CM M3 ANM".split():  # PCA cANM tANM ANM CM M M2 M3
    for propname in "twobodyE dressedtotalE atomicE electronicE".split():
        label = f"{propname}@{rep}"
        if label in lcs:
            continue
        lcs[label] = get_learning_curve(fetch_energies(), rep, propname)
pd.DataFrame(lcs).to_pickle("lcs.cache")
# endregion
# region alchemy
# find all labels which are strict alchemical enantiomers of each other
# strategy: test whether any symmetry opertations cancels the dz vectors
@functools.lru_cache(maxsize=1)
def find_all_strict_alchemical_enantiomers():
    def are_strict_alchemical_enantiomers(dz1, dz2):
        """Special to naphthalene in a certain atom order."""
        permA = [3, 2, 1, 0, 7, 6, 5, 4, 9, 8]
        permB = [7, 6, 5, 4, 3, 2, 1, 0, 8, 9]
        if sum(dz1[[9, 8]]) != 0:
            return False
        if sum(dz1[[0, 3, 4, 7]]) != 0:
            return False
        if sum(dz1[[1, 2, 5, 6]]) != 0:
            return False
        if np.abs(dz1 + dz2).sum() == 0:
            return True
        if np.abs(dz1 + dz2[permA]).sum() == 0:
            return True
        if np.abs(dz1 + dz2[permB]).sum() == 0:
            return True
        if np.abs(dz1 + dz2[permA][permB]).sum() == 0:
            return True
        return False

    # test cases
    topleft = np.array((0, 1, -1, -1, 0, 0, 0, 1, 0, 0))
    topright = np.array((0, -1, 1, 1, 0, 0, 0, -1, 0, 0))
    botleft = np.array((0, 1, -1, 1, 0, 0, 0, -1, 0, 0))
    botright = np.array((0, -1, 1, -1, 0, 0, 0, 1, 0, 0))
    assert are_strict_alchemical_enantiomers(topleft, topright)
    assert not are_strict_alchemical_enantiomers(topleft, botleft)
    assert are_strict_alchemical_enantiomers(botleft, botright)
    assert not are_strict_alchemical_enantiomers(topright, botright)

    rels = []
    for nbn, group in fetch_energies().groupby("nBN"):
        for idx, i in enumerate(group.label.values):
            dz1 = np.array([int(_) for _ in str(i)]) - 6
            for j in group.label.values[idx + 1 :]:
                dz2 = np.array([int(_) for _ in str(j)]) - 6
                if are_strict_alchemical_enantiomers(dz1, dz2):
                    rels.append({"one": i, "other": j})
    return pd.DataFrame(rels)


def are_strict_alchemical_enantiomers(dz1, dz2):
    """Special to naphthalene in a certain atom order."""
    permA = [3, 2, 1, 0, 7, 6, 5, 4, 9, 8]
    permB = [7, 6, 5, 4, 3, 2, 1, 0, 8, 9]
    if sum(dz1[[9, 8]]) != 0:
        return False
    if sum(dz1[[0, 3, 4, 7]]) != 0:
        return False
    if sum(dz1[[1, 2, 5, 6]]) != 0:
        return False
    if np.abs(dz1 + dz2).sum() == 0:
        return True
    if np.abs(dz1 + dz2[permA]).sum() == 0:
        return True
    if np.abs(dz1 + dz2[permB]).sum() == 0:
        return True
    if np.abs(dz1 + dz2[permA][permB]).sum() == 0:
        return True
    return False


# endregion

#%%
get_learning_curve(fetch_energies(), "CM", "atomicE", VDWBaseline)

# %%
# region visualisation
def learning_curves():
    markers = {
        "CM": "o",
        "M": "s",
        "MS": "v",
        "M2": ">",
        "M+CM": "^",
        "M3": "<",
        "ANM": "x",
        "tANM": "+",
        "cANM": "s",
        "PCA": "*",
    }
    order = "totalE atomicE electronicE nuclearE dressedtotalE dressedelectronicE twobodyE".split()
    for label, lc in lcs.items():
        propname, repname = label.split("@")
        if (
            label
            not in "atomicE@CM twobodyE@CM dressedtotalE@CM atomicE@ANM twobodyE@ANM".split()
        ):
            continue
        # if repname not in "CM".split():
        #    continue
        # if propname not in "twobodyE atomicE":
        #    continue
        plt.loglog(
            lc.index,
            lc.values * kcal,
            f"{markers[repname]}-",
            label=label,
            color=f"C{order.index(propname)}",
        )
    plt.legend(bbox_to_anchor=(0, 1, 1, 0.1), ncol=2)
    plt.xlabel("Training set size")
    plt.ylabel("MAE [kcal/mol]")


def distancediffplot(repname):
    for nBN, group in fetch_energies().groupby("nBN"):
        distances = []
        differences = []
        labels = group.label.values
        energies = group.electronicE.values
        for i, lbl in enumerate(labels):
            for j in range(i + 1, len(labels)):
                r1 = get_representation(get_compound(lbl), repname)
                r2 = get_representation(get_compound(labels[j]), repname)
                distances.append(np.linalg.norm(r1 - r2))
                differences.append(np.abs(energies[i] - energies[j]))
                if len(differences) > 1000:
                    break
            if len(differences) > 1000:
                break

        plt.scatter(distances, differences)


def baselinehistograms():
    opts = {"cumulative": True, "range": (-0.8, 0.8), "bins": 100}
    o = fetch_energies()["atomicE"]
    plt.hist(o - o.mean(), histtype="step", label="atomic", **opts)
    plt.hist(
        fetch_energies()["dressedtotalE"], histtype="step", label="dressed", **opts
    )
    plt.hist(
        fetch_energies()["twobodyE"], histtype="step", label="atom+twobody", **opts
    )
    plt.legend()


# endregion

#%%
# region baselines


class Baseline:
    def __init__(self, df):
        pass

    def fit(self, trainidx, propertyname):
        pass

    def transform(self, testidx):
        pass


class IdentityBaseline(Baseline):
    LABEL = ""

    def __init__(self, df):
        self._df = df

    def fit(self, trainidx, propertyname):
        return len(self._df) * 0

    def transform(self, testidx):
        return np.zeros(len(testidx))


class DressedBaseline(Baseline):
    LABEL = "Dressed"

    def __init__(self, df):
        self._df = df
        nC, nBN = [], []
        for label in df.label.values:
            label = str(label)
            nC.append(len([_ for _ in label if _ == "6"]))
            nBN.append((10 - nC[-1]) / 2)
        self._A = np.array((nC, nBN)).T

    def fit(self, trainidx, propertyname):
        properties = self._df[propertyname].values[trainidx]
        A = self._A[trainidx, :]
        self._coeff = np.linalg.lstsq(A, properties)[0]
        return np.dot(A, self._coeff)

    def transform(self, testidx):
        A = self._A[testidx, :]
        return np.dot(A, self._coeff)


class VDWBaseline:
    LABEL = "vdW"
    """ Hard-coded for the naphthalene case."""

    def __init__(self, df):
        self._df = df
        self._build_matrices(df.label.values)

    def _build_matrices(self, labels):
        coordinates = get_compound(6666666666).coordinates
        dm = ssd.squareform(ssd.pdist(coordinates))

        kinds = []
        for i in (1, 5, 6, 7):
            for j in (1, 5, 6, 7):
                a, b = sorted((i, j))
                kinds.append(f"{a}{b}")
        kinds = sorted(set(kinds))

        mat6 = np.zeros((len(labels), len(kinds)))
        mat12 = np.zeros((len(labels), len(kinds)))
        for labelidx in range(len(labels)):
            label = str(labels[labelidx]) + "11111111"
            for i in range(18):
                for j in range(i + 1, 18):
                    a, b = sorted((label[i], label[j]))
                    mat6[labelidx, kinds.index(f"{a}{b}")] += 1 / dm[i, j] ** 6
                    mat12[labelidx, kinds.index(f"{a}{b}")] += 1 / dm[i, j] ** 12

        self._kinds = kinds
        self._mat6 = mat6
        self._mat12 = mat12

    @staticmethod
    def _vdw_energy(params, label):
        coordinates = get_compound(6666666666).coordinates
        dm = ssd.squareform(ssd.pdist(coordinates))
        epsilons, sigmas = params[:4], params[4:]
        energy = 0
        label = str(label) + "11111111"
        order = "1567"
        for i in range(18):
            element_i_id = order.index(label[i])
            for j in range(i + 1, 18):
                element_j_id = order.index(label[j])
                epsilon = np.sqrt(
                    epsilons[element_i_id] ** 2 * epsilons[element_j_id] ** 2
                )
                sigma = (sigmas[element_i_id] ** 2 + sigmas[element_j_id] ** 2) / 2
                r = dm[i, j]
                energy += epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
        return energy

    @staticmethod
    def test():
        true_parameters = np.ones(8) + np.random.normal(size=8) / 10
        print(true_parameters)

        # calculate vdW energies
        labels = fetch_energies().label.values
        true_energies = np.array(
            [VDWBaseline._vdw_energy(true_parameters, _) for _ in labels]
        )

        # predict
        v = VDWBaseline()
        v.fit(labels, true_energies)

    def fit(self, trainidx, propertyname):
        hint = jnp.ones(8)
        self._expected = self._df[propertyname].values[trainidx]

        self._restrict = trainidx
        result = sco.differential_evolution(
            self._residuals_np,
            bounds=(
                (0.5, 2),
                (0.5, 2),
                (0.5, 2),
                (0.5, 2),
                (0.5, 2),
                (0.5, 2),
                (0.5, 2),
                (0.5, 2),
            ),
        )
        pred = self._predict_np(result.x)
        self._best_params = result.x
        return pred

    def transform(self, testidx):
        self._restrict = testidx
        return self._predict_np(self._best_params)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _predict_ad(self, parameters):
        order = "1567"
        sigmas = jnp.zeros(10)
        epsilons = jnp.zeros(10)
        for kidx, kind in enumerate(self._kinds):
            e1 = order.index(kind[0])
            e2 = order.index(kind[1])
            eps1 = parameters[e1] * parameters[e1]
            eps2 = parameters[e2] * parameters[e2]
            sig1 = parameters[e1 + 4] * parameters[e1 + 4]
            sig2 = parameters[e2 + 4] * parameters[e2 + 4]
            epsilons = epsilons.at[kidx].set(jnp.sqrt(eps1 * eps2))
            sigmas = sigmas.at[kidx].set((sig1 + sig2) / 2)
        pred = jnp.dot(self._mat12 * sigmas ** 12 - self._mat6 * sigmas ** 6, epsilons)
        return pred

    @functools.partial(jax.jit, static_argnums=(0,))
    def _residuals_ad(self, parameters):
        pred = self._predict_ad(parameters)
        return jnp.linalg.norm(pred - self._expected)

    def _predict_np(self, parameters):
        order = "1567"
        sigmas = np.zeros(10)
        epsilons = np.zeros(10)
        for kidx, kind in enumerate(self._kinds):
            e1 = order.index(kind[0])
            e2 = order.index(kind[1])
            eps1 = parameters[e1] * parameters[e1]
            eps2 = parameters[e2] * parameters[e2]
            sig1 = parameters[e1 + 4] * parameters[e1 + 4]
            sig2 = parameters[e2 + 4] * parameters[e2 + 4]
            epsilons[kidx] = np.sqrt(eps1 * eps2)
            sigmas[kidx] = (sig1 + sig2) / 2
        pred = np.dot(
            self._mat12[self._restrict, :] * sigmas ** 12
            - self._mat6[self._restrict, :] * sigmas ** 6,
            epsilons,
        )
        return pred

    def _residuals_np(self, parameters):
        return np.linalg.norm(self._predict_np(parameters) - self._expected)


# endregion

#%%
# @functools.lru_cache(maxsize=1)
#     def twobody_subsets():

#         dm =
#         subsets = []
#         for distance in set(dm.reshape(-1)):
#             if distance < 1e-3:
#                 continue
#             subset = []
#             for i in range(10):
#                 for j in range(i+1, 10):
#                     if abs(dm[i,j]  - distance) > 1e-3:
#                         continue
#                     subset.append((i+1, j+1))
#             subsets.append(subset)
#         return subsets

#     def bonds_from_subset(label, subset):
#         # target structure
#         kinds = {}
#         for i in (5,6,7,):
#             for j in (5,6,7):
#                 a, b = sorted((i, j))
#                 kinds[f"{a}{b}"] = 0
#         bonds = subset

#         # build entries
#         label = str(label)
#         for i, j in bonds:
#             a, b = sorted((label[i-1], label[j-1]))
#             kinds[f"{a}{b}"] += 1

#         return np.array([kinds[_] for _ in sorted(kinds.keys())])

#     def twobody_from_label(label):
#         coeffs = [bonds_from_subset(label, _) for _ in twobody_subsets()]
#         label = str(label)
#         kinds = {'15': 0, '16': 0, '17': 0}
#         for a in label[2:]:
#             kinds[f"1{a}"] += 1
#         coeffs.append(np.array([kinds[_] for _ in sorted(kinds.keys())]))
#         return np.concatenate(coeffs)


#     def fit(X, Y):
#         components =

#         BONDS = np.array([twobody_from_label(_) for _ in df['label']])

#         nC = 10-2*df.nBN.values
#         nBN = df.nBN.values
#         ATOMS = np.array((nC, nBN)).T

#         A = np.hstack((ATOMS,BONDS))
#         coeff = np.linalg.lstsq(A, df.totalE.values)

#         return df.totalE.values - np.dot(A, coeff[0])


#     def fit(X, Y):

#     df['twobodyE'] = leftover(df)
#     def transform(X):

#%%
xs = (8, 32, 128, 512, 2048)
cm = np.array((0.089137, 0.069463, 0.041231, 0.024067, 0.013137))
cmbl = np.array((0.031417, 0.023576, 0.019008, 0.014052, 0.009398))
plt.loglog(xs, cm * kcal, "o-", label="CM atomisation E")
plt.loglog(xs, cmbl * kcal, "s-", label="CM atomisation E+vdW baseline")
plt.legend()
plt.xlabel("Training set size")
plt.ylabel("MAE [kcal/mol]")
# %%
