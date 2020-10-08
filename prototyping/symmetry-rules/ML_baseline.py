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
import jax
import jax.numpy as jnp

# Helper
import pandas as pd
import matplotlib.pyplot as plt
import functools
import requests
import numpy as np
import itertools as it
import scipy.spatial.distance as ssd

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

    # dressed atom
    nC = 10 - 2 * df.nBN.values
    nBN = df.nBN.values
    A = np.array((nC, nBN)).T
    coeff = np.linalg.lstsq(A, df.electronicE.values)
    df["dressedelectronicE"] = df.electronicE.values - np.dot(A, coeff[0])
    coeff = np.linalg.lstsq(A, df.totalE.values)
    df["dressedtotalE"] = df.totalE.values - np.dot(A, coeff[0])

    df = df[
        "label nBN totalE nuclearE electronicE atomicE dressedtotalE dressedelectronicE".split()
    ].copy()
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


def get_learning_curve(df, repname, propname):
    X = np.array(
        [get_representation(get_compound(_), repname) for _ in df.label.values]
    )
    Y = df[propname].values

    rows = []
    totalidx = np.arange(len(X), dtype=np.int)
    for sigma in 2.0 ** np.arange(-2, 10):
        print(sigma)
        Ktotal = qml.kernels.gaussian_kernel(X, X, sigma)

        for lval in (1e-7, 1e-9, 1e-11, 1e-13):
            for ntrain in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
                maes = []
                for k in range(5):
                    np.random.shuffle(totalidx)
                    train, test = totalidx[:ntrain], totalidx[ntrain:]

                    K_subset = Ktotal[np.ix_(train, train)]
                    K_subset[np.diag_indices_from(K_subset)] += lval
                    alphas = qml.math.cho_solve(K_subset, Y[train])

                    K_subset = Ktotal[np.ix_(train, test)]
                    pred = np.dot(K_subset.transpose(), alphas)
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


# endregion

#%%
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


#%%


class VDWBaseline:
    """ Hard-coded for the naphthalene case."""

    def fit(self, labels, properties):
        # parameters = np.array((1,1,1,1,2,2,2,2), dtype=np.float)
        parameters = np.array(
            [
                0.06788786,
                0.9530318,
                0.12076921,
                0.99212635,
                1.3718097,
                1.7186185,
                1.3297557,
                1.953924,
            ]
        )
        coordinates = get_compound(6666666666).coordinates
        dm = ssd.squareform(ssd.pdist(coordinates))

        def _vdwonelabel(epsilons, sigmas):
            print("new onelabel compile")
            energy = 0
            for i in range(18):
                ei = epsilons[i] * epsilons[i]
                oi = sigmas[i] * sigmas[i]
                for j in range(i + 1, 18):
                    ej = epsilons[j] * epsilons[j]
                    oj = sigmas[j] * sigmas[j]
                    e = jnp.sqrt(ei * ej)
                    o = (oi + oj) / 2
                    q = (o / dm[i, j]) ** 6
                    energy += e * (q ** 2 - q)
            return energy

        _vdwonelabel = jax.jit(_vdwonelabel)

        def _vdw(Q, labels, dm):
            energies = jnp.zeros(len(labels))
            print("new vdw compile")
            for labelidx in range(len(labels)):
                epsilons = Q[labels[labelidx, :18]]
                sigmas = Q[labels[labelidx, 18:]]
                energies = energies.at[labelidx].set(_vdwonelabel(sigmas, epsilons))

            return jnp.linalg.norm(energies - properties) / energies.shape[0]

        _vdw = jax.jit(_vdw, static_argnums=(1, 2))
        gradf = jax.grad(_vdw)
        for i in range(20):
            # print (parameters)
            print("iteration", _vdw(parameters, labels, dm))
            # print ("v")
            g = gradf(parameters, labels, dm)
            g /= np.linalg.norm(q)
            g *= 0.1
            parameters -= g
            # print ("^")
        return parameters


def label2es(label):
    label = str(label)
    result = np.zeros(18 * 2, dtype=int)
    for i in range(10):
        result[i] = {"5": 1, "6": 2, "7": 3}[label[i]]
        result[i + 18] = {"5": 1 + 4, "6": 2 + 4, "7": 3 + 4}[label[i]]
    return result


b = VDWBaseline()
labels = np.array([label2es(_) for _ in fetch_energies().label.head(3).values])
properties = fetch_energies().atomicE.head(3).values
b.fit(labels, properties)

# %%
def build_matrices(labels):
    coordinates = get_compound(6666666666).coordinates
    dm = ssd.squareform(ssd.pdist(coordinates))

    kinds = []
    for i in (
        1,
        5,
        6,
        7,
    ):
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
    return kinds, mat6, mat12


kinds, mat6, mat12 = build_matrices(fetch_energies().label.values[100:])

#%%
parameters = jnp.array(
    [
        0.06788786,
        0.9530318,
        0.12076921,
        0.99212635,
        1.3718097,
        1.7186185,
        1.3297557,
        1.953924,
    ]
)


@jax.jit
def predict(parameters):
    order = "1567"
    sigmas = jnp.zeros(10)
    epsilons = jnp.zeros(10)
    for kidx, kind in enumerate(kinds):
        e1 = order.index(kind[0])
        e2 = order.index(kind[1])
        eps1 = parameters[e1] * parameters[e1]
        eps2 = parameters[e2] * parameters[e2]
        sig1 = parameters[e1 + 4] * parameters[e1 + 4]
        sig2 = parameters[e2 + 4] * parameters[e2 + 4]
        epsilons = epsilons.at[kidx].set(jnp.sqrt(eps1 * eps2))
        sigmas = sigmas.at[kidx].set((sig1 + sig2) / 2)
    pred = jnp.dot(mat12 * sigmas ** 12 - mat6 * sigmas ** 6, epsilons)
    return pred


@jax.jit
def residuals(parameters):
    expected = fetch_energies().atomicE.values[100:]
    pred = predict(parameters)
    return jnp.linalg.norm(pred - expected)


x0 = jnp.array([2, 2, 2, 2, 0.5, 0.5, 0.5, 0.5])
result = sco.minimize(residuals, x0=jnp.ones(8), jac=jax.grad(residuals), method="CG")
print(result.x, result.fun)

#%%
q = fetch_energies().atomicE.values
qq = {
    "range": (0, 0.7),
    "bins": 100,
    "cumulative": True,
    "histtype": "step",
    "density": True,
}
plt.hist(np.abs(q - np.average(q)), label="atomic", **qq)
q = predict(
    [
        1.3210275,
        1.4704485,
        1.2422168,
        1.0012943,
        0.7454262,
        1.0606761,
        1.1231637,
        1.1754208,
    ]
)
plt.hist(np.abs(q - np.average(q)), label="reduced", **qq)
plt.legend()

#%%


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
