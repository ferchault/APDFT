#!/usr/bin/env python
# Story APDFT#248
# Question APDFT#249: Are ANM locally optimal representations, i.e. is there an orthogonal transformation such that learning naphthalene gets better?
# References:
#  - Generalized Euler angles, 10.1063/1.1666011

#%%
# region imports
# AD
# keep jax config first
from jax.config import config
from jax.experimental import loops

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)
import jax
import jax.numpy as jnp

# Test for double precision import
x = jax.random.uniform(jax.random.PRNGKey(0), (1,), dtype=jnp.float64)
assert x.dtype == jnp.dtype("float64"), "JAX not in double precision mode"

import qml
import sys
import requests

import numpy as np
import matplotlib.pyplot as plt
import functools
import pandas as pd
import scipy
import multiprocessing as mp

anmhessian = np.loadtxt("hessian.txt")
_, anmvectors = np.linalg.eig(anmhessian)

sys.path.append("..")
import mlmeta
import sklearn.metrics as skm

# endregion
# %%
# region dataset
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


def get_rep(transformation, mol):
    dz = jnp.array(mol.nuclear_charges[:10]) - 6
    return jnp.dot(jnp.dot(transformation, anmvectors.T), dz)


#%%
def ds(reps):
    nmols = len(reps)
    with loops.Scope() as s:
        s.K = jnp.zeros((nmols, nmols))
        for a in s.range(nmols * nmols + 1):
            i = (
                nmols
                - 1
                - jnp.floor((-1 + jnp.sqrt((2 * nmols + 1) ** 2 - 8 * (a + 1))) / 2)
            ).astype(int)
            j = (a - i * (2 * nmols - i - 1) // 2).astype(int)
            d = jax.lax.cond(
                i == j,
                lambda _: 0.0,
                lambda _: jnp.linalg.norm(reps[_[0]] - reps[_[1]]),
                (i, j),
            )
            s.K = s.K.at[i, j].set(d)
            s.K = s.K.at[j, i].set(d)
        K = s.K
    return K


# %%


# @jax.partial(jax.jit, static_argnums=(0, 2))
def get_lc_endpoint(df, transformation, Y):
    X = jnp.array([get_rep(transformation, get_compound(_)) for _ in df.label.values])

    totalidx = np.arange(len(X), dtype=np.int)
    maes = []

    dscache = ds(X)
    for sigma in 2.0 ** np.arange(-2, 10):
        inv_sigma = -0.5 / (sigma * sigma)
        Ktotal = jnp.exp(dscache * inv_sigma)

        mae = []
        for ntrain in (int(0.9 * len(X)),):
            for k in range(1):
                np.random.shuffle(totalidx)
                train, test = totalidx[:ntrain], totalidx[ntrain:]

                lval = 10 ** -10
                K_subset = Ktotal[np.ix_(train, train)]
                K_subset = K_subset.at[np.diag_indices_from(K_subset)].add(lval)
                step1 = jax.scipy.linalg.cho_factor(K_subset)
                alphas = jax.scipy.linalg.cho_solve(step1, Y[train])
                # alphas = positive_definite_solve(K_subset, Y[train])

                K_subset = Ktotal[np.ix_(train, test)]
                pred = jnp.dot(K_subset.transpose(), alphas)
                actual = Y[test]

                thismae = jnp.abs(pred - actual).mean()
                mae.append(thismae)
        maes.append(jnp.average(mae))
    return jnp.min(jnp.array(maes))


# %%
def get_transformed_mae(sigma, dscache, trainidx, testidx, Y):
    inv_sigma = -0.5 / (sigma * sigma)
    Ktotal = np.exp(dscache * inv_sigma)

    ntrain = len(trainidx)

    lval = 10 ** -10
    K_subset = Ktotal[np.ix_(trainidx, trainidx)]
    K_subset[np.diag_indices_from(K_subset)] += lval
    step1 = scipy.linalg.cho_factor(K_subset)
    alphas = scipy.linalg.cho_solve(step1, Y[trainidx])

    K_subset = Ktotal[np.ix_(trainidx, testidx)]
    pred = np.dot(K_subset.transpose(), alphas)
    actual = Y[testidx]

    return np.abs(pred - actual).mean()


def transformedkrr(df, trainidx, testidx, transform, Y):
    X = np.array([get_rep(transform, get_compound(_)) for _ in df.label.values])

    dscache = np.array(skm.pairwise_distances(X, n_jobs=-1))
    sigmas = 2.0 ** np.arange(-2, 10)
    maes = [
        get_transformed_mae(sigma, dscache, trainidx, testidx, Y) for sigma in sigmas
    ]

    return np.min(np.array(maes))


def optimize_representation(ntrain=40):
    print("fetch")
    xs = np.arange(len(fetch_energies()))
    np.random.shuffle(xs)
    trainidx, testidx = xs[:ntrain], xs[ntrain:]
    traindf = fetch_energies().iloc[trainidx].copy()
    Y = fetch_energies().atomicE.values

    def inlinewrapper(transform):
        transform = transform.reshape(10, 10)
        return get_lc_endpoint(traindf, transform, Y)

    valgrad = jax.value_and_grad(inlinewrapper)

    print("start")
    transform = jnp.identity(10).reshape(-1)

    with mp.Pool() as pool:
        for i in range(3):
            optmae, optgrad = valgrad(transform)
            krrmae = transformedkrr(
                fetch_energies(),
                trainidx,
                testidx,
                np.asarray(transform).reshape(10, 10),
                Y,
            )
            transform -= optgrad * 0.1
            print(i, optmae, np.linalg.norm(optgrad), krrmae)


mlmeta.profile(optimize_representation)
#%%
# %%
