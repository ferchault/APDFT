#!/usr/bin/env python
# Story APDFT#248
# Question APDFT#249: Are ANM locally optimal representations, i.e. is there an orthogonal transformation such that learning naphthalene gets better?
# References:
#  - Generalized Euler angles, 10.1063/1.1666011

#%%
# AD
# keep jax config first
from jax.config import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp

# Test for double precision import
x = jax.random.uniform(jax.random.PRNGKey(0), (1,), dtype=jnp.float64)
assert x.dtype == jnp.dtype("float64"), "JAX not in double precision mode"

import numpy as np
import matplotlib.pyplot as plt

anmhessian = np.loadtxt("hessian.txt")
_, anmvectors = np.linalg.eig(anmhessian)

# %%
def gea_matrix_a(angles):
    """
    Generalized Euler Angles
    Return the parametric angles described on Eqs. 15-19 from the paper:
    Generalization of Euler Angles to N-Dimensional Orthogonal Matrices
    David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
    Journal of Mathematical Physics 13, 528 (1972)
    doi: 10.1063/1.1666011
    """
    n = len(angles)
    matrix_a = jnp.eye(n)

    for i in range(n - 1):
        matrix_a = matrix_a.at[i, i].set(jnp.cos(angles[i]))
        matrix_a = matrix_a.at[i, n - 1].set(jnp.tan(angles[i]))
        for j in range(i + 1):
            matrix_a = matrix_a.at[i, n - 1].mul(jnp.cos(angles[j]))

    for i in range(n):
        for k in range(n - 1):
            if i > k:
                matrix_a = matrix_a.at[i, k].set(
                    -jnp.tan(angles[i]) * jnp.tan(angles[k])
                )
                for l in range(k, i + 1):
                    matrix_a = matrix_a.at[i, k].mul(jnp.cos(angles[l]))

    matrix_a = matrix_a.at[n - 1, n - 1].set(jnp.tan(angles[n - 1]))
    for j in range(n):
        matrix_a = matrix_a.at[n - 1, n - 1].mul(jnp.cos(angles[j]))

    return matrix_a


def gea_orthogonal_from_angles(angles_list):
    """
    Generalized Euler Angles
    Return the orthogonal matrix from its generalized angles
    Generalization of Euler Angles to N-Dimensional Orthogonal Matrices
    David K. Hoffman, Richard C. Raffenetti, and Klaus Ruedenberg
    Journal of Mathematical Physics 13, 528 (1972)
    doi: 10.1063/1.1666011
    :param angles_list: List of angles, for a n-dimensional space the total number
                        of angles is k*(k-1)/2
    """

    b = jnp.eye(2)
    n = int(jnp.sqrt(len(angles_list) * 8 + 1) / 2 + 0.5)
    tmp = jnp.array(angles_list)

    # For SO(k) there are k*(k-1)/2 angles that are grouped in k-1 sets
    # { (k-1 angles), (k-2 angles), ... , (1 angle)}
    for i in range(1, n):
        angles = jnp.concatenate((jnp.array(tmp[-i:]), jnp.array([jnp.pi / 2])))
        tmp = tmp[:-i]
        ma = gea_matrix_a(angles)  # matrix i+1 x i+1
        b = jnp.dot(b, ma.T).T
        # We skip doing making a larger matrix for the last iteration
        if i < n - 1:
            c = jnp.eye(i + 2, i + 2)
            c = c.at[:-1, :-1].set(b)
            b = c
    return b


gea_orthogonal_from_angles(np.zeros(45))
# %%
