#!/usr/bin/env python
#%%
import mpmath

mpmath.mp.dps = 100

from RHF import *
from matrices import *
from integrals import *
from basis import *
from molecules import *


def energy(lval):
    return 2 * lval ** 2 + lval ** 3


def cfd(h, n):
    value = mpmath.mpf("0.0")
    for i in range(0, n + 1):
        binom = mpmath.binomial(n, i)
        xval = (n / 2 - i) * h
        value += energy(xval) * binom * (-1) ** i
    return value / h ** n / mpmath.fac(n)


# region
import functools
@functools.lru_cache(maxsize=1)
def get_ee(bs):
    return EE_list(bs)

def energy(lval):
    print(lval)
    mol = [
        Atom(
            "H",
            (mpmath.mpf(0), mpmath.mpf(0), mpmath.mpf(0)),
            mpmath.mpf("1.0") + lval,
            ["1s"],
        ),
        Atom(
            "H",
            (mpmath.mpf(0), mpmath.mpf(0), mpmath.mpf("1.4")),
            mpmath.mpf("1.0") - lval,
            ["1s"],
        ),
    ]
    bs = sto3g_H2
    N = 2
    K = bs.K
    S = S_overlap(bs)
    X = X_transform(S)
    Hc = H_core(bs, mol)
    ee = get_ee(bs)
    Pnew = mpmath.matrix(K, K)
    P = mpmath.matrix(K, K)
    iter = 1
    while True:
        Pnew, F, E = RHF_step(bs, mol, N, Hc, X, P, ee, False)
        dp = delta_P(P, Pnew)
        if dp < mpmath.mpf(f"1e-{mpmath.mp.dps-3}"):
            break
        P = Pnew
        iter += 1
    return energy_el(P, F, Hc)


coeffs = mpmath.taylor(energy, 0, 40, direction=0, h=mpmath.mpf("1e-65"), addprec=100)
ref = energy(mpmath.mpf("0.0"))
target = energy(mpmath.mpf("1.0"))
total = mpmath.mpf("0")
for order, c in enumerate(coeffs):
    total += c
    print(order, c, total)
print("ref", ref)
print("target", target)
