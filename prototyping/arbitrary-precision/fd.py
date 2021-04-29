#!/usr/bin/env python
#%%
import mpmath
import click
import json

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


def energy(lval, distance):
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
            (mpmath.mpf(0), mpmath.mpf(0), mpmath.mpf(distance)),
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


@click.command()
@click.option("--distance", default="1.4")
@click.option("--dps", default=100)
@click.option("--orders", default=40)
@click.option("--deltaexp", default=65)
def main(distance, dps, orders, deltaexp):
    meta = {
        "distance": distance,
        "dps": dps,
        "ref": "H2",
        "target": "He",
        "orders": orders,
        "deltaexp": deltaexp,
    }

    mpmath.mp.dps = dps
    coeffs = mpmath.taylor(
        lambda _: energy(_, distance),
        0,
        orders,
        direction=1,
        h=mpmath.mpf(f"1e-{deltaexp}"),
        addprec=100,
    )
    total = mpmath.mpf("0")
    for order, c in enumerate(coeffs):
        total += c
        thisdict = {"order": order, "coefficient": str(c), "total": str(total)}
        thisdict.update(meta)
        print(json.dumps(thisdict))

    ref = energy(mpmath.mpf("0.0"))
    target = energy(mpmath.mpf("1.0"))
    print("ref", ref)
    print("target", target)
    thisdict = {"ref": str(ref), "target": str(target)}
    thisdict.update(meta)
    print(json.dumps(thisdict))


if __name__ == "__main__":
    main()
