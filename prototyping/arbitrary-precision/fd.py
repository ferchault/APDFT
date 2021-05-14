#!/usr/bin/env python
#%%
import mpmath
import configparser
import json
from multiprocessing import Pool, pool
import multiprocessing as mp

from RHF import *
from matrices import *
from integrals import *
from basis import *
from molecules import *

# region
import functools


@functools.lru_cache(maxsize=1)
def get_ee(bs):
    return EE_list(bs)


def energy(lval, dps, distance):
    bs = sto3g_H2
    ee = get_ee(bs)
    print(str(lval), dps)
    mpmath.mp.dps = dps
    mol = [
        Atom(
            "H",
            (mpmath.mpf(0), mpmath.mpf(0), mpmath.mpf(0)),
            mpmath.mpf("2.0") - lval,
            ["1s"],
        ),
        Atom(
            "H",
            (mpmath.mpf(0), mpmath.mpf(0), mpmath.mpf(distance)),
            mpmath.mpf("1.0") + lval,
            ["1s"],
        ),
    ]
    N = 2
    K = bs.K
    S = S_overlap(bs)
    X = X_transform(S)
    Hc = H_core(bs, mol)
    # ee = get_ee(bs)
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
    return (lval, dps), mpmath.chop(energy_el(P, F, Hc))


def main(distance, dps, orders, deltaexp):
    mp.set_start_method("spawn")
    mpmath.mp.dps = dps

    direction = 1
    around = 0
    args = around, orders
    kwargs = {
        "h": mpmath.mpf(f"1e-{deltaexp}"),
        "addprec": 100,
        "direction": direction,
        "method": "step",
    }

    pos = []
    coeffs = mpmath.taylor(
        lambda _: pos.append((_, mpmath.mp.dps)) or 1, *args, **kwargs
    )

    content = [(*_, distance) for _ in pos]
    with Pool(40) as p:
        res = p.starmap(energy, tqdm.tqdm(content, total=len(content)), chunksize=1)
    res = dict(res)

    coeffs = mpmath.taylor(lambda _: res[(_, mpmath.mp.dps)], *args, **kwargs)

    total = mpmath.mpf("0")
    vals = []
    ref = energy(mpmath.mpf("0.0"), dps, distance)[1]
    target = energy(mpmath.mpf("1.0"), dps, distance)[1]
    print("ref", ref)
    print("target", target)

    for order, c in enumerate(coeffs):
        total += c
        vals.append(total)
        thisdict = {
            "order": order,
            "coefficient": str(c),
            "total": str(total),
            "error": str(total - target),
        }
        thisdict.update(meta)
        print(json.dumps(thisdict))

    # thisdict = {"ref": str(ref), "target": str(target)}
    # thisdict.update(meta)
    # print(json.dumps(thisdict))
    return vals


if __name__ == "__main__":
    main()