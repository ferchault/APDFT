#!/usr/bin/env python
#%%
import pickle
import mpmath
import configparser
import hashlib
import click
from multiprocessing import Pool, pool
import multiprocessing as mp
from RHF import *
from matrices import *
from integrals import *
from basis import *

# region
import functools
import basis_set_exchange as bse


@functools.lru_cache(maxsize=1)
def get_ee(basis, cachename):
    K = basis.K

    EE = mpmath.matrix(K, K, K, K)
    with open(cachename + "-ee.cache", "rb") as fh:
        results = pickle.load(fh)
    for result in results:
        i, j, k, l, E = result
        EE[i, j, k, l] = E
    return EE


def build_system(config, lval):
    reference_Zs = config["meta"]["reference"].strip().split()
    basis_Zs = config["meta"]["basis"].strip().split()
    target_Zs = config["meta"]["target"].strip().split()
    coords = config["meta"]["coords"].strip().split("\n")

    mol = []

    N = 0
    for ref, tar, bas, coord in zip(reference_Zs, target_Zs, basis_Zs, coords):
        N += int(ref)
        element = bse.lut.element_data_from_Z(int(bas))[0].capitalize()
        Z = mpmath.mpf(tar) * lval + (1 - lval) * mpmath.mpf(ref)
        atom = Atom(element, tuple([mpmath.mpf(_) for _ in coord.split()]), Z, bas)
        mol.append(atom)
    bs = Basis(config["meta"]["basisset"], mol)

    return mol, bs, N


def cache_EE_integrals(dps, config):
    cachename = config["meta"]["cache"] + "-ee.cache"
    if os.path.exists(cachename):
        return
    prevdps = mpmath.mp.dps
    mpmath.mp.dps = dps
    mol, bs, N = build_system(config, 0)
    ee = EE_list(bs)
    with open(cachename, "wb") as fh:
        pickle.dump(ee, fh)
    mpmath.mp.dps = prevdps


def energy(lval, dps, config):
    mpmath.mp.dps = dps
    mol, bs, N = build_system(config, lval)
    ee = get_ee(bs, config["meta"]["cache"])
    K = bs.K
    S = S_overlap(bs)
    X = X_transform(S)
    Hc = H_core(bs, mol)
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
        if iter > 10000:
            raise ValueError("Unconverged")
    return (lval, dps), (mpmath.chop(energy_el(P, F, Hc)), iter)


@click.command()
@click.argument("infile")
@click.argument("outfile")
def main(infile, outfile):
    config = configparser.ConfigParser()
    with open(infile) as fh:
        config.read_file(fh)
    with open(infile, "rb") as fh:
        config["meta"]["cache"] = hashlib.sha256(fh.read()).hexdigest()

    mp.set_start_method("spawn")
    dps = config["meta"].getint("dps")
    mpmath.mp.dps = dps

    direction = config["meta"]["direction"]
    direction = {"forward": 1, "central": 0, "backward": -1}[direction]
    around = 0
    args = around, config["meta"].getint("orders")

    kwargs = {
        "h": mpmath.mpf(f'1e-{config["meta"].getint("deltalambda")}'),
        "direction": direction,
        "method": "step",
    }

    pos = []
    _ = mpmath.taylor(lambda _: pos.append((_, mpmath.mp.dps)) or 1, *args, **kwargs)
    maxdps = max([_[1] for _ in pos])
    cache_EE_integrals(maxdps, config)

    content = [(*_, config) for _ in pos]
    with Pool(os.cpu_count()) as p:
        res = p.starmap(energy, tqdm.tqdm(content, total=len(content)), chunksize=1)
    res = dict(res)
    config.add_section("singlepoints")
    for c, item in enumerate(res.items()):
        k, v = item
        lval, d = k
        v, iter = v
        config["singlepoints"][f"pos-{c}"] = str(lval)
        config["singlepoints"][f"dps-{c}"] = str(d)
        config["singlepoints"][f"energy-{c}"] = str(v)
        config["singlepoints"][f"iter-{c}"] = str(iter)

    coeffs = mpmath.taylor(lambda _: res[(_, mpmath.mp.dps)][0], *args, **kwargs)

    total = mpmath.mpf("0")
    config.add_section("endpoints")
    ref = energy(mpmath.mpf("0.0"), dps, config)[1][0]
    target = energy(mpmath.mpf("1.0"), dps, config)[1][0]
    config["endpoints"]["reference"] = str(ref)
    config["endpoints"]["target"] = str(target)

    # prepare output
    config.add_section("coefficients")
    config.add_section("totals")
    config.add_section("errors")
    for order, c in enumerate(coeffs):
        total += c
        config["coefficients"][f"order-{order}"] = str(c)
        config["totals"][f"order-{order}"] = str(total)
        config["errors"][f"order-{order}"] = str(total - target)
    with open(outfile, "w") as fh:
        config.write(fh)


if __name__ == "__main__":
    main()
