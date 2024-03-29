#!/usr/bin/env python
#%%
import pickle
import mpmath
import configparser
import hashlib
import click
import pyscf
import basis_set_exchange as bse
from multiprocessing import Pool, Value, pool
import multiprocessing as mp
from RHF import *
from matrices import *
from integrals import *
from basis import *

# region
import functools
import basis_set_exchange as bse


@functools.lru_cache(maxsize=1)
def get_ee(cachename):
    with open(cachename + "-ee.cache", "rb") as fh:
        results = pickle.load(fh)
    K = 0
    for result in results:
        i, j, k, l, E = result
        K = max(K, max(i, j, k, l))
    K = K + 1
    EE = mpmath.matrix(K, K, K, K)
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


class DIIS:
    def __init__(self, maxlen, S):
        self._maxlen = maxlen
        self._S = S
        self._P = []
        self._F = []
        self._e = []

    def update(self, F, P):
        # print("F", F)
        F = from_np(F.copy())
        P = from_np(P.copy())
        self._F.append(F)
        self._P.append(P)
        sdf = self._S * P * F
        e = sdf.T - sdf
        self._e.append(to_np(e).reshape(-1))
        if len(self._P) > self._maxlen:
            self._P = self._P[1:]
            self._F = self._F[1:]
            self._e = self._e[1:]
            e = np.array(self._e)
            N = len(self._e)
            A = np.zeros((N + 1, N + 1)) - 1
            for i in range(N):
                for j in range(N):
                    A[i, j] = np.dot(e[i], e[j])
            A[N, N] = 0
            b = np.zeros(N + 1)
            b[N] = -1

            ci, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
            # print(sum(ci))
            Fprime = to_np(F) * 0
            for i in range(N):
                Fprime += ci[i] * to_np(self._F[i])

            Fprime = to_np(Fprime)
        else:
            Fprime = to_np(F)

        return Fprime


def energy(lval, dps, config, guess):
    mpmath.mp.dps = dps
    mol, bs, N = build_system(config, lval)
    ee = get_ee(config["meta"]["cache"])
    K = bs.K
    S = S_overlap(bs)
    X = X_transform(S)
    Hc = H_core(bs, mol)
    Pnew = mpmath.matrix(K, K)
    P = from_np(guess)
    iter = 1
    manager = DIIS(8, S)
    while True:
        # print("P", P)

        # G = G_ee(bs, mol, P, ee)
        # print("T", T_kinetic(bs))
        # print("V", Hc - T_kinetic(bs))
        # print("Hc", Hc)
        # print("Veff", G)
        # print("S", S)
        Pnew, F, E = RHF_step(bs, mol, N, Hc, X, P, ee, False, manager)

        dp = delta_P(P, Pnew)
        # print("##", mpmath.nstr(dp), mpmath.nstr(mpmath.mpf(f"1e-{mpmath.mp.dps-3}")))
        if dp < mpmath.mpf(f"1e-{mpmath.mp.dps-3}"):
            break
        P = Pnew
        iter += 1
        if iter > 50000:
            got = mpmath.nstr(dp)
            expected = mpmath.nstr(mpmath.mpf(f"1e-{mpmath.mp.dps-3}"))
            raise ValueError(f"Unconverged: {got} instead of {expected}")
    return (lval, dps), (mpmath.chop(energy_el(P, F, Hc)), iter)


def compare_to_pyscf(config):
    mol, bs, N = build_system(config, 0)
    ee = get_ee(config["meta"]["cache"])
    basis_Zs = config["meta"]["basis"].strip().split()
    coords = config["meta"]["coords"].strip().split("\n")
    bsname = config["meta"]["basisset"]

    atomspec = []
    for Z, cs in zip(basis_Zs, coords):
        parts = [float(_) * 0.52917721067 for _ in cs.split()]
        atomspec.append(f"{Z} {parts[0]} {parts[1]} {parts[2]}")
    atomspec = ";".join(atomspec)

    basisspec = {}
    for nuclear_number in set(basis_Zs):
        basisspec[nuclear_number] = bse.get_basis(bsname, nuclear_number, fmt="nwchem")

    mol = pyscf.gto.Mole()
    mol.atom = atomspec
    mol.basis = basisspec
    mol.build()

    c = pyscf.scf.RHF(mol)
    c.run()
    S_this = to_np(S_overlap(bs)).astype(np.float64)
    S_pyscf = mol.get_ovlp()

    # from functools import reduce

    # P = mol.make_rdm1()
    # F = c.get_hcore() + c.get_veff(c.mol, P)
    # print("REFREFREF v")
    # print("P", P)
    # print("T", mol.intor_symmetric("int1e_kin"))
    # print("V", mol.intor_symmetric("int1e_nuc"))
    # print("F", F)
    # print("Hc", c.get_hcore())
    # print("Veff", c.get_veff(c.mol, P))
    # print("S", S_pyscf)
    # sdf = reduce(np.dot, (S_pyscf, P, F))
    # print("SD", np.dot(S_pyscf, P))
    # print("SDF", sdf)
    # print("REFREFREF ^")
    # e = sdf.T.conj() - sdf
    # print("EEEE", np.linalg.norm(e))

    if not np.allclose(S_this, S_pyscf):
        raise ValueError("Reference overlap does not agree with pyscf")

    return mol.make_rdm1()


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

    # compare EE integrals to pyscf
    dm = compare_to_pyscf(config)
    # print(dm)

    content = [(*_, config, dm) for _ in pos]
    with Pool(os.cpu_count()) as p:
        res = p.starmap(energy, tqdm.tqdm(content, total=len(content)), chunksize=1)
    # res = [energy(*_) for _ in content]
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
    ref = energy(mpmath.mpf("0.0"), dps, config, dm)[1][0]
    target = energy(mpmath.mpf("1.0"), dps, config, dm)[1][0]
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
