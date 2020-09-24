#!/usr/bin/env python

#%%
# atoms 012 345 in benzene
import numpy as np
import itertools as it

upz = np.array((-1, 1, 1, -1, 0, 0))
dnz = -upz


def array_to_dict(ohs):
    res = {}
    for count, integral in ohs:
        if count == 0:
            continue
        if integral not in res:
            res[integral] = 0
        res[integral] += count
    return res


def simplify(lhs, rhs):
    lhs = array_to_dict(lhs)
    rhs = array_to_dict(rhs)

    # cancel identical terms
    for lkey in list(lhs.keys()):
        if lkey in rhs:
            lhs[lkey] -= rhs[lkey]
            del rhs[lkey]
            if lhs[lkey] == 0:
                del lhs[lkey]

    return lhs, rhs


def canonical_label(label):
    parts = label.split("|")
    dvpos = int(parts[0][1:])
    pderivs = [int(_) for _ in parts[1].split(",")]
    nderivs = sorted([(_ - dvpos) % 6 for _ in pderivs])
    onederivs = [str(_) for _ in nderivs]
    onelabel = f"I0|{','.join(onederivs)}"

    otherderivs = sorted([6 - _ for _ in nderivs])
    otherderivs = [str(_) for _ in otherderivs]
    otherlabel = f"I0|{','.join(otherderivs)}"

    return sorted((otherlabel, onelabel))[0]


def first_order():
    lhs = []
    rhs = []
    for j in range(6):
        for i in range(6):
            label = canonical_label(f"I{j}|{i}")
            lhs.append((upz[j] * upz[i], label))
            rhs.append((dnz[j] * dnz[i], label))
    lhs, rhs = simplify(lhs, rhs)
    return lhs, rhs


def second_order():
    lhs = []
    rhs = []
    for j in range(6):
        for i in range(6):
            for k in range(6):
                label = canonical_label(f"I{j}|{i},{k}")
                lhs.append((upz[j] * upz[i] * upz[k], label))
                rhs.append((dnz[j] * dnz[i] * dnz[k], label))
    lhs, rhs = simplify(lhs, rhs)
    return lhs, rhs


def nth_order_in_density(n):
    lhs = []
    rhs = []
    for j in range(6):
        for nderivs in it.product(range(6), repeat=n):
            sitelabels = map(str, nderivs)
            label = canonical_label(f"I{j}|{','.join(sitelabels)}")
            lhs_prefactor = upz[j]
            rhs_prefactor = dnz[j]
            for idx in nderivs:
                lhs_prefactor *= upz[idx]
                rhs_prefactor *= dnz[idx]
            lhs.append((lhs_prefactor, label))
            rhs.append((rhs_prefactor, label))
    lhs, rhs = simplify(lhs, rhs)
    return lhs, rhs


def get_endpoint_differences(energy_order):
    n = energy_order - 1
    if n == 0:
        return {}
    if np.sum(upz) != 0:
        raise ValueError("Unknown case 1")
    lhs, rhs = nth_order_in_density(n)
    if len(rhs) != 0:
        raise ValueError("Unknown case 2")
    return lhs


get_endpoint_differences(4)


# %%
nth_order(2)
# %%
