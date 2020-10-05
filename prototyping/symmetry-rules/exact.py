#!/usr/bin/env python

#%%
# atoms 012 345 in benzene
import numpy as np
import itertools as it

upz = np.array((1, -1, 0, 0, -1, 1))
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
def term_n(n, is_B=True):
    energy_derivative_order = 2 * n
    if is_B:
        energy_derivative_order += 1
    latex = f"B_{n}&="
    term = []
    triviallabels = []
    simplified = []
    for j in range(6):
        for nderivs in it.product(range(6), repeat=energy_derivative_order - 1):
            sitelabels = list(map(str, nderivs))
            triviallabel = f"I{j}|{','.join(sitelabels)}"
            label = canonical_label(triviallabel)
            lhs_prefactor = upz[j]
            for idx in nderivs:
                lhs_prefactor *= upz[idx]
            if lhs_prefactor != 0:
                term.append((lhs_prefactor, label))
                latex += (
                    "\\int \\dr \\frac{"
                    + str(lhs_prefactor)
                    + "}{|\\fatr-\\fatR_"
                    + str(j)
                    + "|}\\frac{\\partial^2\\rho}{\\partial Z_"
                    + str(sitelabels[0])
                    + "\\partial Z_"
                    + str(sitelabels[1])
                    + "}+"
                )
                triviallabels.append((lhs_prefactor, triviallabel))

    arr = array_to_dict(term)
    for k in list(arr.keys()):
        if arr[k] == 0:
            del arr[k]
    print(arr)


term_n(1, is_B=True)


# %%
import matplotlib.pyplot as plt
import findiff

# %%
# assume everything is an gaussian, and that mixed derivatives are less important

basesigma = 0.4


def gaussian(xs, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (xs / sigma) ** 2)


def nfd(xs, order):
    mode = findiff.coefficients(deriv=order, acc=order)["center"]
    result = xs * 0
    delta = basesigma / 10
    for offset, coefficient in zip(mode["offsets"], mode["coefficients"]):
        result += coefficient * gaussian(xs, basesigma + delta * offset)
    return result / (delta ** order)


xs = np.linspace(0, 5, 500)

ds = (2.6, 4.5, 5.3)
es = []
for distance in ds:
    es.append((nfd(xs + distance, 1) * 4 * np.pi * xs).sum())
plt.plot(ds, es)

# %%

# %%
# assuming Kato's cusp
import scipy.special


def nth_derivative(xs, order):
    result = 0
    alpha = 0.273
    beta = 3.56
    Z = 6
    for k in range(order + 1):
        result += (
            scipy.special.binom(order, k)
            * alpha
            * beta ** (order - k)
            * Z ** (beta - order + k)
            * (-2 * xs) ** k
            * np.exp(-2 * Z * xs)
        )
    return result


r = np.linspace(0, 1)
d = 4
plt.plot(r, nth_derivative(r, d))
plt.plot(-r, nth_derivative(r, d))

# %%
2.8 * 1.889725988
# %%

