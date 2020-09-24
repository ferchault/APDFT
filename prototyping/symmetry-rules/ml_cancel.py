#!/usr/bin/env python

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as ssd
import pandas as pd
import qml
import scipy.special as ss


#%%
f, axs = plt.subplots(5, 1, sharex=True, figsize=(3, 8))

q = 3
xs = np.linspace(-q * np.pi * 2, q * np.pi * 2, 1000)

alpha_1 = 1
alpha_2 = 1
alpha_3 = 1
alpha_4 = 1
alpha_5 = 1
beta_1 = 1.01
beta_3 = 0.1
f1 = alpha_1 * np.cos(xs)
axs[0].plot(xs, f1, label="Highly symmetric")
f2 = -alpha_1 * np.cos(beta_1 * xs)
axs[0].plot(xs, f2, label="Almost canceling")
axs[0].plot(xs, f1 + f2, label="Residual")
# axs[0].legend()

f3 = alpha_2 * 2 * (xs - q * np.pi) ** 2 / (3 * np.pi * q) ** 2
axs[1].plot(xs, f3, label="Simple trend")
# axs[1].legend()

f4 = alpha_3 * ss.erf(xs + q * np.pi)
axs[2].plot(xs, f4, label="Steep cliff")
# axs[2].legend()

f5 = alpha_5 * xs * np.exp(-(beta_3 * xs) ** 2) / (np.exp(-0.5) / (beta_3 * np.sqrt(2)))
axs[3].plot(xs, f5, label="Hard to approximate\nwith polynomials")
# axs[3].legend()

axs[4].plot(xs, f1 + f2 + f3 + f4 + f5)
plt.subplots_adjust(hspace=0)


# %%
# example function
SCALE = 10
Q = 2
BETA = 1.01


def electronic(xs):
    return SCALE * np.sin(Q * BETA * xs) + np.cos(xs)


def nuclear(xs):
    return -SCALE * np.sin(Q * xs)


def total(xs):
    return electronic(xs) + nuclear(xs)


xs = np.linspace(0, Q * 5 * np.pi, 100)
ys = electronic(xs)  # unknown, hard
ys2 = nuclear(xs)  # known, simple
plt.plot(xs, electronic(xs), label="electronic")
plt.plot(xs, nuclear(xs), label="nuclear")
plt.plot(xs, total(xs), label="total E")
plt.legend()

# %%
def modmodel(sigma, Ntrain, func, Ntest=100):
    pts = np.random.uniform(low=0, high=2 * np.pi, size=Ntrain + Ntest)
    training, test = pts[:Ntrain], pts[Ntrain:]
    K = np.exp(
        -ssd.cdist(training.reshape(-1, 1), training.reshape(-1, 1)) / (2 * sigma ** 2)
    )
    ys_train = func(training)
    ys_test = func(test)
    alphas = qml.math.cho_solve(K, ys_train)
    K = np.exp(
        -ssd.cdist(training.reshape(-1, 1), test.reshape(-1, 1)) / (2 * sigma ** 2)
    )
    pred = np.dot(K.transpose(), alphas)
    return np.abs(pred - ys_test).mean()


def model(sigma, Ntrain, func, Ntest=100):
    pts = np.random.uniform(low=0, high=Q * 2 * np.pi, size=Ntrain + Ntest)
    training, test = pts[:Ntrain], pts[Ntrain:]
    K = np.exp(
        -ssd.cdist(training.reshape(-1, 1), training.reshape(-1, 1)) / (2 * sigma ** 2)
    )
    ys_train = func(training)
    ys_test = func(test)
    alphas = qml.math.cho_solve(K, ys_train)
    K = np.exp(
        -ssd.cdist(training.reshape(-1, 1), test.reshape(-1, 1)) / (2 * sigma ** 2)
    )
    pred = np.dot(K.transpose(), alphas)
    return np.abs(pred - ys_test).mean()


# model(2, 100, total)
rows = []
k = 10
funcs = {"total": total, "electronic": electronic}
for kind in "total electronic".split():
    for sigma in 2.0 ** np.arange(-2, 10):
        for Ntrain in (4, 8, 16, 32, 64, 128, 256, 512):
            f = model
            if kind == "electronic":
                f = modmodel
            mae = np.array([f(sigma, Ntrain, funcs[kind]) for _ in range(k)]).mean()
            rows.append({"sigma": sigma, "N": Ntrain, "kind": kind, "mae": mae})
rows = pd.DataFrame(rows)


# %%
for kind, group in rows.groupby("kind"):
    s = group.groupby("N").min()["mae"]
    ys = s.values
    if kind == "electronic":
        ys /= 1
    plt.loglog(s.index, ys, "o-", label=kind)
    plt.xlabel("Training set size")
    plt.ylabel("MAE")
plt.legend()
# %%
plt.semilogy(rows.query("kind== 'total'").groupby("sigma").min()["mae"])
# %%
