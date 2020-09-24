#!/usr/bin/env python

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as ssd
import pandas as pd
import qml


# %%
# example function
SCALE = 10
Q = 5
def electronic(xs):
    return SCALE * np.sin(xs) + np.cos(xs)
def nuclear(xs):
    return -SCALE * np.sin(xs)
def total(xs):
    return electronic(xs) + nuclear(xs)
xs = np.linspace(0, Q*5*np.pi, 100)
ys = electronic(xs)        # unknown, hard
ys2 = nuclear(xs)                    # known, simple
plt.plot(xs, electronic(xs), label="electronic")
plt.plot(xs, nuclear(xs), label="nuclear")
plt.plot(xs, total(xs), label="total E")
plt.legend()

# %%
def modmodel(sigma, Ntrain, func, Ntest=100):
    pts = np.random.uniform(low=0, high=2*np.pi, size=Ntrain+Ntest)
    training, test = pts[:Ntrain], pts[Ntrain:]
    K = np.exp(-ssd.cdist(training.reshape(-1,1), training.reshape(-1,1))/(2*sigma**2))
    ys_train = func(training)
    ys_test = func(test)
    alphas = qml.math.cho_solve(K, ys_train)
    K = np.exp(-ssd.cdist(training.reshape(-1,1), test.reshape(-1,1))/(2*sigma**2))
    pred = np.dot(K.transpose(), alphas)
    return np.abs(pred - ys_test).mean()
def model(sigma, Ntrain, func, Ntest=100):
    pts = np.random.uniform(low=0, high=Q*2*np.pi, size=Ntrain+Ntest)
    training, test = pts[:Ntrain], pts[Ntrain:]
    K = np.exp(-ssd.cdist(training.reshape(-1,1), training.reshape(-1,1))/(2*sigma**2))
    ys_train = func(training)
    ys_test = func(test)
    alphas = qml.math.cho_solve(K, ys_train)
    K = np.exp(-ssd.cdist(training.reshape(-1,1), test.reshape(-1,1))/(2*sigma**2))
    pred = np.dot(K.transpose(), alphas)
    return np.abs(pred - ys_test).mean()

#model(2, 100, total)
rows = []
k = 10
funcs = {'total': total, 'electronic': electronic}
for kind in 'total electronic'.split():
    for sigma in 2.**np.arange(-2, 10):
        for Ntrain in (4, 8, 16, 32, 64, 128, 256,512):
            f = model
            if kind == "electronic":
                f = modmodel
            mae = np.array([f(sigma, Ntrain, funcs[kind]) for _ in range(k)]).mean()
            rows.append({'sigma': sigma, 'N': Ntrain, 'kind': kind, 'mae': mae})
rows = pd.DataFrame(rows)


# %%
for kind, group in rows.groupby("kind"):
    s = group.groupby("N").min()['mae']
    ys = s.values
    if kind == 'electronic':
        ys /= 1
    plt.loglog(s.index, ys, 'o-', label=kind)
    plt.xlabel("Training set size")
    plt.ylabel("MAE")
plt.legend()
# %%
plt.semilogy(rows.query("kind== 'total'").groupby("sigma").min()['mae'])
# %%
