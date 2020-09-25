#!/usr/bin/env python

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as ssd
import pandas as pd
import qml
import scipy.special as ss
import scipy.optimize as sco

#%%
def figure():
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

    f4 = alpha_3 * ss.erf(2 * (xs + q * np.pi))
    axs[2].plot(xs, f4, label="Steep cliff")
    # axs[2].legend()

    f5 = (
        alpha_5
        * xs
        * np.exp(-(beta_3 * xs) ** 2)
        / (np.exp(-0.5) / (beta_3 * np.sqrt(2)))
    )
    axs[3].plot(xs, f5, label="Hard to approximate\nwith polynomials")
    # axs[3].legend()

    axs[4].plot(xs, f1 + f2 + f3 + f4 + f5)
    plt.subplots_adjust(hspace=0)


# %%
# example function
def example():
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
# rows = []
# k = 10
# funcs = {"total": total, "electronic": electronic}
# for kind in "total electronic".split():
#     for sigma in 2.0 ** np.arange(-2, 10):
#         for Ntrain in (4, 8, 16, 32, 64, 128, 256, 512):
#             f = model
#             if kind == "electronic":
#                 f = modmodel
#             mae = np.array([f(sigma, Ntrain, funcs[kind]) for _ in range(k)]).mean()
#             rows.append({"sigma": sigma, "N": Ntrain, "kind": kind, "mae": mae})
# rows = pd.DataFrame(rows)


# %%
# for kind, group in rows.groupby("kind"):
#     s = group.groupby("N").min()["mae"]
#     ys = s.values
#     if kind == "electronic":
#         ys /= 1
#     plt.loglog(s.index, ys, "o-", label=kind)
#     plt.xlabel("Training set size")
#     plt.ylabel("MAE")
# plt.legend()
# # %%
# plt.semilogy(rows.query("kind== 'total'").groupby("sigma").min()["mae"])

# %%
def components(xs, alphas, betas):  # =(1.01, 1, 0.1)
    a = alphas[0] * (np.cos(xs) - np.cos(betas[0] * xs))
    b = alphas[1] * 2 * (xs - q * np.pi) ** 2 / (3 * np.pi * q) ** 2
    c = alphas[2] * ss.erf(betas[1] * (xs + q * np.pi))
    d = (
        alphas[3]
        * xs
        * np.exp(-(betas[2] * xs) ** 2)
        / (np.exp(-0.5) / (betas[2] * np.sqrt(2)))
    )
    return a, b, c, d, a + b + c + d


q = 3
domain = (-q * np.pi * 2, q * np.pi * 2)


def predict_one(K, Kpred, ys, idxs):
    res = []
    for idx in idxs:
        alphas = qml.math.cho_solve(K, ys[idx][: K.shape[0]])
        actual = ys[idx][K.shape[0] :]
        pred = np.dot(Kpred.transpose(), alphas)
        res.append(np.abs(pred - actual).mean())
    return res


def predict_combined(alphas, sigmas, Ntrain, Ntest, betas):
    pts = np.random.uniform(low=domain[0], high=domain[1], size=Ntrain + Ntest)
    ys = components(pts, alphas, betas)
    pred = 0
    for idx in (0, 1, 2, 3):
        K = np.exp(
            -ssd.cdist(pts[:Ntrain].reshape(-1, 1), pts[:Ntrain].reshape(-1, 1))
            / (2 * sigmas[idx] ** 2)
        )
        Kpred = np.exp(
            -ssd.cdist(pts[:Ntrain].reshape(-1, 1), pts[Ntrain:].reshape(-1, 1))
            / (2 * sigmas[idx] ** 2)
        )
        alphas = qml.math.cho_solve(K, ys[idx][: K.shape[0]])
        pred += np.dot(Kpred.transpose(), alphas)
    return np.abs(pred - ys[-1][K.shape[0] :]).mean()


def model(sigma, Ntrain, Ntest, alphas, k, betas):
    A, B, C, D, joint = 0, 0, 0, 0, 0
    for n in range(k):
        pts = np.random.uniform(low=domain[0], high=domain[1], size=Ntrain + Ntest)
        ys = components(pts, alphas, betas)

        # build large kernel
        K = np.exp(
            -ssd.cdist(pts[:Ntrain].reshape(-1, 1), pts[:Ntrain].reshape(-1, 1))
            / (2 * sigma ** 2)
        )
        Kpred = np.exp(
            -ssd.cdist(pts[:Ntrain].reshape(-1, 1), pts[Ntrain:].reshape(-1, 1))
            / (2 * sigma ** 2)
        )

        m = predict_one(K, Kpred, ys, [0, 1, 2, 3])
        A += m[0]
        B += m[1]
        C += m[2]
        D += m[3]
        joint += predict_one(K, Kpred, ys, [4])[0]

    return {
        "sigma": sigma,
        "Ntrain": Ntrain,
        "alphas": alphas,
        "A": A / k,
        "B": B / k,
        "C": C / k,
        "D": D / k,
        "joint": joint / k,
    }


#%%
def build_rows():
    rows = []
    k = 5

    for sigma in 2.0 ** np.arange(-2, 10):
        print(sigma)
        for case in (
            (1, 1, 1, 1),
            (10, 1, 1, 1),
            (1, 10, 1, 1),
            (1, 1, 10, 1),
            (1, 1, 1, 10),
        ):
            for Ntrain in (4, 8, 16, 32, 64, 128, 256, 512):
                rows.append(model(sigma, Ntrain, 100, case, k))
    return pd.DataFrame(rows)


# rows = build_rows()


# %%
# f, axs = plt.subplots(len(rows.groupby("alphas")), 1, figsize=(5, 10))
# idx = 0
# for case, group in rows.groupby("alphas"):
#     trains = []
#     joints = []
#     separates = []
#     for Ntrain in sorted(rows.Ntrain.unique()):
#         sigmas = [
#             group.query("Ntrain == @Ntrain").sort_values(_).sigma.values[0]
#             for _ in "ABCD"
#         ]
#         best_joint = (
#             rows.query("alphas==@case & Ntrain == @Ntrain")
#             .sort_values("joint")
#             .joint.values[0]
#         )
#         separate = np.array(
#             [predict_combined(case, sigmas, Ntrain, 100) for _ in range(2)]
#         ).mean()
#         trains.append(Ntrain)
#         joints.append(best_joint)
#         separates.append(separate)
#     axs[idx].plot(
#         trains, np.array(joints) / np.array(separates), label="improvement separate"
#     )
#     axs[idx].legend()
#     idx += 1

# %%
def does_separation_pay(case, betas, k=5):
    # scan hyperparameters
    rows = []
    for sigma in 2.0 ** np.arange(-2, 10):
        for Ntrain in (4, 8, 16, 32, 64, 128, 256, 512):
            rows.append(model(sigma, Ntrain, 100, case, k, betas))
    rows = pd.DataFrame(rows)

    j, s = [], []
    for Ntrain in sorted(rows.Ntrain.unique()):
        sigmas = [
            rows[rows.Ntrain == Ntrain].sort_values(_).sigma.values[0]
            for _ in "ABCD"
        ]
        best_joint = (
            rows.query("alphas==@case & Ntrain == @Ntrain")
            .sort_values("joint")
            .joint.values[0]
        )
        separate = np.array(
            [predict_combined(case, sigmas, Ntrain, 100, betas) for _ in range(k)]
        ).mean()
        j.append(best_joint)
        s.append(separate)

    return np.max(np.array(j) / np.array(s))


# does_separation_pay((100, 1, 100, 1), (1.01, 3, 0.1), k=3)
# %%


def fom(x):
    f = does_separation_pay(tuple(x), (1.01, 3, 0.1), k=50)
    print(-f, x)
    return -f


result = sco.differential_evolution(
    fom, bounds=[(1, 100)] * 4 , popsize=32, updating="deferred", workers=32, disp=True
)
print ("FINAL", result.x, result.fun)

# %%

