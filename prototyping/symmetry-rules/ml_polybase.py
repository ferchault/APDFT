#!/usr/bin/env python
#%%
import autograd
import autograd.numpy as anp
import numpy
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd

# %%
def to_learn(xs):
    return xs ** 2 + xs + 4


def baseline(xs, a, b):
    return a * xs ** 2 + b * xs


def cdist(A, B, norm):
    ret = np.zeros((len(A), len(B)))
    for i in range(len(B)):
        for j in range(i, len(A)):
            ret[i, j] = norm(A[i] - B[j])
            ret[j, i] = ret[i, j]
    return ret


def np_alphas(xs, ys, sigma):
    xs = xs.reshape(-1, 1)
    K = np.exp(-ssd.cdist(xs, xs) / (2 * sigma ** 2))
    Kinv = np.linalg.inv(K)
    return np.dot(Kinv, ys)


def ad_alphas(xs, ys, sigma):
    K = anp.exp(-cdist(training, training, anp.linalg.norm) / (2 * sigma ** 2))
    Kinv = anp.linalg.inv(K)
    return anp.dot(Kinv, ys_train)


def np_predict(xs, xs2, sigma, alphas):
    K = np.exp(-ssd.cdist(xs, xs2, np.linalg.norm) / (2 * sigma ** 2))
    return np.dot(K.transpose(), alphas)


baseline_params = (0.0, 0.0)
xs = np.linspace(-5, 5, 100)
plt.plot(xs, to_learn(xs), label="target")
plt.plot(xs, baseline(xs, *baseline_params), label="baseline")
sigma = 1
Ntrain = 4
Ntest = 100

training = np.random.uniform(-5, 5, size=Ntrain)
ys_train = to_learn(training) - baseline(training, *baseline_params)
refalphas = np_alphas(training, ys_train, sigma)
alphas = ad_alphas(training, ys_train, sigma)
assert np.allclose(np.array(alphas), refalphas), "AD mismatch alphas"
test = xs
ys_test = to_learn(test)
pred = np_predict()
pred += baseline(xs, *baseline_params)

plt.plot(xs, pred, label="ML")
plt.legend()

# %%
refalphas, alphas
# %%
