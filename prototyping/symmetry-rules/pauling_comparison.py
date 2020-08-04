#!/usr/bin/env python

# %%
import numpy as np
import scipy.optimize as sco

# %%
# upper triangle data Li...F, extracted from 10.1021/ja00150a030, part of Table 7
raw = (
    23.9,
    42.3,
    44.4,
    46.0,
    72.4,
    103.0,
    136.6,
    71.7,
    82.5,
    91.9,
    121.6,
    147.7,
    176.9,
    104.0,
    103.2,
    141.0,
    151.2,
    169.7,
    87.5,
    82.5,
    90.5,
    109.5,
    63.9,
    62.7,
    68.9,
    49.2,
    48.3,
    37.6,
)
data = np.zeros((7, 7))
data[np.triu_indices_from(data)] = raw
data

# %%
def pauling_fit(data):
    def table(*xs):
        homo = np.diag(data) / 2
        table = np.tile(homo, 7).reshape(-1, 7) + np.tile(homo, 7).reshape(-1, 7).T
        table += (
            23 * (np.tile(xs, 7).reshape(-1, 7) - np.tile(xs, 7).reshape(-1, 7).T) ** 2
        )
        return table

    def target(*xs):
        tab = table(xs)
        delta = (tab - data)[np.triu_indices_from(data, k=1)]
        return np.sqrt((delta ** 2).mean())

    result = sco.minimize(
        lambda xs: target(*xs), x0=np.arange(7), method="Nelder-Mead"
    ).x
    tab = table(result)
    delta = (tab - data)[np.triu_indices_from(data, k=1)]
    print(np.round(delta))
    print("RMSE kcal/mol", target(result))
    return result


pauling_fit(data)

# %%
Zs = np.arange(3, 10)
nuclear = np.tile(Zs, 7).reshape(-1, 7) * np.tile(Zs, 7).reshape(-1, 7).T
pauling_fit(data - nuclear * 630 / 4)


# %%
def apdft(data):
    deltas = []
    for i in range(5):
        j = i + 2
        center = i + 1
        if j - center == center - i:
            print(i, center, j)
            actual = data[i, center]
            pred = (data[i, i] - data[j, j]) / 2 + data[center, j]
            print(actual, pred)
            deltas.append(actual - pred)
    return np.abs(deltas).mean()


apdft(data - nuclear * 630 / 4)

# %%
