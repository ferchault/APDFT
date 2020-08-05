#!/usr/bin/env python

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
# taken from https://pubs.acs.org/doi/pdf/10.1021/jp050857o, Table 1
raw = """109.2 180.4 188 193.5 302.4 430.3 572.9 96.59 152.7 181 214 219.8 335.3 
469.7 297.7 342.3 380.4 503.1 613.3 739 152.1 233 278.7 319 326.7 440.1 
558.9 431.6 426.8 583.3 626 705.4 148 249.3 317.5 352.6 362.9 466.8 518.2 
359.2 337.7 370.2 452 139.8 261.3 335.9 358.8 284.1 296.7 338.1 259.6 
253.5 283.6 216.9 356.3 456.2 421.3 306.3 265.7 246.6 195.2 194.9 333.3 
469.9 541.2 506.4 359.7 279.1 224.6 151.7 478.6 604.1 665.5 624.7 454.7 
336.2 249.2 87.09 134.4 155.5 181.7 183.2 285.1 413.4 190.4 219.9 250.5 
253.9 358 479.6 257.9 289.4 294.5 402.5 502.5 313.2 287.3 357.4 448.4 
225.8 263.7 315.1 252 258.3 234.6"""
elements = np.fromstring(raw, dtype=np.float, sep=" ")
NELEMENTS = 14
data = np.zeros((NELEMENTS, NELEMENTS))
data[np.triu_indices_from(data)] = elements
data = data + data.T - np.diag(np.diag(data))
Zs = np.arange(3, 18)[np.arange(3, 18) != 10]
LABELS = "_ H He Li Be B C N O F Ne Na Mg Al Si P S Cl".split()

# %%
def rule_1(data, nuclear):
    results = []

    for A in range(NELEMENTS - 2):
        R = A + 1
        B = A + 2
        if (Zs[A] < 10 and Zs[B] > 10) or (Zs[A] > 10 and Zs[B] < 10):
            continue

        period = 2
        if Zs[A] > 10:
            period = 3

        # RHS: AR ~ BR + (AA - BB)/2
        actual = data[A, R]
        pred = (
            data[B, R]
            + (data[A, A] - nuclear[A, A] - data[B, B] + nuclear[B, B]) / 2
            + nuclear[A, R]
            - nuclear[B, R]
        )
        results.append(
            {
                "unknown": A,
                "known": B,
                "R": R,
                "actual": actual,
                "pred": pred,
                "kind": "LHS",
                "period": period,
            }
        )

        # LHS: BR ~ AR + (BB - AA)/2
        actual = data[B, R]
        pred = (
            data[A, R]
            + (data[B, B] - nuclear[B, B] - data[A, A] + nuclear[A, A]) / 2
            + nuclear[B, R]
            - nuclear[A, R]
        )
        results.append(
            {
                "unknown": B,
                "known": A,
                "R": R,
                "actual": actual,
                "pred": pred,
                "kind": "RHS",
                "period": period,
            }
        )

    return pd.DataFrame(results)


df = rule_1(data, nuclear * 2625.4996395718035)


# %%
for name, group in df.groupby("kind period".split()):
    plt.scatter(-group.actual, -group.pred, label=name)
plt.legend()
plt.xlim(-500, -100)
plt.ylim(-500, -100)
for idx, row in df.iterrows():
    A = int(row.unknown)
    R = int(row.R)
    label = f"{LABELS[Zs[A]]}-{LABELS[Zs[R]]}"
    if -500 < -row.actual < -100 and -500 < -row.pred < -100:
        plt.text(-row.actual, -row.pred, label)
plt.plot((-500, -100), (-500, -100), alpha=0.5)


# %%
np.abs(df.actual - df.pred).mean()

# %%
len(df)

# %%
