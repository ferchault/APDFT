#!/usr/bin/env python
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASEDIR = "/home/ferchault/data/emily/cases_def2"
offsets = [[0], [-1, 1], [-1,0, 1], [-2, -1, 1, 2]]
weights = [[1], [-0.5, 0.5], [1, -2, 1], [-0.5, 1.0, -1.0, 0.5]]
delta = 0.001

def get_energies(folder):
    extract = {}
    with open(f"{BASEDIR}/{folder}/run.log") as fh:
        lines = fh.readlines()
        for key, identifiers in {
            "HF": ("Total Energy       :",),
            "MP2": ("E(MP2)",),
            "CCSD": ("E(CCSD)        ", "E(TOT)"),
            "CCSD(T)": ("E(CCSD(T))       ",)
        }.items():
            col = -1
            if key == "HF":
                col = 3
            for identifier in identifiers:
                try:
                    extract[key] = float(
                        [_ for _ in lines if identifier in _][0].strip().split()[col]
                    )
                except IndexError:
                    continue
    extract["MP2"] += extract["HF"]
    return extract


def get_df():
    cases = "C_m0.001 C_m0.002 C_m0.500 C_m1.000 C_m1.500 C_m2.000 C_p0.000 C_p0.001 C_p0.002 C_p0.500 C_p1.000 C_p1.500 C_p2.000"
    rows = []
    for case in cases.split():
        row = get_energies(case)
        lval = float(case[3:]) * (2 * ("p" in case) - 1)
        row["lval"] = lval
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lval")


df = get_df()

# %%
for lot in "HF MP2 CCSD CCSD(T)".split():
    plt.plot(df.lval, df[lot], label=lot)
    plt.plot(df.lval, pred)
plt.legend()
# %%
def taylor(column):
    coefficients = []
    for order in range(len(offsets)):
        coefficient = 0
        for offset, weight in zip(offsets[order], weights[order]):
            lval = offset * delta
            coefficient += df.query("lval == @lval")[column].values[0] * weight
        coefficient /= delta ** order
        coefficient /= np.math.factorial(order)
        coefficients.append(coefficient)
    return np.array(coefficients)

def evaluate(column):
    actual = df[column]
    for order in range(1, 5):
        pred = np.polyval(taylor(column)[:order][::-1], df.lval)
        plt.plot(df.lval, pred-actual, label=f"APDFT{order-1}")
    plt.legend()
    plt.ylabel("Error [Ha]")
    plt.xlabel("Lambda value")
    plt.ylim(-1,1)
    plt.title(column)
    plt.show()

for lot in "HF MP2 CCSD CCSD(T)".split():
    evaluate(lot)
# %%
