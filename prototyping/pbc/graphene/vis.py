#!/usr/bin/env python

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# region
def read_combinations(fn):
    with open(fn) as fh:
        lines = fh.readlines()

    rows = []
    for line in lines:
        if line.startswith("#"):
            continue
        if "[" not in line:
            system, rep = line.strip().split()
        if line.startswith("N "):
            ns = [int(_) for _ in line.strip().replace("[", "").replace("]", "").split()[1:]]
            continue
        if line.startswith("M "):
            ms = [float(_) for _ in line.strip().replace("[", "").replace("]", "").split()[1:]]
            continue
        if line.startswith("S "):
            ss = [float(_) for _ in line.strip().replace("[", "").replace("]", "").split()[1:]]
            for n, m, s in zip(ns, ms, ss):
                rows.append(
                    {"system": system, "rep": rep, "Npts": n, "mae": m, "stddev": s}
                )
            continue
    return rows


df = pd.DataFrame(
    read_combinations("production.results")
)
# region
for system, group in df.groupby("system"):
    for rep, group in group.groupby("rep"):
        plt.errorbar(
            group.Npts,
            group.mae / 128 * 1000,
            yerr=group.stddev / 128 * 1000,
            label=rep,
        )
    plt.title(system)
    plt.legend()
    plt.xlim(64, 2048)
    plt.xticks()
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("meV/atom")
    # plt.axhline(0.6 + 0.2)
    plt.show()
# region
