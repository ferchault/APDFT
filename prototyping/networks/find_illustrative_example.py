#!/usr/bin/env python
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import scipy.stats as sts


def read_reference_energies():
    folders = glob.glob("naphtalene/validation-molpro/*/")
    res = []
    for folder in folders:
        this = {}

        basename = folder.split("/")[-2]
        this["label"] = basename.split("-")[-1]
        this["nbn"] = int(basename.split("-")[1])

        try:
            with open(folder + "direct.out") as fh:
                lines = fh.readlines()
            this["energy"] = float(lines[-6].strip().split()[-1])
            this["nn"] = float(
                [_ for _ in lines if "Nuclear energy" in _][0].strip().split()[-1]
            )
        except:
            with open(folder + "run.log") as fh:
                lines = fh.readlines()
            this["energy"] = float(lines[-7].strip().split()[-1])
            this["nn"] = float(
                [_ for _ in lines if "Nuclear repulsion energy" in _][0]
                .strip()
                .split()[-1]
            )

        res.append(this)
    return pd.DataFrame(res)


df = read_reference_energies()

# %%
def read_report(fn, restrict):
    with open(fn) as fh:
        lines = fh.readlines()

    order = []
    groups = []
    count = 0
    started = False
    if restrict is None:
        started = True
    for line in lines:
        # check for relevant section
        if restrict is not None:
            if "stoichiometry" in line:
                nbn = len([_ for _ in line if _ == "5"])
                if nbn == restrict:
                    started = True
                else:
                    started = restrict is None
        if not started:
            continue
        if "Found:" in line:
            label = "".join(
                line.split("[")[1].split("]")[0].replace(" ", "").split(",")
            )
            label = label.replace("5", "B").replace("6", "C").replace("7", "N")
            order.append(label)
            count += 1

        if "Group energy" in line:
            groups.append(count)
            count = 0
    return order, groups[1:]


report = read_report("scanning/2.2-CM", None)

# %%
def electronic(bnlabel):
    label = bnlabel.replace("B", "5").replace("C", "6").replace("N", "7")
    s = df.query("label==@label")
    return s.energy.values[0] - s.nn.values[0]


def find_groups(report):
    start = 0
    minspan = 1000
    for group in report[1]:
        if group > 2:
            es = [electronic(_) for _ in report[0][start : start + group]]
            span = (max(es) - min(es)) * 1000
            if span < minspan:
                print("Look here", report[0][start])
                minspan = span
            print(group, span)
        start += group


find_groups(report)

# %%
np.array(
    [
        float(_)
        for _ in "BBNCNCNBCC".replace("B", "5").replace("C", "6").replace("N", "7")
    ]
)

# %%
electronic("BBNCNCNBCC"), electronic("BBNBCNCNCC"), electronic("BBNBNCNCCC")

# %%
