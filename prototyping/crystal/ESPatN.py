#!/usr/local/env python
#%%
import glob
import io
import functools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

basepath = "/mnt/c/Users/guido/data/vasp/n2_co/alchemical-vasp"

#%%
def get_epn(lvalstr):
    """Extracts the locally averaged electronic electrostatic potential from the .129 file.

    Hard coded for this particular data set.

    Parameters
    ----------
    lvalstr : str
        Mixing parameter as string as in the output file names.

    Returns
    -------
    tuple
        EPN in eV
    """
    with open(f"{basepath}/{lvalstr}.129") as fh:
        lines = fh.readlines()

    epns = []
    section_lines = []
    in_section = False
    for line in lines:
        if in_section:
            if "POT" in line or "DIST" in line:
                in_section = False
                A = np.loadtxt(io.StringIO("".join(section_lines)))
                # weighted sum around nucleus
                A = A[A[:, 0] < 0.1]
                epns.append((A[:, 1] * A[:, 2]).sum() / A[:, 2].sum())
                section_lines = []
            else:
                section_lines.append(line)
        if "DIST(IND)" in line:
            in_section = True
    return epns[0], epns[-1]


def get_epn_averaged(lines):
    """Extracts the averaged electronic electrostatic potential from the log file.

    Hard-coded for that particular data set.

    Parameters
    ----------
    lines : list of str
        Log file lines

    Returns
    -------
    tuple
        EPN in eV
    """
    for idx, line in enumerate(lines):
        if "average (electrostatic) potential at core" in line:
            epnline = lines[idx + 3]
            break

    parts = epnline.strip().split()
    return float(parts[1]), float(parts[-1])


@functools.lru_cache(maxsize=1)
def get_energies():
    def _get_last_number(lines, keyword):
        relevant = []
        for line in lines:
            if "Iteration      2" in line:
                break
            if keyword in line:
                relevant.append(line)
        relevant = relevant[-1]
        parts = relevant.strip().split()[::-1]
        for part in parts:
            try:
                part = float(part)
            except:
                continue
            return part

    rows = []
    for outcar in glob.glob(f"{basepath}/OUTCAR_*"):
        with open(outcar) as fh:
            lines = fh.readlines()

        energies = {
            "free": "TOTEN",
            "NN": "TEWEN",
            "entropy": "EENTRO",
            "reference": "EATOM",
        }
        details = {k: _get_last_number(lines, v) for k, v in energies.items()}
        lvalstr = outcar.split("_")[-1]
        details["lval"] = float(lvalstr)
        epns = get_epn(lvalstr)
        epns_averaged = get_epn_averaged(lines)
        details["pseudoepn"] = epns[1] - epns[0]
        details["averagedepn"] = epns_averaged[1] - epns_averaged[0]
        rows.append(details)
        if lvalstr != "0.0":
            details = details.copy()
            details["lval"] *= -1
            details["pseudoepn"] *= -1
            details["averagedepn"] *= -1
        rows.append(details)

    df = pd.DataFrame(rows)
    df["total"] = df.free - df.entropy - df.reference
    df["electronic"] = df.total - df.NN

    return df["lval electronic NN total pseudoepn averagedepn".split()].sort_values(
        "lval"
    )


# %%
plt.plot(get_energies().lval, get_energies().electronic, "o-", label="actual")

# tangent pseudoepn
delta = 0.2
first = True
for idx, row in get_energies().iterrows():
    xminus = row.lval - delta
    xplus = row.lval + delta
    for colidx, colname in enumerate("pseudoepn averagedepn".split()):
        yminus = row.electronic - delta * row[colname]
        yplus = row.electronic + delta * row[colname]
        label = None
        if first:
            label = colname
        plt.plot((xminus, xplus), (yminus, yplus), color=f"C{colidx+1}", label=label)
    first = False
plt.legend()
plt.xlabel("Mixing parameter $\lambda$")
plt.ylabel("Electronic energy [eV]")

# %%
polymodel = np.polyfit(get_energies().lval, get_energies().electronic, 3)
modelderivative = np.poly1d(np.polyder(polymodel))

# plt.plot(get_energies().lval, get_energies().lval * 0, label="ideal")
for colidx, colname in enumerate("pseudoepn averagedepn".split()):
    plt.plot(
        get_energies().lval,
        modelderivative(get_energies().lval) - get_energies()[colname],
        color=f"C{colidx+1}",
        label=colname,
    )

# linear fitting
ys = modelderivative(get_energies().lval) - get_energies()["pseudoepn"].values
shifted = ys - ys[-1] * get_energies().lval
plt.plot(get_energies().lval, shifted, color=f"C3", label="Linear shift")

plt.xlabel("Mixing parameter $\lambda$")
plt.ylabel("Deviation from first derivative [eV/$\lambda$]")
plt.legend()

# %%
# A: treat \int dr 1/|r-R| \rho^1 as element, not system dependent
# Q Chasz: multiple iterations in 1.0 OUTCAR? Looks like geo opt instead of SPC
# Q Florian: How to evaluate \int dr 1/|r-R| \rho^1 from CHGCAR and POTCAR

# %%
import pymatgen
import pymatgen.analysis.ewald


def ewald(lval):
    a = 20
    s = pymatgen.core.structure.Structure(
        [[a, 0, 0], [0, a, 0], [0, 0, a]],
        species="C O".split(),
        coords=[[10, 0, 0], [11.12, 0, 0]],
        charge=0,
        coords_are_cartesian=True,
        site_properties={"charge": [7 - lval, 7 + lval]},
    )
    e = pymatgen.analysis.ewald.EwaldSummation(s)
    return e.total_energy  # eV


plt.ylabel("$\Delta E_{NN}$")
plt.xlabel("Mixing parameter $\lambda$")
plt.plot(
    get_energies().lval,
    [ewald(_) - ewald(0) for _ in get_energies().lval],
    label="pymatgen",
)
plt.plot(get_energies().lval, get_energies().NN - 219.079885, label="VASP")
a = [ewald(_) - ewald(0) for _ in get_energies().lval][-1]
b = (get_energies().NN.values - 219.079885)[-1]
print(a - b)
plt.legend()
# %%

# %%
