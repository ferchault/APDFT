#!/usr/bin/env python
#%%
import numpy as np
import shlex
import scipy.spatial.distance as ssd
import subprocess
import json

XTBPATH = "/home/guido/opt/xtb/xtb-6.4.0/"


def get_graphene_poscar(scaling, deltaZ, repeat=8, a=2.46):
    deltaZ = deltaZ.astype(np.int)
    if len(deltaZ) != 2 * repeat ** 2:
        raise ValueError()
    if sum(deltaZ) != 0:
        raise ValueError()
    if len(set(deltaZ) - set((-1, 0, 1))) > 0:
        raise NotImplementedError()

    repeat = 8
    elementstring = ""
    countstring = ""
    for element, count in zip("BCN", np.bincount(deltaZ + 1, minlength=3)):
        if count > 0:
            elementstring += f" {element}"
            countstring += f" {count}"
    header = f"""graphene
{scaling}
{a*repeat} 0.0 0.0
{-a/2*repeat} {a*np.sqrt(3)/2*repeat} 0.0
0.0 0.0 20
{elementstring} 
{countstring}
Direct
"""
    xpts = np.arange(repeat) / repeat
    A = np.vstack(
        (np.tile(xpts, repeat), np.repeat(xpts, repeat), np.zeros(64) + 0.5)
    ).T
    B = np.vstack((A[:, 0] + 1 / 3 / repeat, A[:, 1] + 2 / 3 / repeat, A[:, 2])).T
    merged = np.vstack((A, B))

    # re-sort to match dZ vector
    merged = np.vstack([merged[deltaZ == _] for _ in (-1, 0, 1)])

    lines = []
    for positions in merged:
        lines.append(f"{positions[0]} {positions[1]} {positions[2]}")

    return header + "\n".join(lines)


def random_representative(scaling, nBN):
    if nBN > 64:
        raise ValueError()
    dZ = np.zeros(128, dtype=np.int)
    dZ[:nBN] = 1
    dZ[nBN : 2 * nBN] = -1
    np.random.shuffle(dZ)
    fwd = get_graphene_poscar(1, dZ)
    bwd = get_graphene_poscar(1, -dZ)
    return fwd, bwd, dZ


# region


def run_and_extract(poscar):
    rundir = "/dev/shm/xtb-atomic-rundir/"
    try:
        os.mkdir(rundir)
    except:
        pass

    with open(f"{rundir}/POSCAR", "w") as fh:
        fh.write(poscar)

    cmd = (
        f"{XTBPATH}/bin/xtb --gfn 0 --sp --acc 1e-4 --strict --norestart --json POSCAR"
    )
    env = os.environ.copy()
    env["XTBPATH"] = f"{XTBPATH}/share/xtb"
    output = (
        subprocess.check_output(
            shlex.split(cmd), cwd=rundir, stderr=subprocess.DEVNULL, env=env
        )
        .decode("utf8")
        .split("\n")
    )
    with open(f"{rundir}/xtbout.json") as fh:
        return json.load(fh)["electronic energy"]


# region
def compare_energy_differences(nBN):
    all_energies = []
    enantiomer_deltas = []
    for i in range(50):
        f, b, d = random_representative(1, nBN)
        e_f = run_and_extract(f)
        e_b = run_and_extract(b)
        all_energies += [e_f, e_b]
        enantiomer_deltas.append(e_f - e_b)

    plt.title(f"C$_{{{128-2*nBN}}}$(BN)$_{{{nBN}}}$ in the graphene lattice")
    data = np.abs(enantiomer_deltas)
    plt.plot(sorted(data), np.arange(len(data)) / len(data) * 100, label="Enantiomers")
    data = ssd.pdist(np.array(all_energies).reshape(-1, 1))
    plt.plot(sorted(data), np.arange(len(data)) / len(data) * 100, label="Random pairs")
    plt.xlabel("$\Delta E$ [Ha]")
    plt.ylabel("Share [%]")
    plt.legend()
    plt.show()


compare_energy_differences(64)
compare_energy_differences(32)
compare_energy_differences(16)
compare_energy_differences(8)
compare_energy_differences(4)
compare_energy_differences(2)
compare_energy_differences(1)

#%%
# for nBN in (64, 32, 16):
#     for i in range(1, 51):
#         poscarup, poscardn, _ = random_representative(1, nBN)
#         with open(f"/data/guido/graphene-BN/{nBN}/up-{i}/POSCAR", "w") as fh:
#             fh.write(poscarup)
#         with open(f"/data/guido/graphene-BN/{nBN}/dn-{i}/POSCAR", "w") as fh:
#             fh.write(poscardn)
for nBN in (64,):
    for i in range(1, 2500):
        poscarup, poscardn, _ = random_representative(1, nBN)
        with open(
            f"/home/guido/wrk/APDFT/prototyping/pbc/graphene/production-{nBN}/up/{i}/POSCAR",
            "w",
        ) as fh:
            fh.write(poscarup)
        with open(
            f"/home/guido/wrk/APDFT/prototyping/pbc/graphene/production-{nBN}/dn/{i}/POSCAR",
            "w",
        ) as fh:
            fh.write(poscardn)

# region

# region
import glob
import pandas as pd

rows = []
for folder in glob.glob("/data/guido/graphene-BN/*/*"):
    with open(f"{folder}/OUTCAR") as fh:
        lines = fh.readlines()

    TEWEN = float([_ for _ in lines if "TEWEN" in _][-1].strip().split()[-1])
    TOTEN = float([_ for _ in lines if "TOTEN" in _][-1].strip().split()[-2])

    electronic_energy_eV = TOTEN - TEWEN
    nBN = int(folder.split("/")[-2])
    calc = int(folder.split("-")[-1])
    direction = folder.split("/")[-1].split("-")[0]

    rows.append(
        {
            "nBN": nBN,
            "direction": direction,
            "calc": calc,
            "electronic": electronic_energy_eV,
        }
    )

rows = pd.DataFrame(rows)
# region
rows
# region
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 18,
        "font.sans-serif": ["Fira Sans"],
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 1,
        "xtick.major.size": 6,
        "xtick.minor.width": 1,
        "xtick.minor.size": 3,
        "ytick.major.width": 1,
        "ytick.major.size": 6,
        "ytick.minor.width": 1,
        "ytick.minor.size": 3,
        "axes.edgecolor": "#333",
    }
)
f = plt.figure(dpi=70)
colors = sorted(rows.nBN.unique())
labels = {16: "C$_{96}$(BN)$_{16}$", 32: "C$_{64}$(BN)$_{32}$", 64: "(BN)$_{64}$"}
handles = []
for name, group in rows.groupby("nBN"):
    coloridx = colors.index(name)
    merged = pd.merge(
        group.query("direction == 'dn'"), group.query("direction == 'up'"), on="calc"
    )
    data = np.abs(merged.electronic_x.values - merged.electronic_y.values)
    (handle,) = plt.loglog(
        sorted(data),
        np.arange(1, len(data) + 1) / len(data) * 100,
        label=name,
        color=f"C{coloridx}",
    )
    print(np.average(data))
    handles.append(handle)
legend = plt.legend(
    handles=handles,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(-0.025, 1.1),
    ncol=1,
    title="Alchemical\nEnantiomers",
    columnspacing=0.5,
    handlelength=0.8,
    handletextpad=0.2,
)
legend._legend_box.align = "left"
legend.get_title().set_fontweight("bold")
legend.get_title().set_multialignment("left")
legend.get_title().set_color("#666")
ax = plt.gca().add_artist(legend)

handles = []
for name, group in rows.groupby("nBN"):
    coloridx = colors.index(name)
    data = ssd.pdist(np.array(group.electronic.values).reshape(-1, 1))
    (handle,) = plt.loglog(
        sorted(data),
        np.arange(1, len(data) + 1) / len(data) * 100,
        label=name,
        lw=3,
        color=f"C{coloridx}",
    )
    print(np.average(data))
    handles.append(handle)
plt.xlabel("$|\Delta E|$ [eV]")
plt.ylabel("Share of data points [%]")
legend = plt.legend(
    handles=handles,
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.75, 0.55),
    ncol=1,
    title="Random",
    columnspacing=0.5,
    handlelength=0.8,
    handletextpad=0.2,
)
legend._legend_box.align = "right"
legend.get_title().set_fontweight("bold")
legend.get_title().set_multialignment("right")
legend.get_title().set_color("#666")
plt.xlim(5e-3, 500)
plt.ylim(1, 100)
plt.show()
# region

# region


# region
rows.to_csv("/data/guido/graphene-BN/results.csv")
# region
