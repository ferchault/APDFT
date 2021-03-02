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
0.0 0.0 50
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
        return json.load(fh)["total energy"]


# region
nBN = 0
all_energies = []
enantiomer_deltas = []
for i in range(2):
    f, b, d = random_representative(1, nBN)
    e_f = run_and_extract(f)
    e_b = run_and_extract(b)
    all_energies += [e_f, e_b]
    enantiomer_deltas.append(e_f - e_b)

# region
data = np.abs(enantiomer_deltas)
plt.plot(sorted(data), np.arange(len(data)) / len(data))
data = ssd.pdist(np.array(all_energies).reshape(-1, 1))
plt.plot(sorted(data), np.arange(len(data)) / len(data))
# region


# region
# region
import matplotlib.pyplot as plt

plt.scatter(A[:, 0], A[:, 1])
plt.scatter(A[:, 0] + 1 / 3 / 8, A[:, 1] + 2 / 3 / 8)

A[:, 0] + 1 / 3 / 8, A[:, 1] + 2 / 3 / 8
# region

# region
import os

os.getcwd()
# region
dZ = np.zeros(128)
dZ[0] = -1
dZ[1] = 1
print(get_graphene_poscar(1, dZ))
# region
print(random_representative(1, 32)[0])
# region
