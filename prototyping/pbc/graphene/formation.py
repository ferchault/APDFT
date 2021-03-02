#!/usr/bin/env python
#%%
from basis_set_exchange.misc import compact_elements
import numpy as np
import shlex
import scipy.spatial.distance as ssd
import subprocess
import json
import tqdm

BASEDIR = "/mnt/c/Users/guido/workcopies/apdft/prototyping/pbc/graphene"


def get_poscar(scaling, deltaZ):
    if len(deltaZ) != 128:
        raise ValueError()
    print(sum(deltaZ))
    if sum(deltaZ) != 0:
        raise ValueError()
    if len(set(deltaZ) - set((-1, 0, 1))) > 0:
        raise NotImplementedError()

    with open(f"{BASEDIR}/pure-large/POSCAR") as fh:
        lines = [_.strip() for _ in fh.readlines()]

    elements = " ".join(["CNB"[_] for _ in deltaZ])
    lines[1] = str(scaling)
    lines[5] = elements + " "  # space needed for reading POSCAR in vmd
    lines[6] = "1 " * 128

    return "\n".join(lines)


def random_representative(scaling, nBN):
    if nBN > 64:
        raise ValueError()
    dZ = np.zeros(128, dtype=np.int)
    dZ[:nBN] = 1
    dZ[nBN : 2 * nBN] = -1
    np.random.shuffle(dZ)
    fwd = get_poscar(1, dZ)
    bwd = get_poscar(1, -dZ)
    return fwd, bwd, dZ


# region


def run_and_extract(poscar):
    with open(f"{BASEDIR}/rundir/POSCAR", "w") as fh:
        fh.write(poscar)

    cmd = "/mnt/c/Users/guido/opt/xtb/6.4.0/bin/xtb --gfn 0 --sp --acc 1e-4 --strict --norestart --json POSCAR"
    output = (
        subprocess.check_output(
            shlex.split(cmd), cwd=f"{BASEDIR}/rundir/", stderr=subprocess.DEVNULL
        )
        .decode("utf8")
        .split("\n")
    )
    with open(f"{BASEDIR}/rundir/xtbout.json") as fh:
        return json.load(fh)["electronic energy"]


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
