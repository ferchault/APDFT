#!/usr/bin/env python
# counts realisations of unique dopant patterns in BN-doped graphene
# assumes a regular lattice and a n x n super cell

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as ssd
import scipy.special as scs
import subprocess

# region
# n = numbers of unitcells
def get_combinatorial(n):
    natoms = n * n * 2
    return scs.comb(natoms, natoms / 2)


def spatial_symmetry(n):
    repeat = n
    xpts = np.arange(repeat) / repeat

    # build lattice coordinates
    a = 2.46
    va = np.array((a * repeat, 0.0))
    vb = np.array((-a / 2 * repeat, a * np.sqrt(3) / 2 * repeat))

    A = np.vstack((np.tile(xpts, repeat), np.repeat(xpts, repeat))).T
    B = np.vstack((A[:, 0] + 1 / 3 / repeat, A[:, 1] + 2 / 3 / repeat)).T
    merged = np.vstack((A, B))
    merged = np.outer(merged[:, 0], va) + np.outer(merged[:, 1], vb)
    padded = []
    for dx in range(3):
        for dy in range(3):
            padded.append(merged + dx * va + dy * vb)
    padded = np.vstack(padded)
    merged = padded

    d1 = ssd.pdist(merged)
    threshold = 1.01 * min(d1)
    ds = ssd.squareform(d1)
    natoms = repeat * repeat * 2

    # build nauty commands

    # command = f"echo \"n={natoms};p;g"
    command = f"n={natoms};p;g"
    listcommand = f'echo "n={natoms};'
    edges = []
    for i, j in zip(*np.where(ds < threshold)):
        if i in range(4 * natoms, 5 * natoms) or j in range(4 * natoms, 5 * natoms):
            idx = sorted([i % natoms, j % natoms])
            if len(set(idx)) > 1 and idx not in edges:
                edges.append(idx)
    for i in range(natoms):
        others = [str(_[1]) for _ in edges if _[0] == i]
        command += " ".join(others) + ";"
        listcommand += f"{i}:" + " ".join(others) + ";"

    command += "x"  # | ./dreadnaut | grep -v level | grep -v orbit | grep -v cpu " + "| sed 's/$/-/' | tr -d '\\n' | sed 's/-  [ ]*/ /g;s/-[ ]*/\\n/g'"   f" > g{n}"
    listcommand += "\" | ./dretog | ./vcolg -T -m2 | awk '{ if (($"
    listcommand += "+$".join([str(_) for _ in range(3, 3 + natoms)])
    listcommand += f") == {natoms/2})" + " print }' | wc -l"

    # run nauty to get automorphisms
    output = subprocess.check_output(
        f"bash counts.sh {n*n} {n*n}".split(), input=command.encode("ascii")
    )
    return int(output.decode("ascii"))


for i in range(2, 10):
    print(i, spatial_symmetry(i))


# %%
# run all comamnds before to build g* files
# call polya.py from gh:rosenbrockc/polya as python polya.py n**2 n**2 -generators g$(n)
ns = (2, 3, 4, 5, 6, 7, 8, 9)
unique = (
    6,
    276,
    3139680,
    421373161854,
    1024334597808675200,
    43329272549438609063673856,
    31186388075427214057179489943683072,
    376448337550454757640360161491749551659286528,
)
combinatorial = [get_combinatorial(n) for n in ns]
# %%
ns = np.array(ns)
plt.semilogy(ns * ns * 2, combinatorial, "o-", label="Combinatorial")
plt.semilogy(ns * ns * 2, unique, "o-", label="Unique after spatial symmetry")
plt.semilogy(ns * ns * 2, np.array(unique) / 2, "o-", label="Covered by alchemy")
plt.legend()
plt.xlabel("Number of atoms")
plt.ylabel("Number of BN-doped systems")
# %%
