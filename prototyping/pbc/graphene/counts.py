#!/usr/bin/env python
# counts realisations of unique dopant patterns in BN-doped graphene
# assumes a regular lattice and a n x n super cell

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as ssd
import scipy.special as scs
import subprocess
import networkx as nx

# region
# n = numbers of unitcells
def get_combinatorial(n):
    natoms = n * n * 2
    return scs.comb(natoms, natoms / 2)


def get_graphene_graph(repeat):
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
    edges = []
    for i, j in zip(*np.where(ds < threshold)):
        if i in range(4 * natoms, 5 * natoms) or j in range(4 * natoms, 5 * natoms):
            idx = sorted([i % natoms, j % natoms])
            if len(set(idx)) > 1 and idx not in edges:
                edges.append(idx)
    return (
        natoms,
        edges,
        f"{int(natoms/2)} {int(natoms/2)}",
        f"{int(natoms/4)} {int(natoms/4)} {int(natoms/2)}",
    )


def get_cubic(repeat):
    g = nx.grid_graph(dim=(repeat, repeat, repeat), periodic=True)
    g = nx.convert_node_labels_to_integers(g)
    return (
        repeat ** 3,
        g.edges,
        f"{int(repeat**3/2)} {int(repeat**3/2)}",
        f"{int(repeat**3/4)} {int(repeat**3/4)} {int(repeat**3/2)}",
    )


def get_chain(repeat):
    edges = [[0, repeat - 1]] + [[_, _ + 1] for _ in range(repeat - 1)]
    return (
        repeat,
        edges,
        f"{int(repeat/2)} {int(repeat/2)}",
        f"{int(repeat/4)} {int(repeat/4)} {int(repeat/2)}",
    )


def spatial_symmetry(n, kind, half=False):
    # natoms, edges = get_graphene_graph(n)
    natoms, edges, fullargs, halfargs = kind(n)

    # build nauty commands
    command = f"n={natoms};p;g"
    for i in range(natoms):
        others = [str(_[1]) for _ in edges if _[0] == i]
        command += " ".join(others) + ";"
    command += "x"
    print(command)

    # run nauty to get automorphisms
    try:
        cmd = f"bash counts.sh {fullargs}".split()
        if half:
            cmd = f"bash counts.sh {halfargs}".split()
        output = subprocess.check_output(
            cmd,
            input=command.encode("ascii"),
        )
        return int(output.decode("ascii"))
    except:
        return None


for i in range(2, 20):
    print(
        i, spatial_symmetry(i, get_chain, half=True)
    )  # , spatial_symmetry(i, half=True))


# %%
# cubic 64 full 39785643746726
# cubic 64 half
# cubic 8 full 6
# cubic 8 half 16

# ns = (2, 3, 4, 5, 6, 7, 8, 9)

# %%
ns = np.arange(2, 10)
# graphene_full = np.array([spatial_symmetry(_, half=False) for _ in ns])
# graphene_half = np.array([spatial_symmetry(_, half=True) for _ in ns])
# css = [spatial_symmetry(_, get_chain, half=False) for _ in cs]
# cssh = [spatial_symmetry(_, get_chain, half=True) for _ in cs]
plt.semilogy(
    ns ** 2 * 2, graphene_full / 2, "h-", label="graphene 100% doped", markersize=10
)
plt.semilogy(
    ns[graphene_half != None] ** 2 * 2,
    graphene_half[graphene_half != None] / 2,
    "h-",
    label="graphene 50% doped",
    markersize=10,
)
plt.semilogy(
    (8, 64), (6, 39785643746726), "s-", label="cubic 100% doped", markersize=10
)

cs = (20, 40, 60, 80, 100, 120, 140, 160)
plt.semilogy(cs, css, "-", label="chain 100% doped")
plt.semilogy(cs, cssh, "-", label="chain 50% doped")

plt.legend()
plt.xlabel("Number of atoms")
plt.ylabel("Number of alchemical enantiomers")
plt.xlim(0, 80)
plt.ylim(1, 1e33)
# %%
