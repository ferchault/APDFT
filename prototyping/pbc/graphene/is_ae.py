#!/usr/bin/env python
# works only for fully BN-doped graphene

#%%
import numpy as np
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import functools
import pynauty

#%%
@functools.lru_cache(maxsize=10)
def get_graph(n):
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
    edges = []
    for i, j in zip(*np.where(ds < threshold)):
        if  (
            i in range(4 * natoms, 5 * natoms) or j in range(4 * natoms, 5 * natoms)
        ):
            idx = sorted([i % natoms, j % natoms])
            if len(set(idx)) > 1 and idx not in edges:
                edges.append(idx)
    # build nauty object    
    g = pynauty.Graph(n*n*2)
    for i in range(n*n*2):
        others = [_[1] for _ in edges if _[0] == i]
        g.connect_vertex(i, others)

    return g
# %%
def get_orbits(n, dZ):
    g = get_graph(n)
    g.set_vertex_coloring([set(np.where(dZ== 1)[0])])
    generators, grpsize1, grpsize2, orbits, numorbits = pynauty.autgrp(g)
    print (grpsize1, grpsize2)
    return numorbits
    orbitids = set(orbits)
    retval = []
    for orbitid in orbitids:
        retval.append(np.where(np.array(orbits) == orbitid))
    return retval

def is_special_ae(graph, color1, color2):
    reference = (color1 + color2)/2
    orbits = get_orbits(graph, reference)
    for orbit in orbits:
        if sum(color1[orbit]) != 0 or sum(color2[orbit]) != 0:
            return False
    return True

def is_general_ae(graph, color1, color2):
    automorphisms = get_automorphisms(graph)
    for transform in automorphisms:
        if is_special_ae(graph, color1[transform], color2):
            return True
    return False

q = []
for i in range(1):
    n = 8
    dZ = np.ones(n*n*2)
    dZ[:n*n] *= -1
    #np.random.shuffle(dZ)
    q.append(get_orbits(n, dZ))
np.bincount(q), q[0]
