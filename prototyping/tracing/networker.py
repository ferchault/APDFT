#!/usr/bin/env python
# region
import numpy as np
import pandas as pd
import rmsd
import geopolate
import os
import sys
import concurrent.futures
#endregion

#region
basepath = "/lscratch/vonrudorff/tracing"
os.chdir(basepath)

def get_distance(idxA, idxB):
    i = geopolate.Interpolate(f"data/ci{idxA:04d}.xyz", f"data/ci{idxB:04d}.xyz")
    xs = np.linspace(0, 1, 50)
    ls = []
    for a, b in zip(xs[:-1], xs[1:]):
        ls.append(np.linalg.norm(i._geometry(a) - i._geometry(b)))
    return np.sum(ls)

def get_task(pair):
    origin, destination = pair
    d = get_distance(origin, destination)
    if d < 6:
        return f"python geopolate.py data/ci{origin:04d}.xyz data/ci{destination:04d}.xyz production/path_{origin}-{destination} > production/log_{origin}-{destination}.log"
    else:
        return ""

# endregion
# region
if __name__ == '__main__':
    completed = [1]
    remaining = np.arange(2, 6096)
    np.random.shuffle(remaining)

    pairs = []
    for r in remaining:
        pairs.append((completed[-1], r))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        l = list(executor.map(get_task, pairs))

    for element in l:
        if len(element) > 0:
            print (element)