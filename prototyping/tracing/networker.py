#!/usr/bin/env python
# region
import numpy as np
import pandas as pd
import rmsd
import geopolate
import os
import glob
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
    new = []
    for logfile in glob.glob("production/log*log"):
        origin = int(logfile.split("_")[1].split("-")[0])
        destination = int(logfile.split("-")[1].split(".")[0])

        completed.append(origin)

        with open(logfile) as fh:
            lines = fh.readlines()
        
        if "OK" in lines[-1]:
            new.append(destination)

    remaining = set(np.arange(1, 6096)) - set(completed) - set(new)
    pairs = []
    for existing in new:
        for r in remaining:
            pairs.append((existing, r))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        l = list(executor.map(get_task, pairs))

    for element in l:
        if len(element) > 0:
            print (element)