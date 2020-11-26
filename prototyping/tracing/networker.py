#!/usr/bin/env python
# region
import numpy as np
import pandas as pd
import rmsd
import geopolate
#endregion

#region
basepath = "/home/grudorff/data/qm9-isomers"

def get_distance(idxA, idxB):
    i = geopolate.Interpolate(f"{basepath}/ci{idxA:04d}.xyz", f"{basepath}/ci{idxB:04d}.xyz")
    xs = np.linspace(0, 1, 50)
    ls = []
    for a, b in zip(xs[:-1], xs[1:]):
        ls.append(np.linalg.norm(i._geometry(a) - i._geometry(b)))
    return np.sum(ls)

# endregion
# region
completed = [1]
remaining = np.arange(2, 6096)
np.random.shuffle(remaining)
# region
ds = []
pairs = []
for i in range(100):
    for c in completed:
        d =get_distance(c, remaining[i])
        pairs.append((remaining[i], c))
        ds.append(d)
        print (c, remaining[i], d)
# region
ds
# region
6000/30/
# region
[pairs[_] for _ in np.argsort(ds)[:10]]
# region
