#!/usr/bin/env python
# region
import numpy as np
import pandas as pd
import scipy.optimize as sco
import glob
import functools
# region
class FollowMe:
    def __init__(self, filename, labels=None):
        self._calcs = pd.read_hdf(filename, key="calcs")
        self._sims = pd.read_hdf(filename, key="sims")
        self._nmos = len(self._calcs.occ.values[0])
        self.lvals = sorted(self._calcs.pos.values)
        self._connect(labels)
        

    def _connect(self, labels):
        if labels is None:
            labels = np.array([f"MO-{_}" for _ in range(self._nmos)])
        else:
            labels = labels._positions[-1]
        positions = []
        for idx, destination in enumerate(self.lvals):
            if idx == 0:
                ranking = labels
            else:
                origin = self.lvals[idx - 1]
                sim = self._sims.query("origin == @origin & destination == @destination").sim.values[0]

                row, col = sco.linear_sum_assignment(sim, maximize=True)
                ranking = positions[-1][np.argsort(col)]
            positions.append(ranking)
        self._positions = np.array(positions)
    
    def get_labeled_energies(self, globalreference=False):
        if globalreference:
            idx = 0
        else:
            idx = -1
        return dict(zip(self._positions[idx], self._calcs.sort_values("pos").energies.values[idx]))

def split_name(filename):
    seg = filename.split("_")[-1]
    origin = int(seg.split("-")[0])
    destination = int(seg.split("-")[1].split(".")[0])
    return origin, destination

def get_labeled_energies(idx, toplevel=True):
    if idx == 1:
        fn = glob.glob(f"/lscratch/vonrudorff/tracing/production/path_1-*.h5")[0]
        return FollowMe(fn).get_labeled_energies(globalreference=True)
    fn = glob.glob(f"/lscratch/vonrudorff/tracing/production/*-{idx}.h5")[0]
    reference, _ = split_name(fn)
    if reference != 1:
        reference = get_labeled_energies(reference, toplevel=False)
    else:
        reference = None
    ret = FollowMe(fn, reference)
    if toplevel:
        return ret.get_labeled_energies()
    else:
        return ret

# region
@functools.lru_cache(maxsize=1)
def extract_table():
    valids = [1] + [split_name(_)[1] for _ in glob.glob("/lscratch/vonrudorff/tracing/production/*.h5")]
    return {_: get_labeled_energies(_) for _ in valids}
# region
