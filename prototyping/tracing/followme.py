#!/usr/bin/env python
# region
import numpy as np
import pandas as pd
import scipy.optimize as sco
import glob
import functools
import scipy.signal as scs

BASE = 3753
CROSSING_MAX_GAP = 0.01
# region
class FollowMe:
    def __init__(self, filename, labels=None):
        self._calcs = pd.read_hdf(filename, key="calcs")
        self._sims = pd.read_hdf(filename, key="sims")
        self._nmos = len(self._calcs.occ.values[0])
        self.lvals = sorted(self._calcs.pos.values)
        self._connect(labels)

    def find_crossing_candidates(self, idA, idB):
        # inspired by 10.1016/j.physa.2010.12.017
        xs = self._sims.origin.values
        dl = (self._sims.destination - self._sims.origin).values
        fidelityA = np.array([_[idA,idA] for _ in self._sims.sim])
        fidelityB = np.array([_[idB,idB] for _ in self._sims.sim])
        ys = (1-fidelityA)/dl**2 * (1-fidelityB)/dl**2

        signal = np.interp(np.arange(max(xs)), xs, ys)
        peaks, _ = scs.find_peaks(signal, height=0.2*max(signal))

        # force crossings to be within a given energy window
        deltaE = np.abs([_[idA] - _[idB] for _ in self._calcs.energies.values])
        deltaE = np.interp(np.arange(max(xs)), self._calcs.pos.values, deltaE)

        peaks = [_ for _ in peaks if deltaE[_] < CROSSING_MAX_GAP]
        return peaks

    def _connect(self, labels):
        if labels is None:
            labels = np.array([f"MO-{_}" for _ in range(self._nmos)])
        else:
            labels = labels._positions[-1]
        positions = []
        energies = []
        for idx, destination in enumerate(self.lvals):
            spectrum = self._calcs.query("pos == @destination").energies.values[0]
            if idx == 0:
                ranking = labels
                energies.append(spectrum)
            else:
                origin = self.lvals[idx - 1]
                sim = self._sims.query("origin == @origin & destination == @destination").sim.values[0]

                row, col = sco.linear_sum_assignment(sim, maximize=True)
                ranking = positions[-1][np.argsort(col)]

                # add energies in that order
                nener = np.zeros(len(ranking))
                
                for molabel, moenergy in zip(ranking, spectrum):
                    idx = int(molabel.split("-")[1])
                    nener[idx] = moenergy
                energies.append(nener)
            positions.append(ranking)
        self._positions = np.array(positions)
        self._energies = np.array(energies)
    
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
    if idx == BASE:
        fn = glob.glob(f"/lscratch/vonrudorff/tracing/production/path_{BASE}-*.h5")[0]
        return FollowMe(fn).get_labeled_energies(globalreference=True)
    fn = glob.glob(f"/lscratch/vonrudorff/tracing/production/*-{idx}.h5")[0]
    reference, _ = split_name(fn)
    if reference != BASE:
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
    valids = [BASE] + [split_name(_)[1] for _ in glob.glob("/lscratch/vonrudorff/tracing/production/*.h5")]
    ret = {}
    for _ in valids:
        try:
            ret[_] = get_labeled_energies(_)
        except:
            continue
    return ret
# region
if __name__ == '__main__':
    tbl = extract_table()
    import pickle 
    with open("/lscratch/vonrudorff/tracing/production/table.pkl", "wb") as fh:
        pickle.dump(tbl, fh)

# region
d = FollowMe("/lscratch/vonrudorff/tracing/production/path_3753-1210.h5")
d.find_crossing_candidates(33, 34)
# region
d._positions[-1][30:34]
# region
subset = []
for energies in d._calcs.sort_values("pos").energies.values:
    subset.append(energies[30:35])
subset = np.array(subset)
# region
#-import matplotlib.pyplot as plt
for i in range(subset.shape[-1]-1):
    plt.plot(d.lvals, subset[:, i])
# region
for i in range(30, 34):
    plt.plot(d.lvals, d._energies.T[i])
plt.xlim(13000, 2**14)
 # region
d._sims.query("origin > 1000 & origin < 1200")
# region
d._sims.query("origin == 1152").sim.values[0][30:32, 30:32]
# region
d.get_labeled_energies()
# region
import matplotlib.pyplot as plt
# region

def find_crossing_candidates(idA, idB):
    # inspired by 10.1016/j.physa.2010.12.017
    xs = d._sims.origin.values
    dl = (d._sims.destination - d._sims.origin).values
    fidelityA = np.array([_[idA,idA] for _ in d._sims.sim])
    fidelityB = np.array([_[idB,idB] for _ in d._sims.sim])
    ys = (1-fidelityA)/dl**2 * (1-fidelityB)/dl**2

    signal = np.interp(np.arange(max(xs)), xs, ys)
    peaks, _ = scs.find_peaks(signal, height=0.2*max(signal))
    return peaks

compare_levels(32, 33)
# region
for i in range(subset.shape[-1]-1):
    plt.plot(d.lvals, subset[:, i])

for idxA in range(30, 34):
    peaks = compare_levels(idxA, idxA+1)
    for peak in peaks:
        plt.axvline(peak, ymin=(idxA-30)/5)
    print (idxA, peaks)
# region
d._calcs
# region
