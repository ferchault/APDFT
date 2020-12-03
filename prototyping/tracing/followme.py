#!/usr/bin/env python
# region
import numpy as np
import pandas as pd
import scipy.optimize as sco
import glob
import functools
import scipy.signal as scs
import pyscf
import pyscf.gto
import pyscf.qmmm
import pyscf.scf
from pyscf.data import nist

BASE = 3753
CROSSING_MAX_GAP = 0.01
DELTA_LAMBDA_BUFFER = 1 / 40
# region
class FollowMe:
    def __init__(self, filename, labels=None):
        self._calcs = pd.read_hdf(filename, key="calcs")
        self._sims = pd.read_hdf(filename, key="sims")
        self._nmos = len(self._calcs.occ.values[0])
        self.lvals = sorted(self._calcs.pos.values)
        self._extract_energies()
        self._connect(labels)
        self._adjust_crossings()

    def _extract_energies(self):
        self._sorted_energies = np.vstack(self._calcs.sort_values("pos").energies.values)

    def _find_crossing_candidates(self, idA, idB):
        # inspired by 10.1016/j.physa.2010.12.017
        xs = self._sims.origin.values
        dl = (self._sims.destination - self._sims.origin).values
        fidelityA = np.array([_[idA, idA] for _ in self._sims.sim])
        fidelityB = np.array([_[idB, idB] for _ in self._sims.sim])
        ys = (1 - fidelityA) / dl ** 2 * (1 - fidelityB) / dl ** 2

        signal = np.interp(np.arange(max(xs)), xs, ys)
        peaks, _ = scs.find_peaks(signal, height=0.2 * max(signal))

        # force crossings to be within a given energy window
        deltaE = np.abs([_[idA] - _[idB] for _ in self._calcs.energies.values])
        deltaE = np.interp(np.arange(max(xs)), self._calcs.pos.values, deltaE)

        peaks = [_ for _ in peaks if deltaE[_] < CROSSING_MAX_GAP]
        return peaks
    
    def get_energy(self, moidx):
        pos = self._positions == f"MO-{moidx}"
        return self._sorted_energies[pos]
    def _get_mol(self, pos):
        row = self._calcs.query("pos == @pos")
        mol = pyscf.gto.Mole()
        atom = []
        for Z, coords in zip(row.Zs.values[0], row.coord.values[0]):
            atom.append(f"{Z} {coords[0]} {coords[1]} {coords[2]}")
        mol.atom = ";".join(atom)
        mol.basis = row.basis.values[0]
        mol.build()
        return mol, row.coeff.values[0]


    def _get_sim(self, origin, destination):
        mol_o, coeff_o = _get_mol(self, origin)
        mol_d, coeff_d = _get_mol(self, destination)

        s = pyscf.gto.intor_cross("int1e_ovlp", mol_o, mol_d)
        sim = np.abs(np.dot(np.dot(coeff_o.T, s), coeff_d))
        return sim


    def _adjust_crossings(self):
        maxstep = max(self._calcs.pos.values)
        crossings = self._find_all_crossing_candidates()

        for idx, row in crossings.sort_values("pos").iterrows():
            if row.one < 30 or row.one > 34:
                continue
            before = max(row.pos - maxstep * DELTA_LAMBDA_BUFFER, 0)
            before = self._calcs.query("pos <= @before").sort_values("pos").pos.values[-1]
            after = min(row.pos + maxstep * DELTA_LAMBDA_BUFFER, maxstep)
            after = self._calcs.query("pos >= @after").sort_values("pos").pos.values[0]

            sim = _get_sim(self, before, after)[row.one : row.other + 1, row.one : row.other + 1]
            before_calc_id = self.lvals.index(before)
            pos_calc_id = self.lvals.index(row.pos)
            after_calc_id = self.lvals.index(after)
            before_label_one, before_label_other = self._positions[before_calc_id, [row.one, row.other]]
            after_label_one, after_label_other = self._positions[after_calc_id, [row.one, row.other]]
            
            needs_exchange = False
            if np.linalg.det(sim) < 0:
                # check if exchange acutally needed
                if before_label_one == after_label_one:
                    needs_exchange = True
            else:
                # reverse exchange if done
                if before_label_one != after_label_one:
                    needs_exchange = True
            
            if needs_exchange:
                updated = self._positions[pos_calc_id:, :].copy()
                mask1 = updated == before_label_one
                mask2 = updated == before_label_other
                updated[mask1] = before_label_other
                updated[mask2] = before_label_one
                self._positions[pos_calc_id:, :] = updated

    def _find_all_crossing_candidates(self):
        rows = []
        for i in range(self._nmos - 1):
            for peak in self._find_crossing_candidates(i, i + 1):
                rows.append({"pos": peak, "one": i, "other": i + 1})
        return pd.DataFrame(rows)

    def _connect(self, labels):
        if labels is None:
            labels = np.array([f"MO-{_}" for _ in range(self._nmos)])
        else:
            labels = labels._positions[-1]
        positions = []
        #energies = []
        for idx, destination in enumerate(self.lvals):
            spectrum = self._calcs.query("pos == @destination").energies.values[0]
            if idx == 0:
                ranking = labels
                #energies.append(spectrum)
            else:
                origin = self.lvals[idx - 1]
                sim = self._sims.query(
                    "origin == @origin & destination == @destination"
                ).sim.values[0]

                row, col = sco.linear_sum_assignment(sim, maximize=True)
                ranking = positions[-1][np.argsort(col)]

                # add energies in that order
                #nener = np.zeros(len(ranking))

                #for molabel, moenergy in zip(ranking, spectrum):
                #    idx = int(molabel.split("-")[1])
                #    nener[idx] = moenergy
                #energies.append(nener)
            positions.append(ranking)
        self._positions = np.array(positions)
        #self._energies = np.array(energies)

    def get_labeled_energies(self, globalreference=False):
        if globalreference:
            idx = 0
        else:
            idx = -1
        return dict(
            zip(
                self._positions[idx],
                self._calcs.sort_values("pos").energies.values[idx],
            )
        )


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
    valids = [BASE] + [
        split_name(_)[1]
        for _ in glob.glob("/lscratch/vonrudorff/tracing/production/*.h5")
    ]
    ret = {}
    for _ in valids:
        try:
            ret[_] = get_labeled_energies(_)
        except:
            continue
    return ret


# region
if __name__ == "__main__":
    tbl = extract_table()
    import pickle

    with open("/lscratch/vonrudorff/tracing/production/table.pkl", "wb") as fh:
        pickle.dump(tbl, fh)
