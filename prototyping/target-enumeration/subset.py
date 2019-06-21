#!/usr/bin/env python
import sys
import qml
import numpy as np

c = qml.Compound('/scicore/home/lilienfeld/rudorff/c20-target-manual/scripts/c20.xyz')
def get_rep(ncharges):
    return qml.representations.generate_eigenvalue_coulomb_matrix(ncharges, c.coordinates)
def get_distance(existing, candidate):
    return np.linalg.norm(np.array(existing) - np.array(candidate), axis=1)
def read_list(fn):
	combs = []
	for line in open(fn).readlines():
		values = [int(_) for _ in line.strip()]
		combs.append(values)
	return combs

combs = read_list(sys.argv[1])

existing = []
if len(sys.argv) == 3:
	# merge files
	for comb in combs:
		thisrep = get_rep(comb)
		existing.append(thisrep)
	print (''.join(map(str, comb)))
	combs = read_list(sys.argv[2])

for comb in combs:
	thisrep = get_rep(comb)
	if len(existing) == 0:
		existing.append(thisrep)
		print (''.join(map(str, comb)))
		continue
	distances = get_distance(existing, thisrep)
	if np.min(distances) < 1e-2:
		continue
	existing.append(thisrep)
	print (''.join(map(str, comb)))
