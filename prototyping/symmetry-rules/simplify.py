#!/usr/bin/env python
import sympy
import numpy as np
import sys

with open(sys.argv[1]) as fh:
	lines = fh.readlines()
lines = [_.split("#")[0] for _ in lines if "=" in _]

# obtain symbols in this file
symbols = []
for line in lines:
	parts = line.replace("+", " ").replace("=", " ").split()
	for part in parts:
		var = part[-2:]
		if var not in symbols:
			symbols.append(var)

symbols = sorted(symbols)
print (symbols)

# build matrix
mat = np.zeros((len(lines), len(symbols)+1))
for lidx, line in enumerate(lines):
	lhs, rhs = line.split("=")
	for terms, sign in zip((lhs, rhs), (1, -1)):
		terms = terms.strip().split("+")
		for term in terms:
			var = term[-2:]
			factor = 1
			if len(term) > 2:
				factor = int(term[:-2])
			factor *= sign
			mat[lidx, symbols.index(var)] = factor

#print (mat)

ref, pivots = sympy.Matrix(mat).rref()

for idx in range(len(lines)):
	row = np.array(ref[idx, :], np.float)
	if not np.allclose(row, row*0):
		print (idx, row)

print (pivots)
