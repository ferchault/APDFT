#!/usr/bin/env python
import sympy
import numpy as np
import sys

TOKENLEN = int(sys.argv[2])

with open(sys.argv[1]) as fh:
    lines = fh.readlines()
graphs = [_.split("#")[1] for _ in lines if "=" in _]
lines = [_.split("#")[0] for _ in lines if "=" in _]

# obtain symbols in this file
symbols = []
for line in lines:
    parts = line.replace("+", " ").replace("=", " ").split()
    for part in parts:
        var = part[-TOKENLEN:]
        if var not in symbols:
            symbols.append(var)

symbols = sorted(symbols)

# build equation matrix
mat = np.zeros((len(lines), len(symbols) + 1))
for lidx, line in enumerate(lines):
    lhs, rhs = line.split("=")
    for terms, sign in zip((lhs, rhs), (1, -1)):
        terms = terms.strip().split("+")
        for term in terms:
            var = term[-TOKENLEN:]
            factor = 1
            if len(term) > TOKENLEN:
                factor = int(term[:-TOKENLEN])
            factor *= sign
            mat[lidx, symbols.index(var)] = factor

# print (mat)
print("Check this for any =0 entries indicating that only trivial solutions exist")
solution = sympy.linsolve(sympy.Matrix(mat), [sympy.symbols(_) for _ in symbols])
for variable, element in zip(symbols, solution.args[0]):
    print(variable, "=", element)

# find the minimum set of relations and their corresponding graphs
reference_ref, pivots = sympy.Matrix(mat).rref()
print("Unique reduced row echelon form")
for idx in range(len(lines)):
    row = np.array(reference_ref[idx, :], np.float)
    if not np.allclose(row, row * 0):
        print(idx, row)

mat2 = mat.copy()
reference_ref = np.array(reference_ref, np.float)
reference_ref = reference_ref[np.abs(reference_ref).sum(axis=1) != 0]
while True:
    for i in range(len(mat2)):
        print(i)
        s = np.concatenate((mat2[:i, :], mat2[i + 1 :, :]))
        this_ref, this_pivots = sympy.Matrix(s).rref()
        this_ref = np.array(this_ref, np.float)
        this_ref = this_ref[np.abs(this_ref).sum(axis=1) != 0]
        if this_ref.shape != reference_ref.shape:
            continue
        if np.allclose(this_ref, reference_ref):
            mat2 = s
            del lines[i]
            del graphs[i]
            break
    else:
        break

print("Minimal graphs (vcolg format)")
for graph in graphs:
    print(graph.strip())

print("Unique reduced row echelon form (needs to be the same as above)")
solution = sympy.linsolve(sympy.Matrix(mat2), [sympy.symbols(_) for _ in symbols])
for variable, element in zip(symbols, solution.args[0]):
    print(variable, "=", element)
