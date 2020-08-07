#!/usr/bin/env python
import sys
from math import gcd
from functools import reduce

def bonds_array(colors, vertices, elements, nv):
	bonds = {}
	selements = sorted(elements)
	for i in range(len(selements)):
		for j in range(i, len(selements)):
			bonds[selements[i] + selements[j]] = 0
	
	vertexranks = [0 for _ in range(nv)]
	for a, b in zip(vertices[::2], vertices[1::2]):
		vertexranks[a] += 1
		vertexranks[b] += 1
		element_a = elements[colors[a]]
		element_b = elements[colors[b]]
		element_a, element_b = sorted((element_a, element_b))
		bonds[element_a + element_b] += 1

	# fill CX bonds
	for vertex in range(nv):
		if vertexranks[vertex] > 3:
			raise ValueError("No rank > 3 allowed")
		for missing in range(3-vertexranks[vertex]):
			element_a = elements[colors[vertex]]
			element_b = elements[0]
			element_a, element_b = sorted((element_a, element_b))
			bonds[element_a + element_b] += 1
	return bonds
	
def angles_array(colors, vertices, elements, nv):
	selements = sorted(elements)
	angles = {}
	slements = sorted(elements)
	for i in range(len(selements)):
		for j in range(i, len(selements)):
			for k in selements:
				angles[selements[i] + k + selements[j]] = 0
	
	bonded_to = []
	vertexranks = [0 for _ in range(nv)]
	for i in range(nv):
		bonded_to.append([])
	
	for a, b in zip(vertices[::2], vertices[1::2]):
		bonded_to[a].append(b)
		bonded_to[b].append(a)
		vertexranks[a] += 1
		vertexranks[b] += 1

	for i in range(nv):
		for j in range(i+1, nv):
			for k in range(nv):
				if k == i or k == j:
					continue
				if k in bonded_to[i] and k in bonded_to[j]:
					# j-k-i
					a = elements[colors[i]]
					b = elements[colors[j]]
					c = elements[colors[k]]
					label = min(a, b) + c + max(a, b)
					angles[label] += 1
	
	for i in range(nv):
		for bt in bonded_to[i]:
			a = elements[colors[bt]]
			c = elements[colors[i]]
			b = elements[0]
			label = min(a, b) + c + max(a, b)
			if vertexranks[i] == 1:
				angles[label] += 2
			if vertexranks[i] == 2:
				angles[label] += 1
		a = elements[colors[i]]
		b = elements[0]
		c = elements[0]
		label = min(a, b) + c + max(a, b)
		if vertexranks[i] == 2:
			angles[label] += 2
		if vertexranks[i] == 1:
			angles[label] += 4
			label = b + a + c
			angles[label] += 1

	#for k, v in angles.items():
	#	if v != 0:
	#		print (k, v)
	return angles
	

def simplify(lhs, rhs):
	values = []
	for k in lhs.keys():
		remove = min(lhs[k], rhs[k])
		lhs[k] -= remove
		rhs[k] -= remove
		if lhs[k] > 0:
			values.append(lhs[k])
		if rhs[k] > 0:
			values.append(rhs[k])
	if len(values) > 0:
		red = reduce(gcd,values)
		if red != 1:
			for k in lhs.keys():
				lhs[k] //= red
				rhs[k] //= red
	
	return lhs, rhs
	
def eformat(lhs, rhs):
	lhsstr = ""
	rhsstr = ""
	for k in sorted(lhs.keys()):
		if lhs[k] > 0:
			numrep = ""
			if lhs[k] > 1:
				numrep = str(lhs[k])
			lhsstr += "+%s%s" % (numrep, k)
		if rhs[k] > 0:
			numrep = ""
			if rhs[k] > 1:
				numrep = str(rhs[k])
			rhsstr += "+%s%s" % (numrep, k)
	lhsstr, rhsstr = sorted((lhsstr, rhsstr))
	eqnstr = (lhsstr.strip('+') + "=" + rhsstr.strip("+"))
	return eqnstr


# nv ne {col} {v1 v2}
# 4 3 0 0 0 0  0 3 1 3 2 3
deltaz = [0, -1, 1, -2, 2, -3, 3]
elements_up = "CBDAEFG"
elements_dn = "CDBEAGF"
shown = []
ntuples = int(sys.argv[1])
if ntuples == 2:
	DOBONDS = True
if ntuples == 3:
	DOBONDS = False
for line in sys.stdin:
	parts = line.strip().split()
	nv = int(parts[0])
	ne = int(parts[1])
	colors = parts[2:2+nv]
	colors = [int(_) for _ in colors]
	vertices = [int(_) for _ in parts[2+nv:]]

	lhs = bonds_array(colors, vertices, elements_up, nv)
	rhs = bonds_array(colors, vertices, elements_dn, nv)
	lhs, rhs = simplify(lhs, rhs)	
	eqnstr = eformat(lhs, rhs)
	if DOBONDS:
		if eqnstr != "=" and eqnstr not in shown:
			print (eqnstr, "#", line.strip())
			shown.append(eqnstr)
		continue

	if eqnstr != "=":
		continue
	lhs = angles_array(colors, vertices, elements_up, nv)	
	rhs = angles_array(colors, vertices, elements_dn, nv)
	lhs, rhs = simplify(lhs, rhs)	
	eqnstr = eformat(lhs, rhs)
	
	if eqnstr != "=" and eqnstr not in shown:
		shown.append(eqnstr)
		print (eqnstr, "#", line.strip())
