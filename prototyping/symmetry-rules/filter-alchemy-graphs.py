#!/usr/bin/env python
import sys

# nv ne {col} {v1 v2}
# 4 3 0 0 0 0  0 3 1 3 2 3
deltaz = [0, -1, 1, -2, 2, -3, 3]
for line in sys.stdin:
	parts = line.strip().split()
	nv = int(parts[0])
	colors = parts[2:2+nv]
	colors = [int(_) for _ in colors]
	balance = 0
	maxabs = 0
	deltas = [0,0,0, 0, 0]
	for color in colors:
		d = deltaz[color]
		balance += d
		maxabs = max(abs(d), maxabs)
		deltas[abs(d)] += d/max(abs(d), 1)

	cancel = False
	for e in deltas:
		if abs(e) > 0:
			cancel = True
	if cancel:
		continue
	 
	if balance == 0 and maxabs > 0:
		print (line.strip())
	
