#!/usr/bin/env python
import numpy as np
import sys

mollist = sys.argv[1]

charges = np.fromfile(mollist, dtype=np.int8).reshape(-1, 22)

coords = []
for line in open('inp.xyz').readlines()[2:]:
	parts = line.strip().split()
	coords.append(' '.join(parts[1:]))

lookup = '_____BCN'
for mol in range(len(charges))[:1000]:
	task = ""
	for parts in zip([lookup[_] for _ in charges[mol]] + ['H']*14, coords):
		task += ' '.join(parts) + "###"
	print ("36######" + task[:-3])

