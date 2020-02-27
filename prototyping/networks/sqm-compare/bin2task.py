#!/usr/bin/env python
import numpy as np
import sys
import tqdm

mollist = sys.argv[1]

charges = np.fromfile(mollist, dtype=np.int8).reshape(-1, 22)

coords = []
for line in open('inp.xyz').readlines()[2:]:
	parts = line.strip().split()
	coords.append(' '.join(parts[1:]))

lookup = 'CNB'
for mol in tqdm.tqdm(range(len(charges))):
	task = ""
	for parts in zip([lookup[_] for _ in charges[mol]] + ['H']*14, coords):
		task += ' '.join(parts) + "###"
	print ("36######" + task[:-3])

