#!/usr/bin/env python
import sys
import numpy as np

mollist = sys.argv[1]
rankfile = sys.argv[2]
xyzfile = sys.argv[3]
rankid = int(sys.argv[4])
molecules = np.fromfile(mollist, dtype=np.int8).reshape(-1, 22)
ranking = np.fromfile(rankfile, dtype=np.int32)

molid = ranking[rankid]
molecule = list(molecules[molid])

with open(xyzfile) as fh:
	lines = fh.readlines()
	for i in range(2):
		print (lines.pop(0).strip())
	for line in lines:
		if len(molecule) == 0:
			print (line.strip())
		else:
			print ('CNB'[molecule.pop(0)], ' '.join(line.strip().split()[1:]))
