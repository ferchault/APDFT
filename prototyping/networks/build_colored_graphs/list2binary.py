#!/usr/bin/env python
import numpy as np
data = np.loadtxt("out.list.gz", dtype=np.int32)

groups = data.sum(axis=1)
data += 6
data[data == 8] = 5

for nbn in set(groups):
	np.save("out-nbn-%d" % (nbn/3), data[groups == nbn])
