#!/usr/bin/env python
import numpy as np

q = np.loadtxt("out-2", dtype=np.int8)
print (q)
q.tofile("bin")
