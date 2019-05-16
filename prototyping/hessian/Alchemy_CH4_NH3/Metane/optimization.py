from horton import *
import numpy as np

Methane=IOData.from_file('Methane.xyz')

arrayGeom= np.asarray(Methane.coordinates)
print np.linalg.norm(np.asarray([3,4,5]))
for x in arrayGeom:
    for y in arrayGeom:
        print np.linalg.norm(x-y)
