#!/usr/bin/env python

#%%
import sys

sys.path.append("../../src")
from apdft.math import IntegerPartitions
import numpy as np

# %%
degree = 2
alpha = 2
beta = 2
max_bonds_homo_alpha = 2

left = 0
for partition in IntegerPartitions.partition(degree * (2 * alpha + beta), 6):
    # none of the numbers can be odd
    all_even = True
    for q in partition:
        if (q % 2) == 1:
            all_even = False
    if not all_even:
        continue

    a, b, c, d, e, f = partition
    if a == d and c == e:
        continue  # no alchemical equation

    if a + b / 2 + c / 2 != degree * alpha:
        continue

    if d + b / 2 + e / 2 != degree * alpha:
        continue

    if f + e / 2 + c / 2 != degree * beta:
        continue

    if max(a, b / 2, d, e / 2) > max_bonds_homo_alpha:
        continue

    print(partition)

    left += 1
print(left)

# %%
2, 4, 0, 0, 4, 2
a, b, c, d, e, f
