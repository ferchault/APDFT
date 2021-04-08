import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os
import sys

#Initalize all variables:
plt.rcParams.update({'font.size': 14})
np.set_printoptions(threshold=sys.maxsize)
color = ['#407294', '#ffa500', '#065535', '#800000', '#660066', '#310c0c', '#f7347a', '#696966']

#Get the data:
carbon_norms = []
nitrogen_norms = []
oxygen_norms = []
fluorine_norms = []
f = open('logs/QM9_atomic_norms_log.txt', 'r')
Lines = f.readlines()
for line in Lines:
    x = line.split('\t')
    if x[0] == 'C':
        carbon_norms.append(float(x[1]))
    if x[0] == 'N':
        nitrogen_norms.append(float(x[1]))
    if x[0] == 'O':
        oxygen_norms.append(float(x[1]))
    if x[0] == 'F':
        fluorine_norms.append(float(x[1]))

num_bins = 100
alpha = 0.6

plt.hist(carbon_norms, bins=num_bins, label='C (20% height)', alpha=alpha, weights=0.2*0.001*np.ones((len(carbon_norms))))
plt.hist(oxygen_norms, bins=num_bins, label='O', alpha=alpha, weights=0.001*np.ones((len(oxygen_norms))))
plt.hist(nitrogen_norms, bins=num_bins, label='N', alpha=alpha, weights=0.001*np.ones((len(nitrogen_norms))))
plt.hist(fluorine_norms, bins=num_bins, label='F', alpha=alpha, weights=0.001*np.ones((len(fluorine_norms))))
plt.legend(loc="upper right",prop={'size': 10},framealpha=1, edgecolor='black')
plt.xlim([100,330])
plt.ylabel(r'Number / $10^3$')
plt.xlabel(r'$L_2$-norm of atomic representation')
plt.savefig('figures/QM9_atomic_norms.png', dpi=500)
