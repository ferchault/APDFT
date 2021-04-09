import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from config import *
import sys

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

#Initalize all variables:
plt.rcParams.update({'font.size': 14})
np.set_printoptions(threshold=sys.maxsize)
color = ['#407294', '#ffa500', '#065535', '#800000', '#660066', '#310c0c', '#f7347a', '#696966']

#Get the data:
carbon_norms = []
nitrogen_norms = []
oxygen_norms = []
fluorine_norms = []

#hybridisation
sp_C_hybrid = []
sp2_C_hybrid = []
sp3_C_hybrid = []
sp_N_hybrid = []
sp2_N_hybrid = []
sp3_N_hybrid = []
sp_O_hybrid = []
sp2_O_hybrid = []
sp3_O_hybrid = []
f = open('logs/QM9_atomic_norms_log.txt', 'r')
Lines = f.readlines()
for line in Lines:
    x = line.split('\t') #0: Name   1: Chemical Element    2: Index   3: SMILES  4: Norm
    print(x[0])





    #Get the smiles and the indices of the hybridzed atoms
    sp_indices = list(Chem.MolFromSmiles(x[1]).GetSubstructMatches(Chem.MolFromSmarts('[^1]'))[0])
    sp2_indices = list(Chem.MolFromSmiles(x[1]).GetSubstructMatches(Chem.MolFromSmarts('[^2]'))[0])
    sp3_indices = list(Chem.MolFromSmiles(x[1]).GetSubstructMatches(Chem.MolFromSmarts('[^3]'))[0])
    #Still borked, list of zero length needs special case


    if x[1] == 'C':
        carbon_norms.append(float(x[4]))
        if int(x[2]) in sp3_indices:
            sp3_C_hybrid.append(float(x[4]))
        if int(x[2]) in sp2_indices:
            sp2_C_hybrid.append(float(x[4]))
        if int(x[2]) in sp_indices:
            sp_C_hybrid.append(float(x[4]))
    if x[1] == 'N':
        nitrogen_norms.append(float(x[4]))
        if int(x[2]) in sp3_indices:
            sp3_N_hybrid.append(float(x[4]))
        if int(x[2]) in sp2_indices:
            sp2_N_hybrid.append(float(x[4]))
        if int(x[2]) in sp_indices:
            sp_N_hybrid.append(float(x[4]))
    if x[1] == 'O':
        oxygen_norms.append(float(x[4]))
        if int(x[2]) in sp3_indices:
            sp3_O_hybrid.append(float(x[4]))
        if int(x[2]) in sp2_indices:
            sp2_O_hybrid.append(float(x[4]))
        if int(x[2]) in sp_indices:
            sp_O_hybrid.append(float(x[4]))
    if x[1] == 'F':
        fluorine_norms.append(float(x[4]))



num_bins = 100
alpha = 0.65

plt.hist(carbon_norms, bins=num_bins, label='C (20% height)', color=color[0], alpha=alpha, weights=0.2*0.001*np.ones((len(carbon_norms))))
plt.hist(oxygen_norms, bins=num_bins, label='O', alpha=alpha, color=color[1], weights=0.001*np.ones((len(oxygen_norms))))
plt.hist(nitrogen_norms, bins=num_bins, label='N', alpha=alpha, color=color[3], weights=0.001*np.ones((len(nitrogen_norms))))
plt.hist(fluorine_norms, bins=num_bins, label='F', alpha=alpha, color=color[4], weights=0.001*np.ones((len(fluorine_norms))))
#hybridisations:
plt.hist(sp3_C_hybrid, bins=num_bins, label=r'C $sp^3$ (20% height)', alpha=alpha, histtype='step', fill=False, linestyle='solid', ec=color[0], weights=0.2*0.001*np.ones((len(sp3_C_hybrid))))
plt.hist(sp2_C_hybrid, bins=num_bins, label=r'C $sp^2$ (20% height)', alpha=alpha, histtype='step', fill=False, linestyle='dashed', ec=color[0], weights=0.2*0.001*np.ones((len(sp2_C_hybrid))))
plt.hist(sp_C_hybrid, bins=num_bins, label=r'C $sp$ (20% height)', alpha=alpha, histtype='step', fill=False, linestyle='dotted', ec=color[0], weights=0.2*0.001*np.ones((len(sp_C_hybrid))))
plt.hist(sp3_N_hybrid, bins=num_bins, label=r'N $sp^3$', alpha=alpha, histtype='step', fill=False, linestyle='solid', ec=color[3], weights=0.001*np.ones((len(sp3_N_hybrid))))
plt.hist(sp2_N_hybrid, bins=num_bins, label=r'N $sp^2$', alpha=alpha, histtype='step', fill=False, linestyle='dashed', ec=color[3], weights=0.001*np.ones((len(sp2_N_hybrid))))
plt.hist(sp_N_hybrid, bins=num_bins, label=r'N $sp$', alpha=alpha, histtype='step', fill=False, linestyle='dotted', ec=color[3], weights=0.001*np.ones((len(sp_N_hybrid))))
plt.hist(sp3_O_hybrid, bins=num_bins, label=r'O $sp^3$', alpha=alpha, histtype='step', fill=False, linestyle='solid', ec=color[1], weights=0.001*np.ones((len(sp3_O_hybrid))))
plt.hist(sp2_O_hybrid, bins=num_bins, label=r'O $sp^2$', alpha=alpha, histtype='step', fill=False, linestyle='dashed', ec=color[1], weights=0.001*np.ones((len(sp2_O_hybrid))))
#plt.hist(sp_O_hybrid, bins=num_bins, label=r'O $sp$', alpha=alpha, histtype='step', fill=False, linestyle='dotted', ec=color[1], weights=0.001*np.ones((len(sp_O_hybrid))))
plt.legend(loc="upper right",prop={'size': 10},framealpha=1, edgecolor='black')
plt.xlim([100,380])
plt.ylabel(r'Number / $10^3$')
plt.xlabel(r'$L_2$-norm of atomic representation')
plt.savefig('figures/QM9_atomic_norms.png', dpi=500)
