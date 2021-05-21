import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from AlEn import *
import sys
from openbabel import openbabel as ob

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

    mol = ob.OBMol()
    conv = ob.OBConversion()
    conv.SetInFormat("smi")
    smiles = x[3]
    if not conv.ReadString(mol, smiles):
        print("Could not read SMILES")
    hyb = mol.GetAtom(int(x[2])+1).GetHyb()

    if x[1] == 'C':
        carbon_norms.append(float(x[4]))
        if hyb == 1:
            sp_C_hybrid.append(float(x[4]))
        if hyb == 2:
            sp2_C_hybrid.append(float(x[4]))
        if hyb == 3:
            sp3_C_hybrid.append(float(x[4]))
    if x[1] == 'N':
        nitrogen_norms.append(float(x[4]))
        if hyb == 1:
            sp_N_hybrid.append(float(x[4]))
        if hyb == 2:
            sp2_N_hybrid.append(float(x[4]))
        if hyb == 3:
            sp3_N_hybrid.append(float(x[4]))
    if x[1] == 'O':
        oxygen_norms.append(float(x[4]))
        if hyb == 1:
            sp_O_hybrid.append(float(x[4]))
        if hyb == 2:
            sp2_O_hybrid.append(float(x[4]))
        if hyb == 3:
            sp3_O_hybrid.append(float(x[4]))
    if x[1] == 'F':
        fluorine_norms.append(float(x[4]))



num_bins = 100
alpha = 0.6

plt.hist(carbon_norms, bins=num_bins, label='C (20% height)', color=color[2], alpha=alpha, weights=0.2*0.001*np.ones((len(carbon_norms))))
plt.hist(oxygen_norms, bins=num_bins, label='O', alpha=alpha, color=color[1], weights=0.001*np.ones((len(oxygen_norms))))
plt.hist(nitrogen_norms, bins=num_bins, label='N', alpha=alpha, color=color[3], weights=0.001*np.ones((len(nitrogen_norms))))
#plt.hist(fluorine_norms, bins=num_bins, label='F', alpha=alpha, color=color[4], weights=0.001*np.ones((len(fluorine_norms))))
#hybridisations:
plt.hist(sp3_C_hybrid, bins=num_bins, label=r'C $sp^3$ (20% height)', alpha=alpha, histtype='step', fill=False, linestyle='solid', ec=color[2], weights=0.2*0.001*np.ones((len(sp3_C_hybrid))))
plt.hist(sp2_C_hybrid, bins=num_bins, label=r'C $sp^2$ (20% height)', alpha=alpha, histtype='step', fill=False, linestyle='dashed', ec=color[2], weights=0.2*0.001*np.ones((len(sp2_C_hybrid))))
plt.hist(sp_C_hybrid, bins=num_bins, label=r'C $sp$ (20% height)', alpha=alpha, histtype='step', fill=False, linestyle='dotted', ec=color[2], weights=0.2*0.001*np.ones((len(sp_C_hybrid))))
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
