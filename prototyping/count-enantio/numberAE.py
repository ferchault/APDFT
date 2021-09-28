import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os
import sys
import math

def average_num_AE(input_file_prefix, input_file_postfix, last_log_num):
    if input_file_prefix == 'ZINC_log':
        N = np.array([i for i in range(4,60)]) #Number of atoms
    elif input_file_prefix == 'QM9_log':
        N = np.array([i for i in range(2,10)]) #Number of atoms
    num_moles = np.zeros((len(N))) #Number of times a molecule with N atoms occurs
    num_moles_zero = np.zeros((len(N))) #Number of times a molecule with N atoms occurs
    num_AE = np.zeros((len(N)))
    num_AE_SD = np.zeros((len(N)))
    #Go through all files, get number of AEs per N
    for log_num in range(1,last_log_num+1):
        f = open('logs/'+input_file_prefix+ '00'[:(2-len(str(log_num)))] + str(log_num) +input_file_postfix+'.txt', 'r')
        Lines = f.readlines()
        f.close()
        for line in Lines:
            x = line.split('\t')
            num_AE[int(x[2])-N[0]] += float(x[3])
            num_moles[int(x[2])-N[0]] += 1
            if float(x[3]) == 0:
                num_moles_zero[int(x[2])-N[0]] += 1
    #print("Percentage of non-yielding molecules vs number of heavy atoms")
    #print(100*num_moles_zero/num_moles)
    print("Number of molecules with N atoms")
    print(num_moles)
    #Normalize the function for percentages
    for i in range(len(N)):
        num_AE[i] /= num_moles[i]
    for log_num in range(1,last_log_num+1):
        f = open('logs/'+input_file_prefix+ '00'[:(2-len(str(log_num)))] + str(log_num) +input_file_postfix+'.txt', 'r')
        Lines = f.readlines()
        f.close()
        for line in Lines:
            x = line.split('\t')
            num_AE_SD[int(x[2])-N[0]] += (float(x[3])-num_AE[int(x[2])-N[0]])**2
    for thing in N:
        #print(num_AE_SD)
        num_AE_SD[thing-N[0]] = math.sqrt(num_AE_SD[thing-N[0]]/num_moles[thing-N[0]])
    #print('Percentage: '+str(num_AE*100))
    return N, num_AE, num_AE_SD

#------------------Plot the number of AEs vs N for the paper--------------------
"""

For QM9 and ZINC_named, plot the number of AEs for exact and approximate ones vs
the number of heavy atoms N

"""

N_QM9_exact, av_QM9_exact, av_QM9_exact_SD = average_num_AE('QM9_log', '_dZ1_geom',14)
N_QM9_approx, av_QM9_approx, av_QM9_approx_SD = average_num_AE('QM9_log', '_dZ1_geom_range2',14)

N_ZINC_exact, av_ZINC_exact, av_ZINC_exact_SD = average_num_AE('ZINC_log', '_dZ1_geom',6)
N_ZINC_approx, av_ZINC_approx, av_ZINC_approx_SD = average_num_AE('ZINC_log', '_dZ1_geom_range2',6)

print(av_QM9_exact_SD)

fig, ax = plt.subplots()

#Average energy differences
ax.plot(N_QM9_exact[1:], av_QM9_exact[1:], label=r'Exact (QM9)', color='tab:blue', marker='+')
ax.fill_between(N_QM9_exact[1:], av_QM9_exact[1:]-av_QM9_exact_SD[1:],av_QM9_exact[1:]+av_QM9_exact_SD[1:], color='tab:blue', alpha=0.5)
ax.plot(N_QM9_approx[1:], av_QM9_approx[1:], label=r'Approx. (QM9)', color='tab:blue', marker='+', linestyle='dashed')
ax.plot(N_ZINC_exact[:17], av_ZINC_exact[:17], label=r'Exact (ZINC)', color='tab:red', marker='+')
ax.fill_between(N_ZINC_exact[:17], av_ZINC_exact[:17]-av_ZINC_exact_SD[:17],av_ZINC_exact[:17]+av_ZINC_exact_SD[:17], color='tab:red', alpha=0.5)
ax.plot(N_ZINC_approx[:17], av_ZINC_approx[:17], label=r'Approx. (ZINC)', color='tab:red', marker='+', linestyle='dashed')
ax.set_xlabel(r'$N$', fontsize=14)
ax.set_ylabel(r'Av. $\#$ AEs per mol.', fontsize=14)
ax.set_xlim([2.5,20.5])
ax.set_xticks(range(3,21))
ax.set_ylim([0.05,500])
ax.set_yscale('log')
ax.legend(loc="upper left",framealpha=0, edgecolor='black')


fig.tight_layout()
fig.savefig("figures/numAE.png", dpi=400)
