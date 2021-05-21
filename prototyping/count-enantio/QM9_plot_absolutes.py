import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os
import sys

#Initalize all variables:
plt.rcParams.update({'font.size': 14})
np.set_printoptions(threshold=sys.maxsize)
color = ['#407294', '#ffa500', '#065535', '#800000', '#660066', '#310c0c', '#f7347a', '#696966']
last_log_num = 14
bin_size = 30

def get_times(input_file_prefix, input_file_postfix):
    N = np.array([2,3,4,5,6,7,8,9], dtype='int') #Number of atoms
    num_moles = np.zeros((len(N)), dtype='int') #Number of times a molecule with N atoms occurs
    times = np.zeros((len(N))) #Times of calculating for N atoms
    times_variance = np.zeros((len(N)))
    max_time = 0
    for log_num in range(1,last_log_num+1):
        f = open('logs/'+input_file_prefix+ '00'[:(2-len(str(log_num)))] + str(log_num) +input_file_postfix+'_geom.txt', 'r')
        Lines = f.readlines()
        for line in Lines:
            x = line.split('\t')
            num_moles[int(x[2])-2] += 1
            times[(int(x[2])-2)] += float(x[1])
            if float(x[1]) > max_time:
                max_time = float(x[1])
                max_time_index = x[0]
            times_variance[(int(x[2])-2)] += float(x[1])*float(x[1])
        f.close()
    #print(num_moles)
    times /= num_moles
    times_variance /= num_moles
    times_variance -= times*times
    times_standarddev = np.sqrt(times_variance)
    #print(max_time, max_time_index)
    #print('num_moles '+input_file_prefix+input_file_postfix+str(num_moles))
    return N, times, times_standarddev


def get_CDF(input_file_prefix, input_file_postfix, bin_size = 50):
    N = np.array([2,3,4,5,6,7,8,9], dtype='int') #Number of atoms
    num_moles = np.zeros((8), dtype='int') #Number of times a molecule with N atoms occurs
    max = 0
    for log_num in range(1,last_log_num+1):
        f = open('logs/'+input_file_prefix+ '00'[:(2-len(str(log_num)))] + str(log_num) +input_file_postfix+'_geom.txt', 'r')
        Lines = f.readlines()
        for line in Lines:
            x = line.split('\t')
            num_moles[int(x[2])-2] += 1
            #Find maximum in all files
            if int(x[3]) > max:
                max = int(x[3])
                #print(x[0], str(max))
        f.close()
    #print(max)
    print('Outstanding candidates ('+input_file_prefix+input_file_postfix+'):')
    #Get number of necessary bins:
    num_bin = int(max/bin_size)+2 #One bin just for the zeros
    binning = np.array(range(0,num_bin))*bin_size
    #Initalize array to save the binning; for each N there are num_bin bins
    num_AE = np.zeros((len(N),num_bin), dtype=object)
    #Go through all files again, get number of AEs per bin per N
    for log_num in range(1,last_log_num+1):
        f = open('logs/'+input_file_prefix+ '00'[:(2-len(str(log_num)))] + str(log_num) +input_file_postfix+'_geom.txt', 'r')
        Lines = f.readlines()
        for line in Lines:
            x = line.split('\t')
            #Fill the bins:
            if int(x[3]) == 0: #0 is a special bin
                num_AE[int(x[2])-2][0] += 1
            else:
                num_AE[int(x[2])-2][int(float(x[3])/bin_size)+1] += 1
                if int(x[3]) > 300:
                    print(x)
        f.close()
    return binning, num_AE, num_moles

#-----------------------Sweetspot---------------------------------------------
'''Find the behavior of number of AEs in dataset versus number of heavy atoms,
because: N = 2 is too small for AEs to appear and the trend goes up, however, graphs
with large number of verties N get increasingly asymmetric and hence their group action
orbit number decreases: Hypthesis: Find the sweetspot, where percentage reaches maximum'''

#Get the data:
#Possible number of heavy atoms

QM9_numbers_dZ1 = np.zeros((8))
QM9_numbers_dZ2 = np.zeros((8))

QM9_numbers_uncolored_dZ1 = np.zeros((8))
QM9_numbers_uncolored_dZ2 = np.zeros((8))

binning, num_AE, num_moles = get_CDF('QM9_log', '_dZ1')
for i in range(len(num_moles)):
    QM9_numbers_dZ1[i] = num_AE[i][0]

binning, num_AE, num_moles = get_CDF('QM9_log', '_dZ2')
for i in range(len(num_moles)):
    QM9_numbers_dZ2[i] = num_AE[i][0]

binning, num_AE, num_moles = get_CDF('QM9_uncolored_log', '_dZ1')
for i in range(len(num_moles)):
    QM9_numbers_uncolored_dZ1[i] = num_AE[i][0]

binning, num_AE, num_moles = get_CDF('QM9_uncolored_log', '_dZ2')
for i in range(len(num_moles)):
    QM9_numbers_uncolored_dZ2[i] = num_AE[i][0]

fig, ax = plt.subplots()
ax.scatter([2,3,4,5,6,7,8,9], QM9_numbers_dZ1, marker='x', color=color[0], label='dZ_max = 1 (QM9)')
ax.plot([2,3,4,5,6,7,8,9], QM9_numbers_dZ1,color=color[0])
ax.scatter([2,3,4,5,6,7,8,9], QM9_numbers_dZ2, marker='x', color=color[1], label='dZ_max = 2 (QM9)')
ax.plot([2,3,4,5,6,7,8,9], QM9_numbers_dZ2,color=color[1])

ax.scatter([2,3,4,5,6,7,8,9], QM9_numbers_uncolored_dZ1, marker='x', color=color[2], label='dZ_max = 1 (QM9, i.a.)')
ax.plot([2,3,4,5,6,7,8,9], QM9_numbers_uncolored_dZ1,color=color[2])
ax.scatter([2,3,4,5,6,7,8,9], QM9_numbers_uncolored_dZ2, marker='x', color=color[3], label='dZ_max = 2 (QM9, i.a.)')
ax.plot([2,3,4,5,6,7,8,9], QM9_numbers_uncolored_dZ2,color=color[3])

ax.set_xticks(range(2,10))
#ax.set_yticks(range(0,35,10))
ax.set_xlim([1.7, 9.3])
#ax.set_ylim([0,35])
ax.set(xlabel='Number of heavy atoms N', ylabel='Amount of molecules with AEs')
#ax.grid(which='both')
plt.yscale('log')
ax.legend(loc="upper left",prop={'size': 12},framealpha=1, edgecolor='black')
fig.savefig("figures/QM9_AE_distro.png", dpi=500)
