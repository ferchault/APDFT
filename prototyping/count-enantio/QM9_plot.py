import matplotlib.pyplot as plt
import numpy as np
import os
import sys

np.set_printoptions(threshold=sys.maxsize)

#Initalize all variables:
def get_times():
    N = np.array([2,3,4,5,6,7,8,9], dtype='int') #Number of atoms
    num_moles = np.zeros((len(N)), dtype='int') #Number of times a molecule with N atoms occurs
    times = np.zeros((len(N))) #Times of calculating for N atoms
    times_variance = np.zeros((len(N)))
    max_time = 0
    for log_num in range(1,7):
        f = open('QM9_log0'+ str(log_num) +'.txt', 'r')
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
    print(max_time, max_time_index)
    return N, times, times_standarddev


def get_CDF(bin_size = 50):
    N = np.array([2,3,4,5,6,7,8,9], dtype='int') #Number of atoms
    num_moles = np.zeros((8), dtype='int') #Number of times a molecule with N atoms occurs
    max = 0
    for log_num in range(1,7):
        f = open('QM9_log0'+ str(log_num) +'.txt', 'r')
        Lines = f.readlines()
        for line in Lines:
            x = line.split('\t')
            num_moles[int(x[2])-2] += 1
            #Find maximum in all files
            if int(x[3]) > max:
                max = int(x[3])
        f.close()
    #print(max)
    #Get number of necessary bins:
    num_bin = int(max/bin_size)+2 #One bin just for the zeros
    binning = np.array(range(0,num_bin))*bin_size
    #Initalize array to save the binning; for each N there are num_bin bins
    num_AE = np.zeros((len(N),num_bin))
    #Go through all files again, get number of AEs per bin per N
    for log_num in range(1,7):
        f = open('QM9_log0'+ str(log_num) +'.txt', 'r')
        Lines = f.readlines()
        for line in Lines:
            x = line.split('\t')
            if int(x[3]) == 0: #0 is a special bin
                num_AE[int(x[2])-2][0] += 1
            else:
                num_AE[int(x[2])-2][int(float(x[3])/bin_size)+1] += 1
        f.close()
    #Normalize the function for percentages
    for i in range(len(N)):
        for j in range(num_bin):
            num_AE[i][j] /= num_moles[i]
    return binning, num_AE

N, times, SD = get_times()
binning, num_AE = get_CDF()

fig, ax = plt.subplots()
ax.scatter(N, times, marker='x', color='#1f77b4', label='geometry-based')
plt.errorbar(N, times, yerr=SD, fmt='none', capsize=4, color='#1f77b4')
ax.set_xticks(range(2,10))
ax.set_xlim([1.5, 9.7])
ax.set(xlabel='Number of heavy atoms N', ylabel='Average time / s')
#ax.grid(which='both')
#plt.yscale('log')
#ax.legend(loc="lower right",framealpha=1, edgecolor='black')
fig.savefig("QM9_times.png", dpi=300)