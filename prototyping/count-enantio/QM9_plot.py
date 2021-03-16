import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
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
    for log_num in range(1,11):
        f = open('QM9_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'.txt', 'r')
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
    return N, times, times_standarddev


def get_CDF(bin_size = 50):
    N = np.array([2,3,4,5,6,7,8,9], dtype='int') #Number of atoms
    num_moles = np.zeros((8), dtype='int') #Number of times a molecule with N atoms occurs
    max = 0
    for log_num in range(1,11):
        f = open('QM9_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'.txt', 'r')
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
    #Get number of necessary bins:
    num_bin = int(max/bin_size)+2 #One bin just for the zeros
    binning = np.array(range(0,num_bin))*bin_size
    #Initalize array to save the binning; for each N there are num_bin bins
    num_AE = np.zeros((len(N),num_bin), dtype=object)
    #Go through all files again, get number of AEs per bin per N
    for log_num in range(1,9):
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

bin_size = 50
color = ['#407294', '#ffa500', '#065535', '#800000', '#660066', '#310c0c', '#f7347a', '#696966']

N, times, SD = get_times()
binning, num_AE = get_CDF(bin_size=bin_size)
num_AE *= 100
#print(binning, num_AE)
fig, ax = plt.subplots()
for i in N:
    ax.set_xlabel('Amount of AEs in bins of size '+str(bin_size))
    ax.set_ylabel('Percentage of AEs among all molecules')
    ax.set_ylim([0,105])
    ax.bar(binning[:20],num_AE[i-2][:20], width=bin_size, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    #plt.xscale('log')
    ax.legend(loc="upper right",framealpha=1, edgecolor='black')
    fig.savefig('CDF_AE_N'+str(i)+'.png', dpi=300)
    ax.clear()

plt.xlabel('Amount of AEs in bins of size '+str(bin_size))
plt.ylabel('Percentage of AEs among all molecules with'+ str(i) +'heavy atoms')
plt.ylim([0,105])
for i in N:
    plt.bar(binning[:20],num_AE[i-2][:20], width=bin_size, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    #plt.xscale('log')
    plt.legend(loc="upper right",framealpha=1, edgecolor='black')
    plt.savefig('CDF_AE.png', dpi=300)

#Plot the times
fig, ax = plt.subplots()
ax.scatter(N, times, marker='x', color='#1f77b4', label='geometry-based')
plt.errorbar(N, times, yerr=SD, fmt='none', capsize=4, color='#1f77b4')
ax.set_xticks(range(2,10))
ax.set_xlim([1.5, 9.7])
ax.set(xlabel='Number of heavy atoms N', ylabel='Average time / s')
#ax.grid(which='both')
plt.yscale('log')
#ax.legend(loc="lower right",framealpha=1, edgecolor='black')
fig.savefig("QM9_times.png", dpi=300)


#Plot the CDF; make a histogram here
