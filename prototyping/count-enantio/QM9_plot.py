import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os
import sys

#Initalize all variables:
plt.rcParams.update({'font.size': 14})
np.set_printoptions(threshold=sys.maxsize)
color = ['#407294', '#ffa500', '#065535', '#800000', '#660066', '#310c0c', '#f7347a', '#696966']

def get_times(input_file_prefix, input_file_postfix):
    N = np.array([2,3,4,5,6,7,8,9], dtype='int') #Number of atoms
    num_moles = np.zeros((len(N)), dtype='int') #Number of times a molecule with N atoms occurs
    times = np.zeros((len(N))) #Times of calculating for N atoms
    times_variance = np.zeros((len(N)))
    max_time = 0
    for log_num in range(1,last_log_num+1):
        f = open(input_file_prefix+ '00'[:(2-len(str(log_num)))] + str(log_num) +input_file_postfix+'.txt', 'r')
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
        f = open(input_file_prefix+ '00'[:(2-len(str(log_num)))] + str(log_num) +input_file_postfix+'.txt', 'r')
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
        f = open(input_file_prefix+ '00'[:(2-len(str(log_num)))] + str(log_num) +input_file_postfix+'.txt', 'r')
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
    #Normalize the function for percentages
    #print('num_moles '+input_file_prefix+input_file_postfix+str(num_moles))
    #print('num_AE '+input_file_prefix+input_file_postfix+str(num_AE))
    for i in range(len(N)):
        for j in range(num_bin):
            num_AE[i][j] /= num_moles[i]
    #print('Percentage: '+str(num_AE*100))
    return binning, num_AE, num_moles



last_log_num = 14
bin_size = 30

N, times_dZ2, SD_dZ2 = get_times('QM9_log', '_dZ2')
N, times_dZ1, SD_dZ1 = get_times('QM9_log', '_dZ1')



#--------------------------QM9 as references, dZ1-------------------------------
binning, num_AE, num_moles = get_CDF( 'QM9_log', '_dZ1', bin_size=bin_size)
num_AE *= 100
#print(binning, num_AE)
fig, ax = plt.subplots()
for i in N:
    ax.set_xlabel('Amount of AEs in bins of size '+str(bin_size) + ',  (dZ_max = 1)')
    ax.set_ylabel('AEs among all molecules (ref) / %')
    ax.set_xlim([-bin_size/2,1090])
    ax.set_ylim([0.0012,120])
    ax.bar(binning, num_AE[i-2], width=bin_size, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    plt.yscale('log')
    ax.legend(loc="upper right",framealpha=1, edgecolor='black')
    fig.savefig('figures/CDF_AE_N'+str(i)+'_dZ1.png', dpi=500)
    ax.clear()

#num_moles is needed to rescale the number of AEs depending on their N
all_moles = np.sum(num_moles[1:])
plt.xlabel('Amount of AEs in bins of size '+str(bin_size) + ',  (dZ_max = 1)')
plt.ylabel('AEs among all molecules (ref) / %')
plt.ylim([0.001,50])
plt.xlim([-bin_size/2,1090])
plt.yscale('log')
#plt.xscale('log')
bottom = np.zeros((len(binning[1:])))
for i in N[1:]: #exclude the 2, because it does not have any AEs
    #Calculate bottom:
    plt.bar(binning[1:],num_AE[i-2][1:]*num_moles[i-2]/all_moles, width=bin_size, bottom=bottom, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    #Update bottom:
    #print(bottom)
    #print(num_AE[i-2][1:]*num_moles[i-2]/all_moles)
    bottom = bottom + num_AE[i-2][1:]*num_moles[i-2]/all_moles
    plt.legend(loc="upper right",framealpha=1, edgecolor='black')
    #plt.legend(loc="upper right", bbox_to_anchor=(0.9,1),framealpha=1, edgecolor='black')
    plt.savefig('figures/CDF_AE_dZ1.png', dpi=500)


#--------------------------QM9 as references, dZ2-------------------------------
binning, num_AE, num_moles = get_CDF( 'QM9_log', '_dZ2', bin_size=bin_size)
num_AE *= 100
#print(binning, num_AE)
fig, ax = plt.subplots()
for i in N:
    ax.set_xlabel('Amount of AEs in bins of size '+str(bin_size) + ',  (dZ_max = 2)')
    ax.set_ylabel('AEs among all molecules (ref) / %')
    ax.set_xlim([-bin_size/2,1090])
    ax.set_ylim([0.0012,120])
    ax.bar(binning, num_AE[i-2], width=bin_size, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    plt.yscale('log')
    ax.legend(loc="upper right",framealpha=1, edgecolor='black')
    fig.savefig('figures/CDF_AE_N'+str(i)+'_dZ2.png', dpi=500)
    ax.clear()

#num_moles is needed to rescale the number of AEs depending on their N
all_moles = np.sum(num_moles[1:])
plt.xlabel('Amount of AEs in bins of size '+str(bin_size) + ',  (dZ_max = 2)')
plt.ylabel('AEs among all molecules (ref) / %')
plt.ylim([0.001,50])
plt.xlim([-bin_size/2,1090])
plt.yscale('log')
#plt.xscale('log')
bottom = np.zeros((len(binning[1:])))
for i in N[1:]: #exclude the 2, because it does not have any AEs
    #Calculate bottom:
    plt.bar(binning[1:],num_AE[i-2][1:]*num_moles[i-2]/all_moles, width=bin_size, bottom=bottom, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    #Update bottom:
    #print(bottom)
    #print(num_AE[i-2][1:]*num_moles[i-2]/all_moles)
    bottom = bottom + num_AE[i-2][1:]*num_moles[i-2]/all_moles
    plt.legend(loc="upper right",framealpha=1, edgecolor='black')
    #plt.legend(loc="upper right", bbox_to_anchor=(0.9,1),framealpha=1, edgecolor='black')
    plt.savefig('figures/CDF_AE_dZ2.png', dpi=500)

#---------------------------------Plot the times-------------------------------

fig, ax = plt.subplots()
ax.scatter(N, times_dZ2, marker='x', color=color[1], label='dZ_max = 2')
plt.errorbar(N, times_dZ2, yerr=SD_dZ2, fmt='none', capsize=4, color=color[1])
ax.scatter(N, times_dZ1, marker='x', color=color[0], label='dZ_max = 1')
plt.errorbar(N, times_dZ1, yerr=SD_dZ1, fmt='none', capsize=4, color=color[0])
ax.set_xticks(range(2,10))
ax.set_xlim([1.5, 9.7])
ax.set_ylim([0.001,5])
ax.set(xlabel='Number of heavy atoms N', ylabel='Average time / s')
#ax.grid(which='both')
plt.yscale('log')
ax.legend(loc="upper left",framealpha=1, edgecolor='black')
fig.savefig("figures/QM9_times_dZ1and2.png", dpi=500)


#--------------------------QM9 as targets--------------------------------------
'''last_log_num = 14
bin_size = 30

N, times, SD = get_times('QM9_target_log', '_dZ1')
binning, num_AE, num_moles = get_CDF('QM9_target_log', '_dZ1', bin_size=bin_size)
num_AE *= 100
#print(binning, num_AE)
fig, ax = plt.subplots()
for i in N:
    ax.set_xlabel('Amount of AEs in bins of size '+str(bin_size) + ',  (dZ_max = 1)')
    ax.set_ylabel('AEs among all molecules (tar) / %')
    ax.set_xlim([-bin_size/2,1090])
    ax.set_ylim([0.0012,120])
    ax.bar(binning, num_AE[i-2], width=bin_size, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    plt.yscale('log')
    ax.legend(loc="upper right",framealpha=1, edgecolor='black')
    fig.savefig('figures/CDF_AE_N'+str(i)+'_target_dZ1.png', dpi=500)
    ax.clear()

#num_moles is needed to rescale the number of AEs depending on their N
all_moles = np.sum(num_moles)
plt.xlabel('Amount of AEs in bins of size '+str(bin_size) + ',  (dZ_max = 1)')
plt.ylabel('AEs among all molecules (tar) / %')
plt.ylim([0.001,1.3])
plt.xlim([-bin_size/2,1090])
plt.yscale('log')
#plt.xscale('log')
bottom = np.zeros((len(binning[1:])))
for i in N[1:]: #exclude the 2, because it does not have any AEs
    #Calculate bottom:
    plt.bar(binning[1:],num_AE[i-2][1:]*num_moles[i-2]/all_moles, width=bin_size, bottom=bottom, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    #Update bottom:
    #print(bottom)
    #print(num_AE[i-2][1:]*num_moles[i-2]/all_moles)
    bottom = bottom + num_AE[i-2][1:]*num_moles[i-2]/all_moles
    plt.legend(loc="upper right",framealpha=1, edgecolor='black')
    #plt.legend(loc="upper right", bbox_to_anchor=(0.9,1),framealpha=1, edgecolor='black')
    plt.savefig('figures/CDF_AE_target_dZ1.png', dpi=500)

#Plot the times
fig, ax = plt.subplots()
ax.scatter(N, times, marker='x', color='#1f77b4', label='geometry-based')
plt.errorbar(N, times, yerr=SD, fmt='none', capsize=4, color='#1f77b4')
ax.set_xticks(range(2,10))
ax.set_xlim([1.5, 9.7])
ax.set_ylim([0.001,5])
ax.set(xlabel='Number of heavy atoms N', ylabel='Average time / s')
#ax.grid(which='both')
plt.yscale('log')
#ax.legend(loc="lower right",framealpha=1, edgecolor='black')
fig.savefig("figures/QM9_times_target_dZ1.png", dpi=500)


N, times, SD = get_times('QM9_target_log', '_dZ2')
binning, num_AE, num_moles = get_CDF('QM9_target_log', '_dZ2', bin_size=bin_size)
num_AE *= 100
#print(binning, num_AE)
fig, ax = plt.subplots()
for i in N:
    ax.set_xlabel('Amount of AEs in bins of size '+str(bin_size) + ',  (dZ_max = 2)')
    ax.set_ylabel('AEs among all molecules (tar) / %')
    ax.set_xlim([-bin_size/2,1090])
    ax.set_ylim([0.0012,120])
    ax.bar(binning, num_AE[i-2], width=bin_size, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    plt.yscale('log')
    ax.legend(loc="upper right",framealpha=1, edgecolor='black')
    fig.savefig('figures/CDF_AE_N'+str(i)+'_target_dZ2.png', dpi=500)
    ax.clear()

#num_moles is needed to rescale the number of AEs depending on their N
all_moles = np.sum(num_moles)
plt.xlabel('Amount of AEs in bins of size '+str(bin_size) + ',  (dZ_max = 2)')
plt.ylabel('AEs among all molecules (tar) / %')
plt.ylim([0.001,1.3])
plt.xlim([-bin_size/2,1090])
plt.yscale('log')
#plt.xscale('log')
bottom = np.zeros((len(binning[1:])))
for i in N[1:]: #exclude the 2, because it does not have any AEs
    #Calculate bottom:
    plt.bar(binning[1:],num_AE[i-2][1:]*num_moles[i-2]/all_moles, width=bin_size, bottom=bottom, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    #Update bottom:
    #print(bottom)
    #print(num_AE[i-2][1:]*num_moles[i-2]/all_moles)
    bottom = bottom + num_AE[i-2][1:]*num_moles[i-2]/all_moles
    plt.legend(loc="upper right",framealpha=1, edgecolor='black')
    #plt.legend(loc="upper right", bbox_to_anchor=(0.9,1),framealpha=1, edgecolor='black')
    plt.savefig('figures/CDF_AE_target_dZ2.png', dpi=500)

#Plot the times
fig, ax = plt.subplots()
ax.scatter(N, times, marker='x', color='#1f77b4', label='geometry-based')
plt.errorbar(N, times, yerr=SD, fmt='none', capsize=4, color='#1f77b4')
ax.set_xticks(range(2,10))
ax.set_xlim([1.5, 9.7])
ax.set_ylim([0.001,5])
ax.set(xlabel='Number of heavy atoms N', ylabel='Average time / s')
#ax.grid(which='both')
plt.yscale('log')
#ax.legend(loc="lower right",framealpha=1, edgecolor='black')
fig.savefig("figures/QM9_times_target_dZ2.png", dpi=500)'''

#--------------------------QM9 if uncolored dZ1---------------------------------
N, times_dZ1, SD_dZ1 = get_times('QM9_uncolored_log', '_dZ1')
N, times_dZ2, SD_dZ2 = get_times('QM9_uncolored_log', '_dZ2')
binning, num_AE, num_moles = get_CDF('QM9_uncolored_log', '_dZ1', bin_size=bin_size)
num_AE *= 100
#print(binning, num_AE)
fig, ax = plt.subplots()
for i in N:
    ax.set_xlabel('Amount of AEs in bins of size '+str(bin_size)+',  (dZ_max = 1)')
    ax.set_ylabel('AEs among all molecules (uncolored) / %')
    ax.set_xlim([-bin_size/2,1090])
    ax.set_ylim([0.0012,120])
    ax.bar(binning, num_AE[i-2], width=bin_size, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    plt.yscale('log')
    ax.legend(loc="upper right",framealpha=1, edgecolor='black')
    fig.savefig('figures/CDF_AE_N'+str(i)+'_uncolored_dZ1.png', dpi=500)
    ax.clear()

#num_moles is needed to rescale the number of AEs depending on their N
all_moles = np.sum(num_moles[1:])
plt.xlabel('Amount of AEs in bins of size '+str(bin_size)+',  (dZ_max = 1)')
plt.ylabel('AEs among all molecules (uncolored) / %')
plt.ylim([0.001,50])
plt.xlim([-bin_size/2,1090])
plt.yscale('log')
#plt.xscale('log')
bottom = np.zeros((len(binning[1:])))
for i in N[1:]: #exclude the 2, because it does not have any AEs
    #Calculate bottom:
    plt.bar(binning[1:],num_AE[i-2][1:]*num_moles[i-2]/all_moles, width=bin_size, bottom=bottom, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    #Update bottom:
    #print(bottom)
    #print(num_AE[i-2][1:]*num_moles[i-2]/all_moles)
    bottom = bottom + num_AE[i-2][1:]*num_moles[i-2]/all_moles
    plt.legend(loc="upper right",framealpha=1, edgecolor='black')
    #plt.legend(loc="upper right", bbox_to_anchor=(0.9,1),framealpha=1, edgecolor='black')
    plt.savefig('figures/CDF_AE_uncolored_dZ1.png', dpi=500)


#--------------------------QM9 if uncolored dZ2---------------------------------
binning, num_AE, num_moles = get_CDF('QM9_uncolored_log', '_dZ2', bin_size=bin_size)
num_AE *= 100
fig, ax = plt.subplots()
for i in N:
    ax.set_xlabel('Amount of AEs in bins of size '+str(bin_size)+',  (dZ_max = 2)')
    ax.set_ylabel('AEs among all molecules (uncolored) / %')
    ax.set_xlim([-bin_size/2,1090])
    ax.set_ylim([0.0012,120])
    ax.bar(binning, num_AE[i-2], width=bin_size, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    plt.yscale('log')
    ax.legend(loc="upper right",framealpha=1, edgecolor='black')
    fig.savefig('figures/CDF_AE_N'+str(i)+'_uncolored_dZ2.png', dpi=500)
    ax.clear()

all_moles = np.sum(num_moles[1:])
plt.xlabel('Amount of AEs in bins of size '+str(bin_size)+',  (dZ_max = 2)')
plt.ylabel('AEs among all molecules (uncolored) / %')
plt.ylim([0.001,50])
plt.xlim([-bin_size/2,1090])
plt.yscale('log')
#plt.xscale('log')
bottom = np.zeros((len(binning[1:])))
for i in N[1:]: #exclude the 2, because it does not have any AEs
    #Calculate bottom:
    plt.bar(binning[1:],num_AE[i-2][1:]*num_moles[i-2]/all_moles, width=bin_size, bottom=bottom, label='N = '+str(i), alpha=0.5, edgecolor=color[i-2], facecolor=color[i-2], joinstyle='miter')
    #Update bottom:
    #print(bottom)
    #print(num_AE[i-2][1:]*num_moles[i-2]/all_moles)
    bottom = bottom + num_AE[i-2][1:]*num_moles[i-2]/all_moles
    plt.legend(loc="upper right",framealpha=1, edgecolor='black')
    #plt.legend(loc="upper right", bbox_to_anchor=(0.9,1),framealpha=1, edgecolor='black')
    plt.savefig('figures/CDF_AE_uncolored_dZ2.png', dpi=500)

#----------------------------Plot the times-------------------------------------
fig, ax = plt.subplots()
ax.scatter(N, times_dZ2, marker='x', color=color[1], label='dZ_max = 2')
plt.errorbar(N, times_dZ2, yerr=SD_dZ2, fmt='none', capsize=4, color=color[1])
ax.scatter(N, times_dZ1, marker='x', color=color[0], label='dZ_max = 1')
plt.errorbar(N, times_dZ1, yerr=SD_dZ1, fmt='none', capsize=4, color=color[0])
ax.set_xticks(range(2,10))
ax.set_xlim([1.5, 9.7])
ax.set_ylim([0.001,5])
ax.set(xlabel='Number of heavy atoms N', ylabel='Average time / s')
#ax.grid(which='both')
plt.yscale('log')
ax.legend(loc="lower right",framealpha=1, edgecolor='black')
fig.savefig("figures/QM9_times_uncolored_dZ1and2.png", dpi=500)


#--------------------------Orbits in QM9 (dZ1 and dZ2)--------------------------
#Initalize x and y array, i.e. AEs and related orbit quantity
AEs = np.empty((133885+1-4), dtype='int')
orbit_quantity = np.empty((133885+1-4), dtype='int')
indices = np.array([], dtype='int')

#Get all the data:
for log_num in range(1,last_log_num+1):
    f = open('QM9_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'_dZ1.txt', 'r')
    Lines = f.readlines()
    for line in Lines:
        x = line.split('\t')
        AEs[int(x[0].split('_')[1].lstrip('0'))-4] = int(x[3]) #1: time; 2: number of heavy atoms; 3: number of AEs
    f.close()

for log_num in range(1,last_log_num+1):
    f = open('QM9_orbit_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'_dZ1.txt', 'r')
    Lines = f.readlines()
    for line in Lines:
        x = line.split('\t')
        orbit_quantity[int(x[0].lstrip('0'))-4] = int(x[2]) #1: number of orbits; 2: number of atoms in orbits; 3: maximum sized orbit

        #print out all surprisingly cold contestants and save their index:
        if int(x[3]) == 2 and AEs[int(x[0].lstrip('0'))-4] == 0:
        #    print('tag: '+x[0]+'\tNumber of atoms in orbits: '+str(x[2])+'\tNumber of AEs: '+str(AEs[a])+'\tMax orbit size: '+str(x[3]))
            indices = np.append(indices, int(x[0].lstrip('0'))-4)
    f.close()


fig, ax = plt.subplots(tight_layout=True)
plt.hist2d(orbit_quantity, AEs,bins=([0,1,2,3,4,5,6,7,8,9],[0,100,200,300,400,500,600,700,800,900,1000,1100]), norm = colors.LogNorm())
ax.set(xlabel='Number of vertices (heavy atoms) in orbits', ylabel='Number of AEs,  (dZ_max = 1)')
#ax.set_title('Number of AEs vs size of largest orbit')
plt.colorbar()
plt.clim(1,100000)
fig.savefig("figures/hist2d_maxorbit_vs_numAE_dZ1.png",dpi=500)
ax.clear()

#Delete some elements from the the data
AEs = np.delete(AEs,indices)
orbit_quantity = np.delete(orbit_quantity, indices)

fig, ax = plt.subplots(tight_layout=True)
plt.hist2d(orbit_quantity, AEs, bins=([0,1,2,3,4,5,6,7,8,9],[0,100,200,300,400,500,600,700,800,900,1000,1100]), norm = colors.LogNorm())
ax.set(xlabel='Number of vertices (heavy atoms) in orbits', ylabel='Number of AEs,  (dZ_max = 1)')
plt.colorbar()
plt.clim(1,100000)
fig.savefig("figures/hist2d_maxorbit_vs_numAE_filtered_dZ1.png",dpi=500)
ax.clear()





#Initalize x and y array, i.e. AEs and related orbit quantity
AEs = np.empty((133885+1-4), dtype='int')
orbit_quantity = np.empty((133885+1-4), dtype='int')
indices = np.array([], dtype='int')

#Get all the data:
for log_num in range(1,last_log_num+1):
    f = open('QM9_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'_dZ2.txt', 'r')
    Lines = f.readlines()
    for line in Lines:
        x = line.split('\t')
        AEs[int(x[0].split('_')[1].lstrip('0'))-4] = int(x[3]) #1: time; 2: number of heavy atoms; 3: number of AEs
    f.close()

for log_num in range(1,last_log_num+1):
    f = open('QM9_orbit_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'_dZ2.txt', 'r')
    Lines = f.readlines()
    for line in Lines:
        x = line.split('\t')
        orbit_quantity[int(x[0].lstrip('0'))-4] = int(x[2]) #1: number of orbits; 2: number of atoms in orbits; 3: maximum sized orbit

        #print out all surprisingly cold contestants and save their index:
        if int(x[3]) == 2 and AEs[int(x[0].lstrip('0'))-4] == 0:
        #    print('tag: '+x[0]+'\tNumber of atoms in orbits: '+str(x[2])+'\tNumber of AEs: '+str(AEs[a])+'\tMax orbit size: '+str(x[3]))
            indices = np.append(indices, int(x[0].lstrip('0'))-4)
    f.close()


fig, ax = plt.subplots(tight_layout=True)
plt.hist2d(orbit_quantity, AEs,bins=([0,1,2,3,4,5,6,7,8,9],[0,100,200,300,400,500,600,700,800,900,1000,1100]), norm = colors.LogNorm())
ax.set(xlabel='Number of vertices (heavy atoms) in orbits', ylabel='Number of AEs,  (dZ_max = 2)')
#ax.set_title('Number of AEs vs size of largest orbit')
plt.colorbar()
plt.clim(1,100000)
fig.savefig("figures/hist2d_maxorbit_vs_numAE_dZ2.png",dpi=500)
ax.clear()

#Delete some elements from the the data
AEs = np.delete(AEs,indices)
orbit_quantity = np.delete(orbit_quantity, indices)

fig, ax = plt.subplots(tight_layout=True)
plt.hist2d(orbit_quantity, AEs, bins=([0,1,2,3,4,5,6,7,8,9],[0,100,200,300,400,500,600,700,800,900,1000,1100]), norm = colors.LogNorm())
ax.set(xlabel='Number of vertices (heavy atoms) in orbits', ylabel='Number of AEs,  (dZ_max = 2)')
plt.colorbar()
plt.clim(1,100000)
fig.savefig("figures/hist2d_maxorbit_vs_numAE_filtered_dZ2.png",dpi=500)
ax.clear()


#-----------------------Sweetspot---------------------------------------------
'''Find the behavior of percentage of AEs in dataset versus number of heavy atoms,
because: N = 2 is too small for AEs to appear and the trend goes up, however, graphs
with large number of verties N get increasingly asymmetric and hence their group action
orbit number decreases: Hypthesis: Find the sweetspot, where percentage reaches maximum'''

#Get the data:
#Possible number of heavy atoms

QM9_percentages_dZ1 = np.zeros((8))
QM9_percentages_dZ2 = np.zeros((8))

QM9_percentages_uncolored_dZ1 = np.zeros((8))
QM9_percentages_uncolored_dZ2 = np.zeros((8))

binning, num_AE, num_moles = get_CDF('QM9_log', '_dZ1')
for i in range(len(num_moles)):
    QM9_percentages_dZ1[i] = (1-num_AE[i][0])
QM9_percentages_dZ1 *= 100

binning, num_AE, num_moles = get_CDF('QM9_log', '_dZ2')
for i in range(len(num_moles)):
    QM9_percentages_dZ2[i] = (1-num_AE[i][0])
QM9_percentages_dZ2 *= 100

binning, num_AE, num_moles = get_CDF('QM9_uncolored_log', '_dZ1')
for i in range(len(num_moles)):
    QM9_percentages_uncolored_dZ1[i] = (1-num_AE[i][0])
QM9_percentages_uncolored_dZ1 *= 100

binning, num_AE, num_moles = get_CDF('QM9_uncolored_log', '_dZ2')
for i in range(len(num_moles)):
    QM9_percentages_uncolored_dZ2[i] = (1-num_AE[i][0])
QM9_percentages_uncolored_dZ2 *= 100

theoAE_dZ1 = np.array([0,0,17.91,34.48,37.99,28.07,22.42,17.01])
theoAE_dZ2 = np.array([0,4.54,17.05,22.41,20.56,17.15,14.08])

'''Original data:
Number of atoms: 2	dZ_max: 1	Possibles / % : 0.0
Number of atoms: 2	dZ_max: 2	Possibles / % : 0.0
Number of atoms: 3	dZ_max: 1	Possibles / % : 0.0
Number of atoms: 5	dZ_max: 1	Possibles / % : 34.48275862068966
Number of atoms: 3	dZ_max: 2	Possibles / % : 4.545454545454546
Number of atoms: 4	dZ_max: 1	Possibles / % : 17.91044776119403
Number of atoms: 6	dZ_max: 1	Possibles / % : 37.99843627834245
Number of atoms: 5	dZ_max: 2	Possibles / % : 22.418220856312885
Number of atoms: 7	dZ_max: 1	Possibles / % : 28.072608852973104
Number of atoms: 4	dZ_max: 2	Possibles / % : 17.05426356589147
Number of atoms: 6	dZ_max: 2	Possibles / % : 20.562076100167936
Number of atoms: 8	dZ_max: 1	Possibles / % : 22.422075258986997
Number of atoms: 7	dZ_max: 2	Possibles / % : 17.156894120961944
Number of atoms: 9	dZ_max: 1	Possibles / % : 17.01040415913692
Number of atoms: 8	dZ_max: 2	Possibles / % : 14.080211188147327
'''

fig, ax = plt.subplots()
ax.scatter([2,3,4,5,6,7,8,9], QM9_percentages_dZ1, marker='x', color=color[0], label='dZ_max = 1 (QM9)')
ax.plot([2,3,4,5,6,7,8,9], QM9_percentages_dZ1,color=color[0])
ax.scatter([2,3,4,5,6,7,8,9], QM9_percentages_dZ2, marker='x', color=color[1], label='dZ_max = 2 (QM9)')
ax.plot([2,3,4,5,6,7,8,9], QM9_percentages_dZ2,color=color[1])

ax.scatter([2,3,4,5,6,7,8,9], QM9_percentages_uncolored_dZ1, marker='x', color=color[2], label='dZ_max = 1 (QM9, u.c.)')
ax.plot([2,3,4,5,6,7,8,9], QM9_percentages_uncolored_dZ1,color=color[2])
ax.scatter([2,3,4,5,6,7,8,9], QM9_percentages_uncolored_dZ2, marker='x', color=color[3], label='dZ_max = 2 (QM9, u.c.)')
ax.plot([2,3,4,5,6,7,8,9], QM9_percentages_uncolored_dZ2,color=color[3])

ax.scatter([2,3,4,5,6,7,8,9], theoAE_dZ1, marker='x', color=color[4], label='dZ_max = 1 (theo, QRS)')
ax.plot([2,3,4,5,6,7,8,9], theoAE_dZ1, color=color[4])
ax.scatter([2,3,4,5,6,7,8], theoAE_dZ2, marker='x', color=color[5], label='dZ_max = 2 (theo, PQRST)')
ax.plot([2,3,4,5,6,7,8], theoAE_dZ2, color=color[5])
ax.set_xticks(range(2,10))
ax.set_yticks(range(0,105,10))
ax.set_xlim([1.7, 9.3])
ax.set_ylim([0,102])
ax.set(xlabel='Number of heavy atoms N', ylabel='Amount of molecules with AEs / %')
#ax.grid(which='both')
#plt.yscale('log')
ax.legend(loc="upper right",prop={'size': 8},framealpha=1, edgecolor='black')
fig.savefig("figures/sweetspot.png", dpi=500)
