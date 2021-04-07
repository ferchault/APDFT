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
    fig.savefig('figures/CDF_AE_N'+str(i)+'_dZ1_geom.png', dpi=500)
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
    plt.savefig('figures/CDF_AE_dZ1_geom.png', dpi=500)


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
    fig.savefig('figures/CDF_AE_N'+str(i)+'_dZ2_geom.png', dpi=500)
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
    plt.savefig('figures/CDF_AE_dZ2_geom.png', dpi=500)

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
fig.savefig("figures/QM9_times_dZ1and2_geom.png", dpi=500)


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
    fig.savefig('figures/CDF_AE_N'+str(i)+'_uncolored_dZ1_geom.png', dpi=500)
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
    plt.savefig('figures/CDF_AE_uncolored_dZ1_geom.png', dpi=500)


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
    fig.savefig('figures/CDF_AE_N'+str(i)+'_uncolored_dZ2_geom.png', dpi=500)
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
    plt.savefig('figures/CDF_AE_uncolored_dZ2_geom.png', dpi=500)

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
ax.legend(loc="upper left",framealpha=1, edgecolor='black')
fig.savefig("figures/QM9_times_uncolored_dZ1and2_geom.png", dpi=500)

#--------------------------Norms in QM9 (dZ1 and dZ2)--------------------------
#Initalize x and y array, i.e. AEs and related orbit quantity
AEs = np.empty((133885+1-4), dtype='int')
norm = np.empty((133885+1-4), dtype='float')
indices = np.array([], dtype='int')

#Get all the data:
for log_num in range(1,last_log_num+1):
    f = open('logs/'+'QM9_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'_dZ1_geom.txt', 'r')
    Lines = f.readlines()
    for line in Lines:
        x = line.split('\t')
        AEs[int(x[0].split('_')[1].lstrip('0'))-4] = int(x[3]) #1: time; 2: number of heavy atoms; 3: number of AEs
    f.close()

for log_num in range(1,last_log_num+1):
    f = open('logs/'+'QM9_norm_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'_dZ1_geom.txt', 'r')
    Lines = f.readlines()
    for line in Lines:
        x = line.split('\t')
        norm[int(x[0].lstrip('0'))-4] = float(x[1])*0.001 #1: norm

        '''#print out all surprisingly cold contestants and save their index:
        if int(x[1]) == 2 and AEs[int(x[0].lstrip('0'))-4] == 0:
        #    print('tag: '+x[0]+'\tNumber of atoms in orbits: '+str(x[2])+'\tNumber of AEs: '+str(AEs[a])+'\tMax orbit size: '+str(x[3]))
            indices = np.append(indices, int(x[0].lstrip('0'))-4)'''
    f.close()

bins = ([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180],[0,50,100,150,200,250,300,350,400,450,500,550,600])
fig, ax = plt.subplots(tight_layout=True)
plt.hist2d(norm, AEs, bins=bins, norm = colors.LogNorm())
ax.set(xlabel=r'Norm of Coulomb matrix of the molecule / $10^3$', ylabel='Number of AEs,  (dZ_max = 1)')
#ax.set_title('Number of AEs vs size of largest orbit')
plt.colorbar()
plt.clim(1,100000)
fig.savefig("figures/hist2d_norm_vs_numAE_dZ1_geom.png",dpi=500)
ax.clear()

#Initalize x and y array, i.e. AEs and related orbit quantity
AEs = np.empty((133885+1-4), dtype='int')
norm = np.empty((133885+1-4), dtype='float')
indices = np.array([], dtype='int')

#Get all the data:
for log_num in range(1,last_log_num+1):
    f = open('logs/'+'QM9_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'_dZ2_geom.txt', 'r')
    Lines = f.readlines()
    for line in Lines:
        x = line.split('\t')
        AEs[int(x[0].split('_')[1].lstrip('0'))-4] = int(x[3]) #1: time; 2: number of heavy atoms; 3: number of AEs
    f.close()

print('---------------------Outliers, 2-norm of CM----------------------')
for log_num in range(1,last_log_num+1):
    f = open('logs/'+'QM9_norm_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'_dZ2_geom.txt', 'r')
    Lines = f.readlines()
    for line in Lines:
        x = line.split('\t')
        norm[int(x[0].lstrip('0'))-4] = float(x[1])*0.001 #1: norm

        #print out all contestants with high norms beyond 110,000 and save their index:
        if float(x[1]) > 110000:
            #print('tag: '+x[0]+'\tNumber of atoms in orbits: '+str(x[2])+'\tNumber of AEs: '+str(AEs[a])+'\tMax orbit size: '+str(x[3]))
            print(x[0])
            indices = np.append(indices, int(x[0].lstrip('0'))-4)
    f.close()


fig, ax = plt.subplots(tight_layout=True)
plt.hist2d(norm, AEs,bins=bins, norm = colors.LogNorm())
ax.set(xlabel=r'Norm of Coulomb matrix of the molecule / $10^3$', ylabel='Number of AEs,  (dZ_max = 2)')
#ax.set_title('Number of AEs vs size of largest orbit')
plt.colorbar()
plt.clim(1,100000)
fig.savefig("figures/hist2d_norm_vs_numAE_dZ2_geom.png",dpi=500)
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

fig, ax = plt.subplots()
ax.scatter([2,3,4,5,6,7,8,9], QM9_percentages_dZ1, marker='x', color=color[0], label='dZ_max = 1 (QM9)')
ax.plot([2,3,4,5,6,7,8,9], QM9_percentages_dZ1,color=color[0])
ax.scatter([2,3,4,5,6,7,8,9], QM9_percentages_dZ2, marker='x', color=color[1], label='dZ_max = 2 (QM9)')
ax.plot([2,3,4,5,6,7,8,9], QM9_percentages_dZ2,color=color[1])

ax.scatter([2,3,4,5,6,7,8,9], QM9_percentages_uncolored_dZ1, marker='x', color=color[2], label='dZ_max = 1 (QM9, u.c.)')
ax.plot([2,3,4,5,6,7,8,9], QM9_percentages_uncolored_dZ1,color=color[2])
ax.scatter([2,3,4,5,6,7,8,9], QM9_percentages_uncolored_dZ2, marker='x', color=color[3], label='dZ_max = 2 (QM9, u.c.)')
ax.plot([2,3,4,5,6,7,8,9], QM9_percentages_uncolored_dZ2,color=color[3])

ax.set_xticks(range(2,10))
ax.set_yticks(range(0,45,10))
ax.set_xlim([1.7, 9.3])
ax.set_ylim([0,42])
ax.set(xlabel='Number of heavy atoms N', ylabel='Amount of molecules with AEs / %')
#ax.grid(which='both')
#plt.yscale('log')
ax.legend(loc="upper right",prop={'size': 8},framealpha=1, edgecolor='black')
fig.savefig("figures/sweetspot_geom.png", dpi=500)
