#Compare the same property taken from QM9 with calculation from xTB
from MAG import *
from config import *
import os
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    #Get the data:
    '''last_log_num = 1
    QM9_data = []
    xTB_data = []
    for log_num in range(1,last_log_num+1):
        f = open('logs/xTBvsQM9_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'.txt', 'r')
        Lines = f.readlines() # 0:tag 1:QM9-based internal energy 2:xTB calculation
        f.close()
        for line in Lines:
            x = line.split(' ')
            energy_atomref = get_energy_const_atoms(PathToQM9XYZ, 'dsgdb9nsd_' + x[0] + '.xyz')
            QM9_data.append(float(x[1])-energy_atomref)
            xTB_data.append(float(x[2]))
            #New attempt: nuclear repulsion:
            #xTB_data.append(parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_' + x[0] + '.xyz').get_energy_NN())

    fig, ax = plt.subplots()
    ax.scatter(QM9_data, xTB_data, marker='x')
    #ax.set_xticks(range(2,10))
    ax.set_xlim([-4, 0])
    ax.set_ylim([-30,0])
    ax.set(xlabel='Total energy from QM9 / Hartrees', ylabel='Total energy from xTB/ Hartrees')
    #ax.grid(which='both')
    #plt.yscale('log')
    #ax.legend(loc="upper left",framealpha=1, edgecolor='black')
    fig.savefig("figures/xTBvsQM9.png", dpi=500)'''


    #Get the data:
    '''last_log_num = 1
    QM9_data = []
    SCF_data = []
    for log_num in range(1,last_log_num+1):
        f = open('logs/SCFvsQM9_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'.txt', 'r')
        Lines = f.readlines() # 0:tag 1:QM9-based internal energy 2:xTB calculation
        f.close()
        for line in Lines:
            x = line.split('\t')
            QM9_data.append(float(x[1]))
            SCF_data.append(float(x[2]))

    fig, ax = plt.subplots()
    ax.scatter(QM9_data, SCF_data, marker='x')
    #ax.set_xticks(range(2,10))
    #ax.set_xlim([-4, 0])
    #ax.set_ylim([-30,0])
    ax.set(xlabel='Total energy from QM9 / Hartrees', ylabel='Total energy from scf.hf / Hartrees')
    #ax.grid(which='both')
    #plt.yscale('log')
    #ax.legend(loc="upper left",framealpha=1, edgecolor='black')
    fig.savefig("figures/SCFvsQM9.png", dpi=500)'''


    #Get the data:
    last_log_num = 1
    diff_data = []
    for log_num in range(1,last_log_num+1):
        f = open('logs/SCFvsQM9_log'+ '00'[:(2-len(str(log_num)))] + str(log_num) +'.txt', 'r')
        Lines = f.readlines() # 0:tag 1:QM9-based internal energy 2:xTB calculation
        f.close()
        for line in Lines:
            x = line.split('\t')
            diff_data.append(np.abs(float(x[1])-float(x[2])))

    plt.hist(diff_data, bins=50, alpha=0.9)
    #ax.set_xticks(range(2,10))
    plt.yticks([])
    #ax.set_xlim([-4, 0])
    #ax.set_ylim([-30,0])
    plt.title('Histogram of absolute energy difference')
    plt.xlabel('Total energy difference between QM9 and scf.hf / Hartrees')
    plt.ylabel('a.u.')
    #ax.grid(which='both')
    #plt.yscale('log')
    #ax.legend(loc="upper left",framealpha=1, edgecolor='black')
    plt.savefig("figures/SCFvsQM9_hist.png", dpi=500)
