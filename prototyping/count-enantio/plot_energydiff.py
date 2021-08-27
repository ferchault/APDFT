import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def average_from_file(inputfile, column):
    f = open('logs/'+inputfile, 'r')
    Lines = f.readlines()
    f.close()
    sum = 0
    num = 0
    for line in Lines:
        if len(line) == 1:
            continue
        else:
            #print(inputfile)
            #print(num)
            #print(len(line))
            if line[1] != '-':
                sum += abs(float(line.split('dsg')[int(column)].strip()))
                num += 1
    mean = sum/num
    for line in Lines:
        if len(line) == 1:
            continue
        else:
            if line[1] != '-':
                sum += (abs(float(line.split('dsg')[int(column)].strip()))-mean)**2
    SD = np.sqrt(sum/num)
    return mean, SD


def median_from_file(inputfile, column):
    number_AEs = 0
    f = open('logs/'+inputfile, 'r')
    Lines = f.readlines()
    f.close()
    full_list = []
    for line in Lines:
        if len(line) == 1:
            continue
        elif line[1] == '-':
            continue
        else:
            number_AEs += 1
            try:
                full_list.append(abs(float(line.split('ZINC')[int(column)].strip())))
            except:
                full_list.append(abs(float(line.split('\t')[int(column)].strip())))
    full_list.sort()
    median = full_list[int(len(full_list)*0.5)]
    return median, number_AEs

#---------------------------Plot Yukawa range-----------------------------------
"""
x = []
y = []
y_error = []
for range in [-1,0.2,0.3,0.5,1.0,1.5,4.0]:
    mean, SD = average_from_file('QM9_log_energydiff_dZ1_'+str(range)+'.txt',0)
    if range < 0:
        hline = mean
        hline_pos_err = mean+SD
        hline_neg_err = mean-SD
    else:
        x.append(range)
        y.append(mean)
        y_error.append(SD)

print(x)
print(y)
fig, ax = plt.subplots()
ax.scatter(x, y, marker='x', color='#1f77b4')
ax.axhline(y = hline_pos_err, color = '#000000', linestyle = '--')
ax.axhline(y = hline_neg_err, color = '#000000', linestyle = '--')
ax.axhline(y = hline, color = '#000000', linestyle = '-')
plt.errorbar(x, y, yerr= y_error, fmt='none', capsize=4, color='#1f77b4')
ax.set_xlim([0, 10.5])
ax.set(xlabel="Yukawa range [A]", ylabel=r'Average $\Delta E$ / Ha')
#ax.grid(which='both')
#plt.yscale('log')
#ax.legend(loc="upper right",framealpha=1, edgecolor='black')
fig.savefig("figures/yukawa_range.png", dpi=300)
#plt.show()
"""

#---------------------Plot Yukawa range from QM9, 2nd attempt-------------------
"""
x = []
y = []
y_error = []
for range in [-1,0.5,0.7,0.8,1.0,1.1,1.2,1.3,1.5,2.0,2.5]:

    #mean, SD = average_from_file('QM9_log_energydiff_dZ1_range'+str(range)+'.txt',0)
    median = median_from_file('QM9_log_energydiff_dZ1_range'+str(range)+'.txt',0)
    if range < 0:
        hline = median
        #hline = mean
        #hline_pos_err = mean+SD
        #hline_neg_err = mean-SD
    else:
        x.append(range)
        y.append(median)
        #y.append(mean)
        #y_error.append(SD)

print(x)
print(y)
fig, ax = plt.subplots()
ax.scatter(x, y, marker='x', color='#1f77b4')
#ax.axhline(y = hline_pos_err, color = '#000000', linestyle = '--')
#ax.axhline(y = hline_neg_err, color = '#000000', linestyle = '--')
ax.axhline(y = hline, color = '#000000', linestyle = '-')
#plt.errorbar(x, y, yerr= y_error, fmt='none', capsize=4, color='#1f77b4')
ax.set_xlim([0, 3.1])
#ax.set(xlabel="Yukawa range [A]", ylabel=r'Average $\Delta E$ / Ha')
ax.set(xlabel="Yukawa range [A]", ylabel=r'Median $\Delta E$ / Ha')
#ax.grid(which='both')
plt.yscale('log')
#ax.legend(loc="upper right",framealpha=1, edgecolor='black')
fig.savefig("figures/yukawa_range_QM9.png", dpi=300)
#plt.show()
#---------------------Plot Yukawa range from ZINC, 2nd attempt------------------

xZ = []
yZ = []
y_error = []
for range in [-1,1.0,2.0,2.5,3.0,3.5]:
    #mean, SD = average_from_file('QM9_log_energydiff_dZ1_range'+str(range)+'.txt',0)
    median = median_from_file('ZINC_log_energydiff_dZ1_range'+str(range)+'.txt',0)
    if range < 0:
        hline = median
        #hline = mean
        #hline_pos_err = mean+SD
        #hline_neg_err = mean-SD
    else:
        xZ.append(range)
        yZ.append(median)
        #y.append(mean)
        #y_error.append(SD)

print(xZ)
print(yZ)
fig, ax = plt.subplots()
ax.scatter(xZ, yZ, marker='x', color='#1f77b4')
#ax.axhline(y = hline_pos_err, color = '#000000', linestyle = '--')
#ax.axhline(y = hline_neg_err, color = '#000000', linestyle = '--')
ax.axhline(y = hline, color = '#000000', linestyle = '-')
#plt.errorbar(x, y, yerr= y_error, fmt='none', capsize=4, color='#1f77b4')
ax.set_xlim([0, 5.1])
#ax.set(xlabel="Yukawa range [A]", ylabel=r'Average $\Delta E$ / Ha')
ax.set(xlabel="Yukawa range [A]", ylabel=r'Median $\Delta E$ / Ha')
#ax.grid(which='both')
plt.yscale('log')
#ax.legend(loc="upper right",framealpha=1, edgecolor='black')
fig.savefig("figures/yukawa_range_ZINC.png", dpi=300)
#plt.show()
"""






#------------------Plot Yukawa range for the paper------------------------------
"""

For QM9 and ZINC_named, plot the number of AEs and the respective median Delta E
for the different Yuakwa ranges a

"""

ranges_QM9 = [-1,0.5,0.7,0.8,1.0,1.1,1.2,1.3,1.5,2.0,2.5]
ranges_ZINC = [-1,1.0,2.0,2.5,3.0,3.5]
median_E_QM9 = []
median_E_ZINC = []
numAE_QM9 = []
numAE_ZINC = []

for range in ranges_QM9:
    median, numAE = median_from_file('QM9_log_energydiff_dZ1_range'+str(range)+'.txt',0)
    if range < 0:
        #Yukawa range infinite, i.e. Coulomb potential
        hline_E_QM9 = median
        hline_num_QM9 = numAE/1388 #Normalize the number by batch size
    else:
        median_E_QM9.append(median)
        numAE_QM9.append(numAE/1388)

for range in ranges_ZINC:
    median, numAE = median_from_file('ZINC_log_energydiff_dZ1_range'+str(range)+'.txt',0)
    if range < 0:
        #Yukawa range infinite, i.e. Coulomb potential
        hline_E_ZINC = median
        hline_num_ZINC = numAE/1199 #Normalize the number by batch size
    else:
        median_E_ZINC.append(median)
        numAE_ZINC.append(numAE/1199) #Normalize the number by batch size

fig, ax_Delta_E = plt.subplots()

#Median energy differences
ax_Delta_E.plot(ranges_QM9[1:], median_E_QM9, label=r'$| \Delta E |$ (QM9)', color='tab:blue', marker='+')
ax_Delta_E.plot(ranges_ZINC[1:], median_E_ZINC, label=r'$| \Delta E |$ (ZINC)', color='tab:red', marker='+')
ax_Delta_E.set_xlabel(r'Yukawa range $a$ [$\AA$]', fontsize=14)
ax_Delta_E.set_ylabel(r'$| \Delta E |$  [Ha]', fontsize=14)
ax_Delta_E.axhline(y = hline_E_QM9, color = 'tab:blue', linestyle = 'solid', linewidth = 1, xmin=0.68)
ax_Delta_E.text(2.67,hline_E_QM9+0.0001,r'$| \Delta E |$ (QM9), $a \rightarrow \infty$', color='tab:blue')
ax_Delta_E.axhline(y = hline_E_ZINC, color = 'tab:red', linestyle = 'solid', linewidth = 1, xmin=0.68)
ax_Delta_E.text(2.67,hline_E_ZINC-0.003,r'$| \Delta E |$ (ZINC), $a \rightarrow \infty$', color='tab:red')
ax_Delta_E.set_ylim([0.0004,1])
ax_Delta_E.set_yscale('log')

#Number of AEs
ax_numAE = ax_Delta_E.twinx()
ax_numAE.plot(ranges_QM9[1:], numAE_QM9, label=r'$\#$ AEs (QM9)', color='tab:blue', linestyle = 'dashed', marker='+')
ax_numAE.plot(ranges_ZINC[1:], numAE_ZINC, label=r'$\#$ AEs (ZINC)', color='tab:red', linestyle = 'dashed', marker='+')
ax_numAE.set_ylabel(r'Av. $\#$ AEs per mol.', fontsize=14)
ax_numAE.axhline(y = hline_num_QM9, color = 'tab:blue', linestyle = 'dashed', linewidth = 1, xmin=0.68)
ax_numAE.text(2.67,hline_num_QM9+0.03,r'$\#$ AEs (QM9), $a \rightarrow \infty$', color='tab:blue')
ax_numAE.axhline(y = hline_num_ZINC, color = 'tab:red', linestyle = 'dashed', linewidth = 1, xmin=0.68)
ax_numAE.text(2.67,hline_num_ZINC-3,r'$\#$ AEs (ZINC), $a \rightarrow \infty$', color='tab:red')
ax_numAE.set_ylim([0.1,100])
ax_numAE.set_yscale('log')

#Literature values for method accuracy
LDA_E = 0.0468
GGA_E = 0.0172
hybrid_E = 0.0162
"""
Source: https://dft.uci.edu/pubs/RCFB08.pdf
"""
ax_Delta_E.axhline(y = LDA_E, color = 'black', linestyle = 'solid', linewidth = 1, xmax = 0.32)
ax_Delta_E.text(0.38,LDA_E + 0.005,'LDA')
ax_Delta_E.axhline(y = GGA_E, color = 'black', linestyle = 'solid', linewidth = 1, xmax = 0.32)
ax_Delta_E.text(0.38,GGA_E+0.002,'PBE')
ax_Delta_E.axhline(y = hybrid_E, color = 'black', linestyle = 'solid', linewidth = 1, xmax = 0.32)
ax_Delta_E.text(0.38,hybrid_E-0.004,'TPSSh')

h1, l1 = ax_Delta_E.get_legend_handles_labels()
h2, l2 = ax_numAE.get_legend_handles_labels()
ax_Delta_E.legend(h1+h2, l1+l2,loc="upper right",framealpha=0, edgecolor='black')

fig.tight_layout()
fig.savefig("figures/yukawa.png", dpi=400)




#OUTDATED!!!!!!
#------------------------Plot tolerance vs looseness----------------------------
"""
#tol_loose = np.ones((7,7))
Delta_E = np.ones((7,7))
Delta_E_SD = np.ones((7,7))

values = [0.02,0.1,0.5,0.8,1.0,1.2,1.5,2.0]
values_str = ['002','01','05','08','10','12','15','20']

for i in range(7):
    for j in range(7):
        try:
            median = median_from_file('QM9_log_energydiff_dZ1_-1tol'+values_str[i]+'_loose'+values_str[j]+'.txt',0)
            Delta_E[i][j] = median
        except:
            median = 0
        #mean, SD = average_from_file('QM9_log_energydiff_dZ1_-1tol'+values_str[i]+'_loose'+values_str[j]+'.txt',0)
        #Delta_E[i][j] = mean
        #Delta_E_SD[i][j] = SD


fig, ax = plt.subplots()
im = ax.imshow(Delta_E)

# We want to show all ticks...
ax.set_xticks(np.arange(len(values)))
ax.set_yticks(np.arange(len(values)))
# ... and label them with the respective list entries
ax.set_xticklabels([str(i) for i in values])
plt.xlabel('Tol. (Sim. of mol.)')
plt.ylabel('Loose. (Sim. of chem. env.)')
ax.set_yticklabels([str(i) for i in np.flip(values)])

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)

# Loop over data dimensions and create text annotations.
for i in range(len(values)):
    for j in range(len(values)):
        text = ax.text(j, i, "{:#.3g}".format(Delta_E[i, j]),
                       ha="center", va="center", color="w")

ax.set_title("Median energy difference of AEs when tuning tolerance and looseness")
fig.tight_layout()
fig.savefig("figures/energy_diff_loose_tol.png", dpi=300)
plt.show()
"""
