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
            full_list.append(abs(float(line.split('dsg')[int(column)].strip())))
    full_list.sort()
    median = full_list[int(len(full_list)*0.5)]
    return median

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

#---------------------------Plot Yukawa range, 2nd attempt----------------------

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
fig.savefig("figures/yukawa_range_ver2.png", dpi=300)
#plt.show()

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
