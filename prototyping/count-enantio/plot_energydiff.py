import matplotlib.pyplot as plt
import numpy as np

def average_from_file(inputfile, column):
    f = open('logs/'+inputfile, 'r')
    Lines = f.readlines()
    f.close()
    sum = 0
    num = 0
    for line in Lines:
        sum += abs(float(line.split('dsg')[int(column)].strip()))
        num += 1
    mean = sum/num
    for line in Lines:
        sum += (abs(float(line.split('dsg')[int(column)].strip()))-mean)**2
    SD = np.sqrt(sum/num)
    return mean, SD

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
