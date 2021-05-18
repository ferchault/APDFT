import matplotlib.pyplot as plt
import numpy as np

def average_from_file(inputfile, column):
    f = open('logs/'+inputfile, 'r')
    Lines = f.readlines()
    f.close()
    sum = 0
    num = 0
    for line in Lines:
        sum += abs(float(line.split('\t')[int(column)]))
        num += 1
    return sum/num

x = []
y = []
mass = 0
while mass < 5:
    x.append(mass)
    y.append(average_from_file('QM9_log_energydiff_dZ1_'+str(mass)+'.txt',0))
    mass += 0.5

print(x)
print(y)
