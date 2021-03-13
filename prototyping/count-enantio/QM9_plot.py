import matplotlib.pyplot as plt
import numpy as np
import os

def import QM9_log_line(input_file, line_num):
    if os.path.isfile(input_file):
        #open text file in read mode
        f = open(input_file, "r")
        line = f.read()[line_num]
        f.close()
        # returns time, number of atoms, number of AEs
        return line.split('\t')[1],line.split('\t')[2],line.split('\t')[3]
    else:
        print('File', input_file, 'not found.')
        return 0
