#For plotting all data: python3 QM9_plot.py &
#For drawing a molecule from the QM9 dataset: python3 draw_smiles.py $mode$ $molecule tag$
#For executing the entire search within QM9 (make sure to increase connection timeout):
#ssh -o ConnectTimeout=86400 user@workernode
#python3 AE.py &

import numpy as np

tolerance = 3 #Rounding error
performance_use = 0.90 #portion of cpu cores to be used
PathToNauty27r1 = '/home/simon/nauty27r1/'
PathToQM9XYZ = '/home/simon/QM9/XYZ/'

elements = {'Ghost':0,'H':1, 'He':2,
'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18,
'K':19, 'Ca':20, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,}

inv_elements = {v: k for k, v in elements.items()}

'''Below are all the partitions of splitting m_tot = np.sum(dZ_all[i])
atoms in a pure (i.e. uncolored/isoatomic) molecule in n=len(dZ_all[i]) partitions
for dZ_max = 3 up to m_tot = 8 and n = 2 and 3'''
m_possibilities = np.array([
[1,1],[1,1],[1,1],
[2,1],
[2,2],[2,2],[2,2],[3,1],
[2,3],
[3,3],[3,3],[3,3],[2,4],
[2,6],[4,4],[4,4],[4,4],
[1,1,1],
[2,1,1],
[1,1,3],[1,2,2],
[2,2,2],[1,1,4],[1,2,3],
[1,2,5],[1,2,5],[1,3,4],[1,3,4],[2,2,4],[2,3,3]
], dtype=object)
dZ_possibilities = np.array([
[1,-1],[2,-2],[3,-3],
[-1,2],
[1,-1],[2,-2],[3,-3],[-1,3],
[3,-2],
[1,-1],[+2,-2],[3,-3],[2,-1],
[3,-1],[1,-1],[2,-2],[3,-3],
[3,-2,-1],
[2,-1,-3],
[2,1,-1],[-2,2,-1],
[3,-2,-1],[3,1,-1],[3,-3,1],
[3,1,-1],[1,2,-1],[-2,2,-1],[-1,3,-2],[1,3,-2],[-3,3,-1]
],dtype=object)
