#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:41:52 2019

@author: misa

calculates energy of diatomic molecule (explicitly for H2) where the charge of the nuclei is changed
by addition of an external potential

Input: List with fractional charges that are added in the external potential

Output: energy as a function of the external charged as a: Plot of added charge vs energy; table
Output is stored as a table at save_path = '/home/misa/APDFT/prototyping/gpaw/h2_he/output/alch_chrg_vs_en.txt'


"""

from ase import Atoms
from ase.io import write
from gpaw import GPAW
from gpaw.external import PointChargePotential
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# adds external potential on diatomic moleculeto generate nuclei with fractional charges
# input: fractional charge added to the atoms
# output: energy using these fractionally charged nuclei

def energy(lamb):
    #############################################################
    #      He  created from H2 and external potential          #
    #############################################################
    
    d = 0.74
    a = 6.0
    
    atoms_external = Atoms('H2', positions=[(0, 0, 0),(0, 0, d)], cell=(a,a,a) )
    atoms_external.center()
    
    pc = PointChargePotential( [lamb, -lamb], [ [3.0 , 3.0 , 2.63], [3.0 , 3.0 , 3.37] ] )
    
    output='output/result'+str(lamb).replace('.','_')+'.txt'
    calc = GPAW(xc='PBE', txt=output, external=pc)
    atoms_external.set_calculator(calc)
    
    pot_en = atoms_external.get_potential_energy()
    density = calc.get_all_electron_density(gridrefinement=4)
    write('output/density_'+str(lamb).replace('.','_')+'.cube', atoms_external, data=density)
    
    return( pot_en )

d = 0.74
a = 6.0

step=0.1
#lambda_list=np.arange(0, 1+step, step)
lambda_list=[1.0]

energy_lambda=[]

for lamb in lambda_list:
    tot_energy=energy(lamb)
    energy_lambda.append(tot_energy)

plt.xlabel('Charge of external point potential in atomic units')
plt.ylabel('Energy of system')
plt.plot(lambda_list, energy_lambda, linestyle='--', marker='o', color='b')



# write it
energy_table = pd.DataFrame(data=[lambda_list, energy_lambda])
energy_table = energy_table.transpose()

save_path = '/home/misa/APDFT/prototyping/gpaw/h2_he/output/alch_chrg_vs_en.txt'
energy_table.to_csv(save_path, sep='\t', header=False, index=False)





