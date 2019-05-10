#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:25:51 2019

@author: misa
"""

from ase import Atoms
from ase.io import write
from gpaw import GPAW, PW
from gpaw.external import PointChargePotential
from matplotlib import pyplot as plt
import numpy as np


def energy_calc_with_expot(atom_conf, output, pc, charge_in=0.0):
        
    if pc==0.0:
        calc = GPAW(xc='PBE', txt=output)
    else:
        calc = GPAW(xc='PBE', txt=output, external=pc)
    
    if charge_in != 0.0:
        print('Charge='+str(charge_in))
        calc = GPAW(xc='PBE', txt=output, external=pc, charge=charge_in)
    
    atom_conf.set_calculator(calc)
    
    pot_en = atom_conf.get_potential_energy()
    #density = calc.get_all_electron_density(gridrefinement=4)
    
    #return( pot_en, density )
    return(pot_en)



#d = 1.09
#a = 10.0
#
#atom_conf_CO_from_N2=Atoms('N2', positions=[(0, 0, 0), (0, 0, d)], cell=(a,a,a) )
#atom_conf_CO_from_N2.center()
#pos=atom_conf_CO_from_N2.get_positions()
#print('Position=' + str(pos) )
#
#output_CO_from_N2='output/CO_from_N2.txt'
#
#
#pc_H = PointChargePotential( [-1.0, 1.0], [ pos[0], pos[1] ], rc=1e-5, width=10.0 )
#
#calc = GPAW( mode=PW(600), xc='PBE', txt=output_CO_from_N2, external=pc_H )
#
#atom_conf_CO_from_N2.set_calculator(calc)
#energy_CO_from_N2 = atom_conf_CO_from_N2.get_potential_energy()
##energy_He = energy_calc_with_expot(atom_conf_He, output_He, pc_He)
#print('Energy CO from N2 = ' + str(energy_CO_from_N2) )
#
########################################################################
#atom_conf_CO=Atoms('CO', positions=[(0, 0, 0), (0, 0, d)], cell=(a,a,a) )
#atom_conf_CO.center()
#pos=atom_conf_CO.get_positions()
#print('Position=' + str(pos) )
#
#output_CO='output/CO.txt'
#
#
##pc_H = PointChargePotential( [-1.0, 1.0], [ pos[0], pos[1] ], rc=1e-5, width=10.0 )
#
#calc = GPAW( mode=PW(600), xc='PBE', txt=output_CO)
#
#atom_conf_CO.set_calculator(calc)
#energy_N2 = atom_conf_CO.get_potential_energy()
##energy_He = energy_calc_with_expot(atom_conf_He, output_He, pc_He)
#print('Energy CO = ' + str(energy_N2) )



d = 0.74
a = 6
gs=0.04

atom_conf_He_expot_H2=Atoms('H2', positions=[(0, 0, 0), (0, 0, d)], cell=(a,a,a) )
atom_conf_He_expot_H2.center()
pos=atom_conf_He_expot_H2.get_positions()
print('Position=' + str(pos) )

output_He_expot_H2='/home/misa/APDFT/prototyping/gpaw/h2_he/output/alch_vs_normal/He_expot_H2.txt'


pc_H = PointChargePotential( [1.0, -1.0], [ pos[0], pos[1] ], rc=1e-3, width=20 )

calc = GPAW( xc='PBE', txt=output_He_expot_H2, external=pc_H, h=gs )

atom_conf_He_expot_H2.set_calculator(calc)
energy_He_expot_H2 = atom_conf_He_expot_H2.get_potential_energy()
#energy_He = energy_calc_with_expot(atom_conf_He, output_He, pc_He)
print('Energy He_expot_H2 = ' + str(energy_He_expot_H2) )
#density = calc.get_all_electron_density(gridrefinement=4)
#write('output/density_He_expot_H2.cube', atom_conf_He_expot_H2, data=density)

atom_conf_He=Atoms('He', positions=[(0, 0, d)], cell=(a,a,a) )
atom_conf_He.center()

output_He='output/alch_vs_normal/He.txt'


calc = GPAW( xc='PBE', txt=output_He, h=gs)

atom_conf_He.set_calculator(calc)
energy_He = atom_conf_He.get_potential_energy()
#energy_He = energy_calc_with_expot(atom_conf_He, output_He, pc_He)
print('Energy He = ' + str(energy_He) )



##############################################################
#      He calculated from H2 + external potential            #
##############################################################

#output_He_expot_H2 = 'output/He_expot_H2.txt'
#
#atom_conf_He_expot_H2=Atoms('H2', positions=[(0, 0, 0),(0, 0, d)], cell=(a,a,a) )
#atom_conf_He_expot_H2.center()
#
#pos=atom_conf_He_expot_H2.get_positions()
#print('Position H2 = ' + str(pos))
#pc_He_expot_H2 = PointChargePotential( [1.0, -1.0], [ pos[0], pos[1] ] )
#
#calc = GPAW(xc='PBE', txt=output_He_expot_H2, external=pc_He_expot_H2, setups={'H': 'ae'})
#atom_conf_He_expot_H2.set_calculator(calc)
#energy_He_expot_H2 = atom_conf_He_expot_H2.get_potential_energy()
##energy_He_expot_H2 = energy_calc_with_expot(atom_conf_He_expot_H2, output_He_expot_H2, pc_He_expot_H2)
#print('Energy He calculated from H2 + external potential = ' + str(energy_He_expot_H2) )
#density = calc.get_all_electron_density(gridrefinement=4)
#write('output/density_He_expot_H2.cube', atom_conf_He_expot_H2, data=density)

##############################################################
#      He without external potential                         #
##############################################################


#output_He='output/He.txt'
#
#atom_conf_He=Atoms('He', positions=[pos[0]], cell=(a,a,a) )
#pos_He=atom_conf_He_expot_H2.get_positions()
#print('Position He = ' + str(pos_He))
#
#pc_He = PointChargePotential( [0.0], [ atom_conf_He.get_positions() ] )
#
#calc = GPAW(xc='PBE', txt=output_He, external=pc_He, setups={'He': 'ae'})
#
#atom_conf_He.set_calculator(calc)
#energy_He = atom_conf_He.get_potential_energy()
##energy_He = energy_calc_with_expot(atom_conf_He, output_He, pc_He)
#print('Energy He = ' + str(energy_He) )
#
#density = calc.get_all_electron_density(gridrefinement=4)
#write('output/density_He.cube', atom_conf_He, data=density)


###############################################################  [3.0 , 3.0 , 2.63], [3.0 , 3.0 , 3.37]
##      H2 without external potential                         #
###############################################################
#pc_H2=0.0
#output_H2='output/H2.txt'
#
#atom_conf_H2=Atoms('H2', positions=[(0, 0, 0),(0, 0, d)], cell=(a,a,a) )
#atom_conf_H2.center()
#
#energy_H2 = energy_calc_with_expot(atom_conf_H2, output_H2, pc_H2)
#print('Energy H2 = ' + str(energy_H2) )
#
###############################################################
##      He calculated from H^- +   external potential         #
###############################################################
#pc_He_expot_H_minus = PointChargePotential( [1.0], [3.0 , 3.0 , 2.63] )
#output_He_expot_H_minus = 'output/He_expot_H_minus.txt'
#
#atom_conf_He_expot_H_minus = Atoms('H', positions=[(0, 0, 0)], cell=(a,a,a) )
#atom_conf_He_expot_H_minus.center()
#
#energy_He_expot_H_minus = energy_calc_with_expot(atom_conf_He_expot_H_minus, output_He_expot_H_minus, pc_He_expot_H_minus)
#print('Energy He calculated from H^- + external potential = ' + str(energy_He_expot_H_minus) )
