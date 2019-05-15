#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:16:34 2019

@author: misa

Test parameters for point charge
"""

from ase import Atoms
from ase.io import write
from gpaw import GPAW
from gpaw.external import PointChargePotential
from matplotlib import pyplot as plt

def energy_calc_with_expot(atom_conf, output, pc):
        
    calc = GPAW(xc='PBE', txt=output, external=pc)
    atom_conf.set_calculator(calc)
    
    pot_en = atom_conf.get_potential_energy()
    #density = calc.get_all_electron_density(gridrefinement=4)
    
    #return( pot_en, density )
    return(pot_en)

d = 0.74
a = 6.0

rc_list=[-1, -0.1, -1e-2, 1e-2, 1e-1, 1]


for rc_item in rc_list:
    output_He_expot_H2 = 'output/par_pc/'+'rc_'+str(rc_item).replace('.','_')+'.txt'
    
    atom_conf_He_expot_H2=Atoms('H2', positions=[(0, 0, 0),(0, 0, d)], cell=(a,a,a) )
    atom_conf_He_expot_H2.center()
    
    pos=atom_conf_He_expot_H2.get_positions()
    print('Position H2 = ' + str(pos))
    pc_He_expot_H2 = PointChargePotential( [1.0, -1.0], [ pos[0], pos[1] ], rc=rc_item )
    
    calc = GPAW(xc='PBE', txt=output_He_expot_H2, external=pc_He_expot_H2)
    atom_conf_He_expot_H2.set_calculator(calc)
    energy_He_expot_H2 = atom_conf_He_expot_H2.get_potential_energy()
    #energy_He_expot_H2 = energy_calc_with_expot(atom_conf_He_expot_H2, output_He_expot_H2, pc_He_expot_H2)
    print('Energy He calculated from H2 + external potential = ' + str(energy_He_expot_H2) )
    density = calc.get_all_electron_density(gridrefinement=4)
    write('output/par_pc/'+'dens_rc_'+str(rc_item).replace('.','_')+'.cube', atom_conf_He_expot_H2, data=density)