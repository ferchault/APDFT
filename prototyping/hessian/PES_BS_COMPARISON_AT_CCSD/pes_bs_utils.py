#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:36:30 2019

@author: giorgiod
"""
import numpy as np
import sys 
import os
import shutil

sys.path.append('/home/giorgiod/Documents/APDFT/dep')
sys.path.append('/home/giorgiod/Documents/APDFT/src')

#from apdft import calculator
proj_path='/home/giorgiod/MRCC_interface/PES-BS-ALCHEMY/'

# general parse 
def parse_energy(log_file,calc):
    calc_str={"HF":"FINAL HARTREE-FOCK ENERGY","ccsd" : "Total CCSD energy", "MP2":"Total MP2 energy [au]",\
             "PBE":"FINAL KOHN-SHAM ENERGY","B3LYP":"FINAL KOHN-SHAM ENERGY"}
    """Parse the hartree Fock energy from an MRCC output file"""
    try:
        with open(log_file,'r') as logf:
            while True:
                line=logf.readline()
                if calc_str[calc] in line:
                    good_line=line
                    for x in good_line.split(' '):
                        try:
                            float(x)
                            return (float(x))
                        except:
                            pass    
    except:
        print('couldn\'t parse energy return 0 for: '+log_file+'_calc__'+ calc)
        return 0    
    
def parse_energy_cc(log_file):
    """Parse the couple cluster energy from an MRCC output file"""
    try:
        with open(log_file,'r') as logf:
            while True:
                line=logf.readline()
                if "Final results:" in line:
                    good_line=logf.readline()
                    if "Total CCSD energy" in good_line:
                        for x in good_line.split(' '):
                            try:
                                float(x)
                                return (float(x))
                            except:
                                pass    
    except:
        print('couldn\'t parse energy return 0 for: '+log_file)
        return 0

def energy(bs,mol,pt,al=0):
    return parse_energy_cc(dirName(bs,mol,pt,al)+'run.log')

def dirName(basis_set,molecule,point,alchemy=0):
    return proj_path+basis_set+'/'+molecule+"/A{}_Radius:{}".format(alchemy,str(point)[0:3])+'/'

def parse_energy_hf(log_file):
    """Parse the hartree Fock energy from an MRCC output file"""
    try:
        with open(log_file,'r') as logf:
            while True:
                line=logf.readline()
                if "FINAL HARTREE-FOCK ENERGY:" in line:
                    good_line=line
                    for x in good_line.split(' '):
                        try:
                            float(x)
                            return (float(x))
                        except:
                            pass    
    except:
        print('couldn\'t parse energy return 0 for: '+log_file)
        return 0
    

def energy_hf(bs,mol,pt,al=0):
    return parse_energy_hf(dirName(bs,mol,pt,al)+'run.log')