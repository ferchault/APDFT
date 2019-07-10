#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:38:57 2019

@author: giorgiod
"""

import sys 
import os
import argparse 
from basis_set_exchange import lut
import numpy as np
import shutil
"""
Given alchemy expansion order  AEO
Given a geometry correction order GCO
Given a xyz DIATOMIC molecule 
I need to setup the (single point ) calculations (and folders) for MRCC
The level of theory for the calculations is CCSD/cc-pVTZ (fixed for now)

1) Parsing the input (xyz file location ,alchemy order AO ,geometry correction order GCO)
2) Reading input and planning the alchemy


"""
### read_xyz from apdft.__init__()   ===> future import from file 
def get_element_number(element):
	return lut.element_Z_from_sym(element)
def read_xyz(fn):       
	with open(fn,'r') as fh:   ## read 'r' status ??? 
		lines = fh.readlines()
	numatoms = int(lines[0].strip())
	lines = lines[2:2 + numatoms]
	nuclear_numbers = []
	coordinates = []
	for line in lines:
		line = line.strip()
		if len(line) == 0:
			break
		parts = line.split()
		nuclear_numbers.append(get_element_number(parts[0]))
		coordinates.append([float(_) for _ in parts[1:4]])
	return np.array(nuclear_numbers), np.array(coordinates)
### end files from apdft project

method="CCSD"
basisset= "cc-pVTZ"
numatoms=2
coordinates=np.asarray([[0.,0.,0.],[0.,0.,0.]])
charges=[0.,0.]
mrcc_code ="""calc={} 
mem=5GB 
core=corr 
ccprog=mrcc 
basis={} 
grdens=on 
rgrid=Log3 
grtol=10 
agrid=LD0770 
molden=off
unit=angs 
geom=xyz

{} 
{}  {}  {}  {}  
{}  {}  {}  {}  

qmmm=Amber
pointcharges
 
{} 
{}  {}  {}  {}
{}  {}  {}  {}  
"""    

#format(method,basisset,numatoms,Ref_Numbers[0],[x for x in coordinates[0]],Ref_Numbers[1]\
#       ,[x for x in coordinates[1]],numatoms,Ref_Numbers[0],[x for x in coordinates[0]],Ref_Numbers[1],[x for x in coordinates[1]]))




#################------------ START PARSER CODE  ----------------------------################################
parser= argparse.ArgumentParser("Parsing posizional arguments: .xyz file location; Alchemy Order; Geometry Correction Order;\
                                [-t] target (diatomic) molecule from space separated atomic numbers eg.CO= \"-t 6 8 \" " )
parser.add_argument("xyz",help=".xyz file location",type=str)
parser.add_argument("ao",help="Alchemical Order",type=int)
parser.add_argument("gco",help="Geometry correction order",type=int)
parser.add_argument("-t", "--target",help="target molecule eg.CO= \"-t 6 8 \"" ,type=int,nargs=2 )
args=parser.parse_args(['co.xyz','5','5','-t','7','7'])

xyz,ao,gco,target = args.xyz,args.ao,args.gco,args.target
print ("the Alchemy expansion was set at {} Order; \n the geometry expansion was set at {} Order".format(xyz,ao,gco))
while True:
    try:
        Ref_Numbers,Ref_coordinates=read_xyz(xyz)
    except Exception as e:
        print (e.__cause__)
        print ("Can't open file: {};".format(xyz))
    #    print ("type a correct file !!!\n ")
        xyz=input("type a correct file name !!!\n ")
    else:
        break


if len(Ref_Numbers)!=2 or len(Ref_coordinates)!=2:
    sys.exit("the reference molecule is not a biatomic")

### --------- default  target lambda = + - 1
if not target:
    if Ref_Numbers[0]<Ref_Numbers[1]:
        target=Ref_Numbers+np.array([1,-1])
    else:
        target=Ref_Numbers-np.array([1,-1])
else:
    target=np.array(target)

ref_name=xyz.split('.')[0]  ## the name of the reference compound 
Main_dir="from_{}_{}_to_{}_A{}_G{}".format(ref_name,str(Ref_Numbers[0])+str(Ref_Numbers[1]),str(target[0])+str(target[1]),ao,gco)
try: 
    os.mkdir(Main_dir)
except FileExistsError:
    while True:
        ovr=input ("a similar calculation was found, do you want to overwrite? (Y/N)")
        if ovr.__contains__('y') or  ovr.__contains__('Y'):
            shutil.rmtree(Main_dir)
            os.mkdir(Main_dir)
            break
        if ovr.__contains__('n') or  ovr.__contains__('N'):
            sys.exit("Change directory name ./{} , then restart".format(Main_dir))

# Planning Alchemy ---  
dl=0.05
DL=Ref_Numbers-target
dl=DL*dl
dx = 0.02 ## Angstrom 
D0=np.linalg.norm(Ref_coordinates[1]-Ref_coordinates[0]) ### coordinates should be (A1) 0. 0. 0. / (A2) D  0. 0. 
os.chdir(Main_dir)
coordinates=np.asarray([[0.,0.,0.],[0.,0.,0.]])
charges=[0.,0.]

### for now we'll do 5 fixed points
for x in range(-2,3): ## change geom
    coordinates[1][0]=D0+dx*x
    for y in range(-2,3):  ## changing in alchemy order
        sub_dir="A_{}_G_{}".format(y,x)
        os.mkdir(sub_dir)
        os.chdir(sub_dir)
        with open("MINP",'w') as minp:
            minp.write(mrcc_code.format(method,basisset,numatoms,\
                    Ref_Numbers[0],coordinates[0][0],coordinates[0][1],coordinates[0][2],\
                    Ref_Numbers[1],coordinates[1][0],coordinates[1][1],coordinates[1][2],\
                    numatoms,\
                    dl[0]*y,coordinates[0][0],coordinates[0][1],coordinates[0][2],\
                    dl[1]*y,coordinates[1][0],coordinates[1][1],coordinates[1][2]))
        os.chdir('..')
        
print ('saludos')
#os.mkdir()