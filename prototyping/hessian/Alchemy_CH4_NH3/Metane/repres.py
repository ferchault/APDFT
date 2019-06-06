from horton import *
import numpy as np
from numpy.linalg import norm
from numpy import dot,cross

class repres:
    cartesian=np.ndarray
    nucl_masses=[]
    derivatives=[]
#    Internal=np.array
    def __init__(self,molecule):
        self.molecule=molecule
        self.cartesian=np.asarray(molecule.coordinates)
        self.cartesian-=self.cartesian[0]
        self.Internal=np.zeros(self.cartesian.shape[0]*3-6)  ### for the future check linearity of the molecule 
        self.total_mass=0.
        for i in range(len(molecule.numbers)):
            self.nucl_masses.append(periodic[molecule.numbers[i]].mass)
            self.total_mass+=periodic[molecule.numbers[i]].mass
        self.update_IC()
        self.update_Cart()
    def get_center_of_mass(self):
        mass_center=np.zeros(3)
        for i in range(len(self.nucl_masses)):
            mass_center+=self.nucl_masses[i]/self.total_mass*self.cartesian[i]
        return mass_center
    def center(self):
        cm=self.get_center_of_mass()
        for i in self.cartesian:
            i-=cm
    def update_IC(self):
        i=self.cartesian.shape[0]
        if i==2:
            self.Internal[0]=np.linalg.norm(self.cartesian[1]-self.cartesian[0])
            return
        elif i==3:
            self.Internal[0]=np.linalg.norm(self.cartesian[1])
            self.Internal[1]=np.linalg.norm(self.cartesian[2])
            self.Internal[2]=np.dot(self.cartesian[2],self.cartesian[1])
            return
        elif i>=4:
            self.Internal[0]=np.linalg.norm(self.cartesian[1])
            self.Internal[1]=np.linalg.norm(self.cartesian[2])
            self.Internal[2]=np.dot(self.cartesian[2],self.cartesian[1])/self.Internal[0]/self.Internal[1]
            for j in range(3,i):
                self.Internal[3*(j-2)]=norm(self.cartesian[j])
                self.Internal[3*(j-2)+1]=np.dot(self.cartesian[j],self.cartesian[j-1])\
                /norm(self.cartesian[j])/norm(self.cartesian[j-1])
                self.Internal[3*(j-2)+2]=dot(self.cartesian[j],cross(self.cartesian[j-1],self.cartesian[j-2]))\
                /norm(self.cartesian[j])/norm(cross(self.cartesian[j-1],self.cartesian[j-2]))   
    def update_Cart(self):
        self.cartesian[0]=[0,0,0]
        self.cartesian[1]=[self.Internal[0],0,0]
        self.cartesian[2]=[self.Internal[1]*self.Internal[2],self.Internal[1]*(1-self.Internal[2]**2)**0.5,0]
        
        for i in range (3,self.cartesian.shape[0]):
            a=self.cartesian[i-1]/norm(self.cartesian[i-1])*self.Internal[3*(i-2)+1]
            b=cross(self.cartesian[i-1],self.cartesian[i-2])/norm(cross(self.cartesian[i-1],self.cartesian[i-2]))*self.Internal[3*(i-2)+2]
            c=cross(cross(self.cartesian[i-1],self.cartesian[i-2]),self.cartesian[i-1])*(1-self.Internal[3*(i-2)+2]**2-self.Internal[3*(i-2)+1]**2)**0.5\
                                          /norm(cross(cross(self.cartesian[i-1],self.cartesian[i-2]),self.cartesian[i-1]))
            self.cartesian[i]=self.Internal[3*(i-2)]*(a+b+c)
        self.molecule.coordinates=self.cartesian.tolist()
    
    def derive(self,cn,h):
        e0=uhf(self.molecule)
        dHV=np.zeros(self.Internal.shape)
        dHV[cn]=h
        self.Internal+=dHV #e(x+h)
        self.update_Cart()
        e1=uhf(self.molecule)
        self.Internal-=2*dHV #e(x-h)
        self.update_Cart()
        e2=uhf(self.molecule)
        f1=(e1-e2)/2./h
        f2=(e1-2*e0+e2)/h**2
        # --- move to next point ---
        dx=f1/f2
        self.Internal+=dHV*(1+dx/h/2.)
        print dHV*(1+dx/h/2.),f1,f2
        self.update_Cart()
        print f1,f2,'  dx ',dx,'energies = ',[e2,e1,e0],'*******************************************************'
        return [e2,e1,e0]
    
    def optimiz(self,h):
 #       print uhf(self.molecule),'_______________HJHJSHJA______'
        energies=[]
        for x in range (self.Internal.shape[0]):  ##0.1.3.6 b 2.4.7 a 5.8
            print '******************----',x
            i=self.derive (x,h)
            energies.append(i)
        return energies
#      to be implemented   
#    def opt_Cart (self,h):
#        e0=uhf(self.molecule)
#        dHV=np.zeros(self.Cartesian.shape)
#        dHV[cn]=h
#        self.Internal+=dHV #e(x+h)
#        self.update_Cart()
#        e1=uhf(self.molecule)
#        self.Internal-=2*dHV #e(x-h)
#        self.update_Cart()
#        e2=uhf(self.molecule)
#        f1=(e1-e2)/2./h
#        f2=(e1-2*e0+e2)/h**2
#        # --- move to next point ---
#        dx=f1/f2
#        self.Internal+=dHV*(1+dx/h/2.)
#        print dHV*(1+dx/h/2.),f1,f2
#        self.update_Cart()
#        print f1,f2,'  dx ',dx,'energies = ',[e2,e1,e0],'*******************************************************'
#        return [e2,e1,e0]
                
        
    def get_cartesian(self):
        return self.cartesian.tolist()# -*- coding: utf-8 -*-

