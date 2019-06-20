#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:24:02 2019

@author: misa

Line plots of density, sqrt of density, density gradient, second derivatives and laplacian
to figure out why own function and GPAW function for \hat{T} \sqrt{\tilde{\rho}} give different results


"""

import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/gpaw/tools')
from gpaw import setup_paths
setup_paths.insert(0, '/home/misa/APDFT/prototyping/gpaw/OFDFT')

import numpy as np
from scipy.ndimage.filters import laplace
from gpaw import GPAW
import processing as pr
from matplotlib import pyplot as plt
from matplotlib import rcParams as rc
from ase.units import Bohr, Hartree

def get_line_plots(func_objs, plot_order):
    fig, ax = plt.subplots(plot_order)
    
    




###############################################################################
#                                    Data                                     #
###############################################################################
    
# load results from GPAW calculation
calc2 = GPAW('/home/misa/APDFT/prototyping/gpaw/OFDFT/result_64_gpts.gpw')

# func values from gpaw
pseudo_dens = calc2.density.nt_sG[0]
sqrt_pseudo_dens = np.sqrt(pseudo_dens)

# Func objects
Pseudo_Dens = pr.Func_3var(calc2) # pseudo density

kwargs_sqrt_pseudo_dens = {'func_value' : sqrt_pseudo_dens, 'length_cell' : [12.0, 12.0, 12.0]} # sqrt pseudo density
Sqrt_Pseudo_Dens = pr.Func_3var(**kwargs_sqrt_pseudo_dens)

gradient_z_sqrt_pseudo_dens = Sqrt_Pseudo_Dens.get_gradient()[2] # first derivative of sqrt ps_dens with respect to z-coordinate
kwargs_gradient_z_sqrt_pseudo_dens = {'func_value' : gradient_z_sqrt_pseudo_dens, 'length_cell' : [12.0, 12.0, 12.0]} 
Grad_Z_Sqrt_Pseudo_Dens = pr.Func_3var(**kwargs_gradient_z_sqrt_pseudo_dens)

hess_xx_sqrt_pseudo_dens = Sqrt_Pseudo_Dens.get_hessian()[0][0] # second derivative of sqrt ps_dens with respect to x-coordinate
kwargs_hess_xx_sqrt_pseudo_dens = {'func_value' : hess_xx_sqrt_pseudo_dens, 'length_cell' : [12.0, 12.0, 12.0]}
Hess_XX_Sqrt_Pseudo_Dens = pr.Func_3var(**kwargs_hess_xx_sqrt_pseudo_dens)

hess_yy_sqrt_pseudo_dens = Sqrt_Pseudo_Dens.get_hessian()[1][1] # second derivative of sqrt ps_dens with respect to y-coordinate
kwargs_hess_yy_sqrt_pseudo_dens = {'func_value' : hess_yy_sqrt_pseudo_dens, 'length_cell' : [12.0, 12.0, 12.0]}
Hess_YY_Sqrt_Pseudo_Dens = pr.Func_3var(**kwargs_hess_yy_sqrt_pseudo_dens)

hess_zz_sqrt_pseudo_dens = Sqrt_Pseudo_Dens.get_hessian()[2][2] # second derivative of sqrt ps_dens with respect to z-coordinate
kwargs_hess_zz_sqrt_pseudo_dens = {'func_value' : hess_zz_sqrt_pseudo_dens, 'length_cell' : [12.0, 12.0, 12.0]}
Hess_ZZ_Sqrt_Pseudo_Dens = pr.Func_3var(**kwargs_hess_zz_sqrt_pseudo_dens)

kin_applied_sqrt_pseudo_dens = Sqrt_Pseudo_Dens.get_kin_en_op()  # negaitve \hat{T} sqrt ps_dens (kinetic energy operator applied to sqrt ps_dens)
kwargs_kin_applied_sqrt_pseudo_dens_neg = {'func_value' : -kin_applied_sqrt_pseudo_dens, 'length_cell' : [12.0, 12.0, 12.0]}
Kin_Applied_Sqrt_Pseudo_Dens_neg = pr.Func_3var(**kwargs_kin_applied_sqrt_pseudo_dens_neg)

kin_applied_sqrt_pseudo_dens = Sqrt_Pseudo_Dens.get_kin_en_op()  # \hat{T} sqrt ps_dens (kinetic energy operator applied to sqrt ps_dens)
kwargs_kin_applied_sqrt_pseudo_dens = {'func_value' : kin_applied_sqrt_pseudo_dens, 'length_cell' : [12.0, 12.0, 12.0]}
Kin_Applied_Sqrt_Pseudo_Dens = pr.Func_3var(**kwargs_kin_applied_sqrt_pseudo_dens)

lapl=laplace(sqrt_pseudo_dens) # \hat{T} sqrt ps_dens using scipy function to apply laplace operator
kwargs_laplace_scipy = {'func_value' : -0.5*lapl, 'length_cell' : [12.0, 12.0, 12.0]}
Laplace_Scipy = pr.Func_3var(**kwargs_laplace_scipy)
Laplace_Scipy.func_value = Laplace_Scipy.func_value/Laplace_Scipy.spacing[2]**2 # rescaling because scipy.laplace assumes spacing = 1

kin_op_gpaw = np.zeros(sqrt_pseudo_dens.shape, dtype=float) # \hat{T} sqrt ps_dens using gpaw function
calc2.wfs.kin.apply(sqrt_pseudo_dens, kin_op_gpaw, phase_cd=None)
kwargs_kin_gpaw = {'func_value': kin_op_gpaw, 'length_cell':[12, 12, 12]}
Kinetic_gpaw = pr.Func_3var(**kwargs_kin_gpaw)



###############################################################################
#                                    Plots                                    #
###############################################################################

# parameters
par_z_func = ['x', 'y'], [6.0/Bohr, 6.0/Bohr] # parameters to select function value along bond axis from Func_3var object
z_coord = Pseudo_Dens.coordinates[2] # z coordinates of cell


## plot pseudo density and sqrt of density
#ps_dens_along_bond = Pseudo_Dens.select_line(par_z_func[0], par_z_func[1]) # func value pseudo density
#sqrt_ps_dens_along_bond = Sqrt_Pseudo_Dens.select_line(par_z_func[0], par_z_func[1]) # func value sqrt density
#fig, ax = plt.subplots(1, 1)
#ax.plot(z_coord, ps_dens_along_bond, label=r'$\tilde{\rho}$')
#ax.plot( z_coord, sqrt_ps_dens_along_bond, label=r'$\sqrt{\tilde{\rho}}$')
#ax.legend()


## plot sqrt, first, second derivative, \hat{T}\sqrt{ps_dens} along bond axis
#obj_list= [Sqrt_Pseudo_Dens, Grad_Z_Sqrt_Pseudo_Dens, Hess_ZZ_Sqrt_Pseudo_Dens, Kin_Applied_Sqrt_Pseudo_Dens_neg]
#label_list = [r'$\sqrt{\tilde{\rho}}$', r'$ \frac{d}{dz} \sqrt{\tilde{\rho}}$', r'$ \frac{d^2}{dz^2} \sqrt{\tilde{\rho}}$', r'$ \frac{1}{2} \nabla^2 \sqrt{\tilde{\rho}}$']
#fig_der, ax_der = plt.subplots(len(obj_list),1)
#for idx, obj in enumerate(obj_list):
#    func_z = obj.select_line(par_z_func[0], par_z_func[1])
#    ax_der[idx].plot( z_coord, func_z, label=label_list[idx])
#    ax_der[idx].legend()


## plot second derivatives along x ,y, z to understand why \hat{T} sqrt ps_dens is < 0 
#obj_list = [Hess_XX_Sqrt_Pseudo_Dens, Hess_YY_Sqrt_Pseudo_Dens, Hess_ZZ_Sqrt_Pseudo_Dens, Kin_Applied_Sqrt_Pseudo_Dens_neg, Laplace_Scipy]
#label_list = [r'$ \frac{d^2}{dx^2} \sqrt{\tilde{\rho}}$', r'$ \frac{d^2}{dy^2} \sqrt{\tilde{\rho}}$', r'$ \frac{d^2}{dz^2} \sqrt{\tilde{\rho}}$', r'$ \frac{1}{2} \nabla^2 \sqrt{\tilde{\rho}}$']
#fig_der, ax_der = plt.subplots(len(obj_list),1)
#for idx, obj in enumerate(obj_list):
#    func_z = obj.select_line(par_z_func[0], par_z_func[1])
#    ax_der[idx].plot( z_coord, func_z, label=label_list[idx])
#    ax_der[idx].legend()


## plot \hat{T} sqrt ps+dens for own, scipy-laplace and gpaw implementation at one gpt size with and without rescaling
#obj_list = [Kin_Applied_Sqrt_Pseudo_Dens, Laplace_Scipy, Kinetic_gpaw]
#label_list = [r'own implentation', r'scipy implentation rescaled', r'gpaw']
#rc.update({'font.size': 20})
#fig_der, ax_der = plt.subplots(1,1)
#for idx, obj in enumerate(obj_list):
#    func_z = obj.select_line(par_z_func[0], par_z_func[1])
#    ax_der.plot( z_coord, func_z, label=label_list[idx])
## scipy without rescaling    
#kwargs_laplace_scipy_no_scaling = {'func_value' : Laplace_Scipy.func_value*Laplace_Scipy.spacing[2]**2, 'length_cell' : [12.0, 12.0, 12.0]}
#Laplace_Scipy_no_scale = pr.Func_3var(**kwargs_laplace_scipy_no_scaling)
#func_z = Laplace_Scipy_no_scale.select_line(par_z_func[0], par_z_func[1])
#ax_der.plot( z_coord, func_z, ':', color='darkorange' ,label='scipy implentation\nbefore rescaling')
#ax_der.set_xlim(8, 14.5)
#ax_der.legend()

# line plots with different gpts and own, scipy, gpaw method for calculation of \hat{T} \sqrt{ps_dens}
obj_list = [Kin_Applied_Sqrt_Pseudo_Dens, Laplace_Scipy, Kinetic_gpaw]
label_list = [r'own implentation', r'scipy implentation rescaled', r'gpaw']

# load results from GPAW calculation
gpt64 = GPAW('/home/misa/APDFT/prototyping/gpaw/OFDFT/result_64_gpts.gpw')
gpt100 = GPAW('/home/misa/APDFT/prototyping/gpaw/OFDFT/result_100_gpts.gpw')
gpt128 = GPAW('/home/misa/APDFT/prototyping/gpaw/OFDFT/result_128_gpts.gpw')
gpt200 = GPAW('/home/misa/APDFT/prototyping/gpaw/OFDFT/result_200_gpts.gpw')

calc_obj_list = [gpt64, gpt100, gpt128, gpt200]
label_list=['64 gpts', '100 gpts', '128 gpts', '200 gpts']
fig, ax = plt.subplots(len(calc_obj_list), 2)



rc.update({'font.size': 10})
fig, ax = plt.subplots(len(calc_obj_list),2)

for idx, calc_obj in enumerate(calc_obj_list):

    # func values from gpaw
    pseudo_dens = calc_obj.density.nt_sG[0]
    sqrt_pseudo_dens = np.sqrt(pseudo_dens)
    
    # Func objects
    kwargs_sqrt_pseudo_dens = {'func_value' : sqrt_pseudo_dens, 'length_cell' : [12.0, 12.0, 12.0]} # sqrt pseudo density
    Sqrt_Pseudo_Dens = pr.Func_3var(**kwargs_sqrt_pseudo_dens)
    
    kin_applied_sqrt_pseudo_dens = Sqrt_Pseudo_Dens.get_kin_en_op()  # \hat{T} sqrt ps_dens (kinetic energy operator applied to sqrt ps_dens)
    kwargs_kin_applied_sqrt_pseudo_dens = {'func_value' : kin_applied_sqrt_pseudo_dens, 'length_cell' : [12.0, 12.0, 12.0]}
    Kin_Applied_Sqrt_Pseudo_Dens = pr.Func_3var(**kwargs_kin_applied_sqrt_pseudo_dens)
    
    lapl=laplace(sqrt_pseudo_dens) # \hat{T} sqrt ps_dens using scipy function to apply laplace operator
    kwargs_laplace_scipy = {'func_value' : -0.5*lapl, 'length_cell' : [12.0, 12.0, 12.0]}
    Laplace_Scipy = pr.Func_3var(**kwargs_laplace_scipy)
    Laplace_Scipy.func_value = Laplace_Scipy.func_value/Laplace_Scipy.spacing[2]**2 # rescaling because scipy.laplace assumes spacing = 1
    
    kin_op_gpaw = np.zeros(sqrt_pseudo_dens.shape, dtype=float) # \hat{T} sqrt ps_dens using gpaw function
    calc_obj.wfs.kin.apply(sqrt_pseudo_dens, kin_op_gpaw, phase_cd=None)
    kwargs_kin_gpaw = {'func_value': kin_op_gpaw, 'length_cell':[12, 12, 12]}
    Kinetic_gpaw = pr.Func_3var(**kwargs_kin_gpaw)
    
    
    obj_list = [Kin_Applied_Sqrt_Pseudo_Dens, Laplace_Scipy, Kinetic_gpaw]
    
    gpts = str(Kin_Applied_Sqrt_Pseudo_Dens.shape[0])
    label_list = [r'own '+ gpts +' gpts', r'scipy '+ gpts +' gpts', r'gpaw '+ gpts +' gpts']
    z_coord =  Kin_Applied_Sqrt_Pseudo_Dens.coordinates[2]
    
    for idx2, obj in enumerate(obj_list):
        func_z = obj.select_line(par_z_func[0], par_z_func[1])
        ax[idx][0].plot(z_coord, func_z, label=label_list[idx2])
        ax[idx][0].set_xlim(8, 14.5)
        ax[idx][0].legend()
    
    
    
    
    
    
    
    
    
    

