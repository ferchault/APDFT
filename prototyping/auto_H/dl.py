# this code is for calculating different lambda without finite difference method.
import finitediff
from finitediff import get_weights
import plotdensity as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial as scs
import scipy.integrate
from scipy.interpolate import CubicSpline
import scipy.spatial.transform as sst
import functools
import copy
from pathlib import Path
import os.path
import shutil
from ase.units import Bohr

def get_sigma(number_of_sigma,PP_filename):
    try:
        f = open(PP_filename)
    except IOError:
        print("PP file do not exist")
    sigma = []
    with open(PP_filename,"r") as f:
        content = f.readlines()
        index_1 = [x for x in range(len(content)) if "ZV" in content[x]][0]
        sigma.append(content[index_1].split()[2])
    with open(PP_filename,"r") as f:
        content = f.readlines()
        index_2 = [x for x in range(len(content)) if "RC" in content[x]][0]
        sigma.append(content[index_2].split()[0])
    with open(PP_filename,"r") as f:
        content = f.readlines()
        index_3 = [x for x in range(len(content)) if "#C" in content[x]][0]
        sigma.append(content[index_3].split()[1])
        sigma.append(content[index_3].split()[2])
    if len(sigma) != number_of_sigma:
        raise ValueError("Number of sigma is not right.")
    f.close()

    return sigma

def get_profile(whole_cube_file):
    file = pl.CUBE(whole_cube_file)
    coordinate = np.linspace(file.origin[0], file.X[0]*file.NX*Bohr, file.NX)
    return coordinate, file.project((0,1))

def generate_input(case,ueg_point,point,no_of_sig,PP_filename,approx_zero):
    with open(PP_filename,"r") as f:
        tmp = f.read()
    with open(PP_filename,"r") as f:
        tmp2 = f.readlines()
    if case == 'ueg':
        with open(PP_filename,"r") as f:
            for cnt, line in enumerate(f):
                if len(line.split()) == 3 and line.split()[0] == "ZV":
                    tmp = tmp.replace(tmp2[cnt],"  ZV =   "+str(ueg_point[0]) +"\n" )
                if len(line.split()) == 2 and line.split()[1] == "RC":
                    tmp = tmp.replace(line.split()[0],str(ueg_point[1]))
                if len(line.split()) >= 4 and line.split()[3] == "#C":
                    tmp = tmp.replace(line.split()[1],str(ueg_point[2]))
                    tmp = tmp.replace(line.split()[2],str(ueg_point[3]))
        with open(PP_filename+'_at_lambda_{}'.format(approx_zero),"w") as f:
            f.write(tmp)

        if os.path.isfile('./lambda/{}/'.format(approx_zero)+PP_filename):
            pass
        else:
            print("PP file at lambda %s do not exist, created." % (approx_zero))
            src = './lambda/{}/'.format(approx_zero)
            os.makedirs(src, exist_ok = True)
            shutil.copy('sub.job',src)
            shutil.copy('./'+PP_filename+'_at_lambda_{}'.format(approx_zero), src+PP_filename)
            run_file = 'run.inp'
            restart_file = 'RESTART.1'
            latest_file = 'LATEST'
            if approx_zero >= 0.5 :
                shutil.copy('big_run.inp',src+run_file)
                shutil.copy('big_RESTART.1',src+restart_file)
                shutil.copy('big_LATEST',src+latest_file)
            else:
                shutil.copy('small_run.inp',src+run_file)
                shutil.copy('small_RESTART.1',src+restart_file)
                shutil.copy('small_LATEST',src+latest_file)

def has_output(case,no_of_sig,approx_zero):
    if case == "ueg":
        return os.path.isfile('./lambda/%s/result.txt.cube' %(approx_zero))


def get_density(case,PP_filename,ueg_point,point,no_of_sig,approx_zero): 
    # first three parameters used for making new folders to distinguish different calculations
    generate_input(case,ueg_point,point,no_of_sig,PP_filename,approx_zero)
    
    if has_output(case,no_of_sig,approx_zero):
        pass
    else:
        print ("Cubefile at lambda {} is missing. Filename should be 'result.txt.cube'".format(approx_zero))
        
    if case == 'ueg':
        result_file = Path('./lambda/%s/result.txt.cube' %(approx_zero))
        if result_file.is_file():  
            with open('./lambda/%s/result.txt.cube' %(approx_zero), "r") as f:
                for i in range(7): f.readline()
                cubefile = np.array([[np.float64(i) for i in u.split()] for u in f.readlines()])
        else:
            cubefile = np.array([])

    return cubefile

        
def profile_first_order(approx_zero,number_of_sigma,PP_filename):
    sigma   = np.array(get_sigma(number_of_sigma,PP_filename),dtype=np.float32)
    zero    = sigma*approx_zero
        
    ueg = get_density("ueg",PP_filename,zero,zero[0],0,approx_zero) 
    
    try:
        whole_cube_file = 1.0 * ueg 
    except:
        return [0,1]
    whole_cube_file = 1.0 * ueg  
    
    headcontent = []
    
    generate_file_or_calc = Path('./lambda/{}/result.txt.cube'.format(approx_zero)) 
    if generate_file_or_calc.is_file():
        with open('./lambda/%s/result.txt.cube' %(approx_zero), "r") as f:
            for i in range(7): 
                headcontent.append(f.readline())
            head="".join(headcontent)[:-1]    
        np.savetxt('whole_cube_file_lambda_{}.txt'.format(approx_zero), whole_cube_file, header=head, delimiter=' ', comments='') 
        return get_profile('whole_cube_file_lambda_{}.txt'.format(approx_zero))
    else:
        return_null = [0.0, 1.0]
        return return_null

def save_E(approx_zero):
    generate_file_or_calc = Path('./lambda/{}/out.log'.format(approx_zero))
    if generate_file_or_calc.is_file():
        with open('./lambda/%s/out.log' %(approx_zero), "r") as f:
            total_E = []
            for line in f:
                if len(line.split()) > 4 and line.split()[0] == '(K+E1+L+N+X)':
                    total_E = np.array([np.float(approx_zero),np.float(line.split()[4])])
    else:
        print ("out.log file at lambda {} is missing. Please generate it.".format(approx_zero))
        return

    with open('total_energy.dat', 'a') as f:
        np.savetxt(f,total_E, fmt='%2.9f', newline=' ')
        f.write('\n')

number_of_sigma = 4
PP_filename = "H_SG_LDA"
fig, ax = plt.subplots(1,1)
approx_zero = 0.0
Number_of_lambda = 10
for approx_zero in np.arange(0.0,1.0+1.0/Number_of_lambda,1.0/Number_of_lambda):   
    approx_zero = np.around(approx_zero,5)
    plt.plot(profile_first_order(approx_zero,number_of_sigma,PP_filename),label = "lambda is {}".format(approx_zero))
    save_E(approx_zero)
