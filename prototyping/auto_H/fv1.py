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

def generate_input(delta,case,ueg_point,point,no_of_sig,PP_filename,delta_i,approx_zero):
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
        with open(PP_filename+'_ueg_at_zero_point_{}'.format(approx_zero),"w") as f:
            f.write(tmp)

        if os.path.isfile('./ueg/{}/'.format(approx_zero)+PP_filename):
            pass
        else:
            print("ueg PP file at %s do not exist, created." % (approx_zero))
            src = './ueg/{}/'.format(approx_zero)
            os.makedirs(src, exist_ok = True)
            shutil.copy2('run.inp',src)
            shutil.copy2('sub.job',src)
            shutil.copy2('RESTART.1',src)
            shutil.copy2('LATEST',src)
            shutil.copy2('./'+PP_filename+'_ueg_at_zero_point_%s' %(approx_zero), src+PP_filename)

    elif case == '1order':
        with open(PP_filename,"r") as f:
            for cnt, line in enumerate(f):
                if no_of_sig == 0:
                    if len(line.split()) == 3 and line.split()[0] == "ZV":
                        tmp = tmp.replace(tmp2[cnt],"  ZV =   "+str(point) +"\n" )
                    if len(line.split()) == 2 and line.split()[1] == "RC":
                        tmp = tmp.replace(line.split()[0],str(ueg_point[1]))
                    if len(line.split()) >= 4 and line.split()[3] == "#C":
                        tmp = tmp.replace(line.split()[1],str(ueg_point[2]))
                        tmp = tmp.replace(line.split()[2],str(ueg_point[3]))
                if no_of_sig == 1:
                    if len(line.split()) == 3 and line.split()[0] == "ZV":
                        tmp = tmp.replace(tmp2[cnt],"  ZV =   "+str(ueg_point[0]) +"\n" )
                    if len(line.split()) == 2 and line.split()[1] == "RC":
                        tmp = tmp.replace(line.split()[0],str(point))
                    if len(line.split()) >= 4 and line.split()[3] == "#C":
                        tmp = tmp.replace(line.split()[1],str(ueg_point[2]))
                        tmp = tmp.replace(line.split()[2],str(ueg_point[3]))
                if no_of_sig == 2:
                    if len(line.split()) == 3 and line.split()[0] == "ZV":
                        tmp = tmp.replace(tmp2[cnt],"  ZV =   "+str(ueg_point[0]) +"\n" )
                    if len(line.split()) == 2 and line.split()[1] == "RC":
                        tmp = tmp.replace(line.split()[0],str(ueg_point[1]))
                    if len(line.split()) >= 4 and line.split()[3] == "#C":
                        tmp = tmp.replace(line.split()[1],str(point))
                        tmp = tmp.replace(line.split()[2],str(ueg_point[3]))
                if no_of_sig == 3:
                    if len(line.split()) == 3 and line.split()[0] == "ZV":
                        tmp = tmp.replace(tmp2[cnt],"  ZV =   "+str(ueg_point[0]) +"\n" )
                    if len(line.split()) == 2 and line.split()[1] == "RC":
                        tmp = tmp.replace(line.split()[0],str(ueg_point[1]))
                    if len(line.split()) >= 4 and line.split()[3] == "#C":
                        tmp = tmp.replace(line.split()[1],str(ueg_point[2]))
                        tmp = tmp.replace(line.split()[2],str(point))

        with open(PP_filename+'_{}_at_{}_ueg_at_{}_point_{}_sigma_{}'.format(case,delta,approx_zero,delta_i,no_of_sig,approx_zero), 'w') as f:
            f.write(tmp)

        if os.path.isfile('./{}/delta_{}/ueg_{}/{}/sigma_{}/'.format(case,delta,approx_zero,delta_i,no_of_sig)+PP_filename):
            pass
        else:
            print("{} PP file of delta {} at No. {} with No. {} sigma and ueg at {} do not exist, created.".format(case,delta,delta_i,no_of_sig, approx_zero))
            src = './{}/delta_{}/ueg_{}/{}/sigma_{}/'.format(case,delta,approx_zero,delta_i,no_of_sig)
            os.makedirs(src, exist_ok = True)
            shutil.copy2('run.inp',src)
            shutil.copy2('RESTART.1',src)
            shutil.copy2('sub.job',src)
            shutil.copy2('LATEST',src)
            shutil.copy2('./'+PP_filename+'_{}_at_{}_ueg_at_{}_point_{}_sigma_{}'.format(case,delta,approx_zero,delta_i,no_of_sig), src+PP_filename)
    else:
        print ("Do not support this order currently.")
        exit()

def has_output(delta,case,no_of_sig,delta_i,approx_zero):
    if case == "ueg":
        return os.path.isfile('./ueg/%s/result.txt.cube' %(approx_zero))
    if case == "1order":
        return os.path.isfile('./{}/delta_{}/ueg_{}/{}/sigma_{}/result.txt.cube'.format(case,delta,approx_zero,delta_i,no_of_sig))


def get_density(delta,case,PP_filename,ueg_point,point,no_of_sig,delta_i,approx_zero): 
    # first three parameters used for making new folders to distinguish different calculations
    generate_input(delta,case,ueg_point,point,no_of_sig,PP_filename,delta_i,approx_zero)
    
    if has_output(delta,case,no_of_sig,delta_i,approx_zero):
        pass
    else:
        print ("{} calculation result at {} with No. {} sigma and ueg at {} is missing. Filename should be 'result.txt.cube'".format(case,delta,no_of_sig, approx_zero))
        
    if case == 'ueg':
        result_file = Path('./ueg/%s/result.txt.cube' %(approx_zero))
        if result_file.is_file():  
            with open('./ueg/%s/result.txt.cube' %(approx_zero), "r") as f:
                for i in range(7): f.readline()
                cubefile = np.array([[np.float64(i) for i in u.split()] for u in f.readlines()])
        else:
            cubefile = np.array([])
    if case == '1order':
        result_file = Path('./{}/delta_{}/ueg_{}/{}/sigma_{}/result.txt.cube'.format(case,delta,approx_zero,delta_i,no_of_sig))
        if result_file.is_file():  
            with open('./{}/delta_{}/ueg_{}/{}/sigma_{}/result.txt.cube'.format(case,delta,approx_zero,delta_i,no_of_sig), "r") as f:
                for i in range(7): f.readline()
                cubefile = np.array([[np.float64(i) for i in u.split()] for u in f.readlines()])
        else:
            cubefile = np.array([])

    return cubefile

        
def profile_first_order(delta,approx_zero,number_of_sigma,PP_filename):
    sigma   = np.array(get_sigma(number_of_sigma,PP_filename),dtype=np.float32)
    zero    = sigma*approx_zero
    delta_1 = [i*delta   for i in sigma]
    delta_2 = [i*delta*2 for i in sigma]
        
    weights = finitediff.get_weights(np.array([approx_zero, 1, 2]), 0, maxorder=1)
    weight_ueg_approx_zero,    weight_ueg_one,    weight_ueg_two    = weights[:,0]
    weight_1order_approx_zero, weight_1order_one, weight_1order_two = weights[:,1]
    
    ueg = get_density(delta,"ueg",PP_filename,zero,zero[0],0,0.0,approx_zero) 
    dct = {}
    for no_of_sig in (range(number_of_sigma)):
        for pos_of_sig in (1,2):
            tmp = eval('delta_'+'%s'  %(pos_of_sig))
            point = [aa for aa in tmp][no_of_sig]
            tmp2 = get_density(delta,"1order",PP_filename,zero,point,no_of_sig,'delta_'+'{}'.format(pos_of_sig),approx_zero )
            dct["s{}{}".format(no_of_sig,pos_of_sig )] = tmp2
    
    try:
        whole_cube_file = 2.0 * ueg + 2.0 * dct['s11']
    except:
        return [0,1]
    whole_cube_file = weight_ueg_approx_zero * ueg  

    for num in range(number_of_sigma):
        whole_cube_file +=  \
        weight_ueg_one * dct['s{}1'.format(num)] + weight_ueg_two * dct['s{}2'.format(num)] + (0.5) * (weight_1order_approx_zero * ueg + weight_1order_one * dct['s{}1'.format(num)] + weight_1order_two * dct['s{}2'.format(num)])/(delta_1[num])*(sigma[num])

    headcontent = []
    with open('./ueg/%s/result.txt.cube' %(approx_zero), "r") as f:
        for i in range(7): 
            headcontent.append(f.readline())
        head="".join(headcontent)[:-1]    
    np.savetxt('whole_cube_file_delta_{}_approx_zero_{}.txt'.format(delta,approx_zero), whole_cube_file, header=head, delimiter=' ', comments='')    
    return get_profile('whole_cube_file.txt')

number_of_sigma = 4
PP_filename = "H_SG_LDA"
fig, ax = plt.subplots(1,1)
for approx_zero in (0.001, 0.01):
    for delta in (0.03, 0.05, 0.1):   
        plt.plot(profile_first_order(delta,approx_zero,number_of_sigma,PP_filename),label = "approx_zero is %f, delta is %f"  % (approx_zero, delta) )
plt.title("Plots of different approx_zero and delta. " )
ax.set_xlabel(r'Cell coordinate $x_0$ (Ang)')
ax.set_ylabel(r'$\rho (x_0)$')
ax.legend()
plt.show()
