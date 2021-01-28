import numpy as np
import qml
import sys
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies/')
import qml_interface as qmi
import sklearn.model_selection as sk
import pickle
import os

def crossvalidate(reps, labels, tr_size, sigma, lam, num_cv):
    errors = []
    for cv in range(num_cv):
        reps_tr, reps_test, labels_tr, labels_test = sk.train_test_split(reps,labels,train_size=tr_size)
        coeffs = qmi.train_kernel(reps_tr, labels_tr, sigma, lam_val)
        labels_predicted = qmi.predict_labels(reps_test, reps_tr, sigma, coeffs)
        errors.append((np.abs(labels_predicted - labels_test)).mean())
    errors = np.array(errors)
    return(errors.mean(), errors.std())

def save_obj(obj, fname ):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
def get_tr_size(data_size):
    """
    largest number of training points is roughly 90% of complete data (largest multiple of 2 that is <= 90%)
    """
    largest_set = int(np.log2(data_size*0.9))
    tr_size = np.logspace(0, largest_set, largest_set+1, base=2).astype(int)
    return(tr_size)

def get_element_symbol(Z):
    if int(Z) == 1:
        return('H')
    elif int(Z) == 6:
        return('C')
    elif int(Z) == 7:
        return('N')
    elif int(Z) == 8:
        return('O')
    elif int(Z) == 9:
        return('F')
    else:
        raise ValueError('Symbol for given charge not available')

#######################################################################################################
# CHECK BEFORE RUNNING 
#######################################################################################################

# property which will be predicted
PROPERTY_TO_LEARN = 'atomisation'
# hyperparameter values
sigmas = np.logspace(-1, 10, 11, base=2) # kernel width
lam_val = 1e-5 # regularizer, no list possible right now
num_cv = 10 # number crossvalidations

# path where best sigma will be saved to
path_best = f'/home/misa/projects/Atomic-Energies/data/lcurves/lcurves_atomisation/hypar_elementwise'
# path where all sigmas will be saved to
path_all = f'/home/misa/projects/Atomic-Energies/data/lcurves/lcurves_atomisation/hypar_elementwise'

#######################################################################################################

# data preparation
paths = qmi.wrapper_alch_data()
exclude='/home/misa/APDFT/prototyping/atomic_energies/results/slice_ve38/dsgdb9nsd_000829/atomic_energies_with_mic.txt'
paths.remove(exclude)
data, molecule_size = qmi.load_alchemy_data(paths)

alch_pots = qmi.generate_label_vector(data, molecule_size.sum(), value=PROPERTY_TO_LEARN)

all_local_reps = qmi.generate_atomic_representations(data, molecule_size)

# split up alchemical potential by element
charges = qmi.generate_label_vector(data, molecule_size.sum(), value='charge')
idc_by_charge = qmi.partition_idx_by_charge(charges)

el_reps =dict()
el_alch_pots = dict()
for k in idc_by_charge.keys():
    el_reps[k] = all_local_reps[idc_by_charge[k]]
    el_alch_pots[k] = alch_pots[idc_by_charge[k]]
    
for charge in el_reps.keys():
    lcurves = dict()

    # define number of training points for which MAE is calculated
    #set_sizes = np.concatenate((get_tr_size(len(el_alch_pots[charge])), np.array([int(len(el_alch_pots[charge])*0.9)])) )
    set_sizes = np.array( [ int(len(el_alch_pots[charge])*0.9) ] )
    print(set_sizes, flush=True)
    
    # special for H
#     set_sizes = np.concatenate((set_sizes, np.array([3300])))

    for sigma in sigmas:
        error_cv = []
        error_std = []
        # calculate error for every training point size
        for idx, tr_size in enumerate(set_sizes):
            err, err_std = crossvalidate(el_reps[charge], el_alch_pots[charge], tr_size, sigma, lam_val, num_cv)
            error_cv.append(err)
            error_std.append(err_std)

        lcurves[f'sig_{sigma}'] = np.array([set_sizes, error_cv, error_std]).T
        
    
    # save best learning curve
    lowest_error = (None, None)
    for k in lcurves.keys():
        if lowest_error[1]==None or lowest_error[1] > np.amin(lcurves[k][:,1]):
            lowest_error = (k, np.amin(lcurves[k][:,1]))
    save_data = lcurves[lowest_error[0]]

    # filename
    el_symbol = get_element_symbol(charge)
    path = os.path.join(path_best, f'best_{PROPERTY_TO_LEARN}_{el_symbol}_b2.txt')

    sig_val = lowest_error[0].split('_')[1]
    header = f'sigma = {sig_val}, lambda = {lam_val}, number cv = {num_cv}'
    np.savetxt(path, save_data, delimiter='\t', header=header)

    # save dictionary of learning curves at all sigmas
    fname = os.path.join(path_all, f'all_sigma_{PROPERTY_TO_LEARN}_{el_symbol}_b2.txt')
    save_obj(lcurves, fname)