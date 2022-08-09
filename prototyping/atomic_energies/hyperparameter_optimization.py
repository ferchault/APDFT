import numpy as np
import sys
sys.path.insert(0, '/home/sahre/git_repositories/APDFT/prototyping/atomic_energies')
import qml_interface as qmi

# newton optimize sigma
def newton_iteration(label_coeff, label_sigma, lam_val, rep_coeff, rep_sigma, sigma):
    
    # error in sigma
    coeffs = qmi.train_kernel(rep_coeff, label_coeff, sigma, lam_val)
    # predict label of sigma_set
    sigma_error = np.abs(qmi.predict_labels(rep_sigma, rep_coeff, sigma, coeffs)[0]-label_sigma)

    # calculate derivative of error with with respect to sigma
    dx = 0.01
    # lower 
    coeffs_l = qmi.train_kernel(rep_coeff, label_coeff, sigma-dx, lam_val)
    # predict label of sigma_set
    sigma_error_l = np.abs(qmi.predict_labels(rep_sigma, rep_coeff, sigma-dx, coeffs_l)[0]-label_sigma)
    # higher 
    coeffs_h = qmi.train_kernel(rep_coeff, label_coeff, sigma+dx, lam_val)
    # predict label of sigma_set
    sigma_error_h = np.abs(qmi.predict_labels(rep_sigma, rep_coeff, sigma+dx, coeffs_h)[0]-label_sigma)

    sigma_prime = (sigma_error_h-sigma_error_l)/(2*dx) # derivative of error with respect to sigma
    
    # calculate new sigma
    sigma_new = sigma - sigma_error/sigma_prime
    # calculate new error
    # error in sigma
    coeffs_new = qmi.train_kernel(rep_coeff, label_coeff, sigma_new, lam_val)
    # predict label of sigma_set
    sigma_error_new = np.abs(qmi.predict_labels(rep_sigma, rep_coeff, sigma_new, coeffs)[0]-label_sigma)
    
    return(coeffs_new, sigma_error_new, sigma_new)

# newton optimize sigma
def sd_iteration(label_coeff, label_sigma, lam_val, rep_coeff, rep_sigma, sigma, step):
    """
    steepest descent
    """
    
    # error in sigma
    coeffs = qmi.train_kernel(rep_coeff, label_coeff, sigma, lam_val)
    # predict label of sigma_set
    sigma_error = np.abs(qmi.predict_labels(rep_sigma, rep_coeff, sigma, coeffs)[0]-label_sigma)

    # calculate derivative of error with with respect to sigma
    dx = 0.01
    # lower 
    coeffs_l = qmi.train_kernel(rep_coeff, label_coeff, sigma-dx, lam_val)
    # predict label of sigma_set
    sigma_error_l = np.abs(qmi.predict_labels(rep_sigma, rep_coeff, sigma-dx, coeffs_l)[0]-label_sigma)
    # higher 
    coeffs_h = qmi.train_kernel(rep_coeff, label_coeff, sigma+dx, lam_val)
    # predict label of sigma_set
    sigma_error_h = np.abs(qmi.predict_labels(rep_sigma, rep_coeff, sigma+dx, coeffs_h)[0]-label_sigma)

    sigma_prime = (sigma_error_h-sigma_error_l)/(2*dx) # derivative of error with respect to sigma
    
    # calculate new sigma
    
    sigma_new = sigma - sigma_prime*step
    # calculate new error
    # error in sigma
    coeffs_new = qmi.train_kernel(rep_coeff, label_coeff, sigma_new, lam_val)
    # predict label of sigma_set
    sigma_error_new = np.abs(qmi.predict_labels(rep_sigma, rep_coeff, sigma_new, coeffs)[0]-label_sigma)
    
    return(coeffs_new, sigma_error_new, sigma_new)

def sigma_grid_search(label_coeff, label_sigma, lam_val, rep_coeff, rep_sigma, sigma_grid, kernel_type = "gaussian"):
    # optimize sigma    
    all_errors = []
    all_coeffs = []
    for sigma in sigma_grid:
        # error in sigma
        coeffs = qmi.train_kernel(rep_coeff, label_coeff, sigma, lam_val, kernel_type)
        all_coeffs.append(coeffs)
        # predict label of sigma_set
        sigma_error = np.abs(qmi.predict_labels(rep_sigma, rep_coeff, sigma, coeffs)[0]-label_sigma)
        all_errors.append(sigma_error)
        
    all_errors = np.array(all_errors)
#     opt_error = np.amin(all_errors)
#     opt_ind = np.where(all_errors == opt_error)[0][0]
    
#     opt_coeffs = all_coeffs[opt_ind]
#     opt_sigma = sigma_grid[opt_ind]
    
    return(all_coeffs, all_errors)

def grid_search(label_coeff, label_sigma, lambda_grid, rep_coeff, rep_sigma, sigma_grid):
    grid_storage =  {'coeffs':[], 'lambda':[], 'sigma':[], 'validation_error':[]}

    
    for lam_val in lambda_grid:
        for sigma in sigma_grid:
            
            coeffs = qmi.train_kernel(rep_coeff, label_coeff, sigma, lam_val) # get coeffs
            prediction = qmi.predict_labels(rep_sigma, rep_coeff, sigma, coeffs)[0] # predict on validation set
            validation_error = np.abs(prediction-label_sigma)

            grid_storage['coeffs'].append(coeffs)
            grid_storage['lambda'].append(lam_val)
            grid_storage['sigma'].append(sigma)
            grid_storage['validation_error'].append(validation_error)
        
    val_errors = np.array(grid_storage['validation_error'])
    opt_error = np.amin(val_errors)
    opt_ind = np.where(val_errors == opt_error)[0][0]
    
    opt_coeffs = grid_storage['coeffs'][opt_ind]
    opt_sigma = grid_storage['sigma'][opt_ind]
    opt_lambda = grid_storage['lambda'][opt_ind]
    
    return(opt_coeffs, opt_lambda, opt_sigma, opt_error)