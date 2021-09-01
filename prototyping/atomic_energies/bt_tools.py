import numpy as np
from scipy.optimize import curve_fit

from ase.units import Bohr
import ase.io as aio
import glob

class BDE():
    def __init__(self, energies, nuc_charges):
        ha2kcal = 630
        # define energies
        self.bde = -energies[:,0]*ha2kcal
        self.bfe = energies[:,0]*ha2kcal
        self.ebfe = energies[:,1]*ha2kcal
        self.nbfe = energies[:,2]*ha2kcal
        
        # define charge combinations Z1, Z2
        self.nuc_charges = nuc_charges
#         if row == 2:
#             self.nuc_charges = [6, 7, 8, 9]
#         elif row == 3:
#             self.nuc_charges = [14, 15, 16, 17]
#         elif row == 4:
#             self.nuc_charges = [32, 33, 34, 35]
        self.init_xdata(self.nuc_charges)
        
    def fit(self, model, p0 = None):
        if p0 is not None:
            self.params = curve_fit(model, self.xdata, self.ebfe, p0, maxfev=100000)
        else:
            self.params = curve_fit(model, self.xdata, self.ebfe)
        
        self.ebfe_fitted = model(self.xdata, *self.params[0])
        self.bde_fitted = -(self.ebfe_fitted + self.nbfe)
        self.mae = np.abs(self.bde_fitted-self.bde).mean()
    
    def init_xdata(self, nuc_charges):
        self.xdata = []
        for i in range(len(nuc_charges)):
            for j in range(i, len(nuc_charges)):
                self.xdata.append([nuc_charges[i], nuc_charges[j]])
        self.xdata = np.array(self.xdata)
    
    def get_coeff_mat(self):
        num_eq = len(self.ebfe)
        num_coeffs = 2*len(self.nuc_charges)
        self.coeff_mat = np.zeros((num_eq, num_coeffs))
        
        ind_a = []
        for i in range(len(self.nuc_charges)):
            for j in range(i, len(self.nuc_charges)):
                ind_a.append([i, j])
        
        for i, z, a in zip(range(num_eq), self.xdata, ind_a):
            # set a
            self.coeff_mat[i][a[0]] += 1
            self.coeff_mat[i][a[1]] += 1
            # set b, the index can be calculated from the indices for a
            b = [a[0]+len(self.nuc_charges), a[1]+len(self.nuc_charges)]
            self.coeff_mat[i][b[0]] += z[1]
            self.coeff_mat[i][b[1]] += z[0]
        return(self.coeff_mat)

    def linear_fit(self):
        self.linear_params, mae_waste, self.linear_rank, self.linear_sg = np.linalg.lstsq(self.get_coeff_mat(), self.ebfe, rcond=None)
        self.ebfe_predicted = self.linear_params@self.coeff_mat.T
        self.bfe_predicted = self.ebfe_predicted + self.nbfe
        self.bde_predicted = -self.bfe_predicted
        self.linear_mae = np.abs(self.bde_predicted - self.bde).mean()
        

class BDE_set(BDE):
    def __init__(self, energies, nuc_charges_set):
        ha2kcal = 630
        # define energies
        self.bde = -energies[:,0]*ha2kcal
        self.bfe = energies[:,0]*ha2kcal
        self.ebfe = energies[:,1]*ha2kcal
        self.nbfe = energies[:,2]*ha2kcal

        # define charge combinations Z1, Z2
        self.nuc_charges_set = nuc_charges_set

        self.init_xdata(self.nuc_charges_set)

    
    def init_xdata(self, nuc_charges_set):
        self.xdata = []
        for nuc_charges in nuc_charges_set:
            for i in range(len(nuc_charges)):
                for j in range(i, len(nuc_charges)):
                    self.xdata.append([nuc_charges[i], nuc_charges[j]])
        self.xdata = np.array(self.xdata)
        
class BDE_dist(BDE):
    def __init__(self, energies, nuc_charges_row, dist):
        ha2kcal = 630
        # define energies
        self.bde = -energies[:,0]*ha2kcal
        self.bfe = energies[:,0]*ha2kcal
        self.ebfe = energies[:,1]*ha2kcal
        self.nbfe = energies[:,2]*ha2kcal
        
        # define charge combinations Z1, Z2
        self.nuc_charges_row = nuc_charges_row
        self.distances = dist
        self.init_xdata(self.nuc_charges_row, self.distances)
        
    def init_xdata(self, nuc_charges_row, distances):
        self.xdata = []
        for nuc_charges in nuc_charges_row:
            for i in range(len(nuc_charges)):
                for j in range(i, len(nuc_charges)):
                    self.xdata.append([nuc_charges[i], nuc_charges[j]])
        self.xdata = np.array(self.xdata)
        self.xdata = np.concatenate((self.xdata, distances.T), axis = 1)
        
    def fit(self, model, p0 = None):
        if p0 is not None:
            self.params = curve_fit(model, self.xdata, self.ebfe, p0, maxfev=100000)
        else:
            self.params = curve_fit(model, self.xdata, self.ebfe)
        
        #self.params = curve_fit(model, self.xdata, self.ebfe, maxfev=100000)
#         self.ebfe_fitted = model(self.xdata, self.params[0][0], self.params[0][1], self.params[0][2], self.params[0][3], self.params[0][4], self.params[0][5])
        self.ebfe_fitted = model(self.xdata, *self.params[0])
        self.bde_fitted = -(self.ebfe_fitted + self.nbfe)
        self.mae = np.abs(self.bde_fitted-self.bde).mean()

        
class BDE_zeff(BDE_dist):
    def __init__(self, energies, nuc_charge_set, zeff_set, dist):
        ha2kcal = 630
        # define energies
        self.bde = -energies[:,0]*ha2kcal
        self.bfe = energies[:,0]*ha2kcal
        self.ebfe = energies[:,1]*ha2kcal
        self.nbfe = energies[:,2]*ha2kcal
        
        # define charge combinations Z1, Z2
        self.nuc_charges_set = nuc_charge_set
        self.zeff_set = zeff_set
        self.distances = dist
        self.init_xdata(self.nuc_charges_set, self.zeff_set, self.distances)
        
    def init_xdata(self, nuc_charge_set, zeff_set, distances):
        self.xdata = []
        for nuc_charges, zeff, d in zip(nuc_charge_set, zeff_set, distances):
            for i in range(len(nuc_charges)):
                for j in range(i, len(nuc_charges)):
                    self.xdata.append([nuc_charges[i], nuc_charges[j], zeff[i], zeff[j], d])
        self.xdata = np.array(self.xdata)
        
        
def get_distances(full_path, sorting_key = None):
    structures = glob.glob(full_path)
    if sorting_key:
        structures.sort(key=sorting_key)
    
    dist = []
    for s in structures:
        mol = aio.read(s)
        dist.append(mol.get_distance(0, 1))
    dist = np.array(dist)
    return(dist)