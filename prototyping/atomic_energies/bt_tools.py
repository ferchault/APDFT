import numpy as np
from scipy.optimize import curve_fit

from ase.units import Bohr
import ase.io as aio
import glob

class BDE():
    def __init__(self, energies, nuc_charges, unit = 'kcal'):
        if unit == 'kcal':
            ha2kcal = 630
        elif unit == 'atomic':
            ha2kcal = 1
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
        self.ebfe_fitted = self.linear_params@self.coeff_mat.T
        self.bfe_fitted = self.ebfe_fitted + self.nbfe
        self.bde_fitted = -self.bfe_fitted
        self.linear_mae = np.abs(self.bde_fitted - self.bde).mean()
        
    def train(self, tr_ind, model, p0 = None):
        if p0 is not None:
            self.params_tr = curve_fit(model, self.xdata[tr_ind], self.ebfe[tr_ind], p0, maxfev=100000)
        else:
            self.params_tr = curve_fit(model, self.xdata[tr_ind], self.ebfe[tr_ind])

        self.ebfe_fitted_tr = model(self.xdata[tr_ind], *self.params_tr[0])
        self.bde_fitted_tr = -(self.ebfe_fitted_tr + self.nbfe[tr_ind])
        self.mae_tr = np.abs(self.bde_fitted_tr-self.bde[tr_ind]).mean()

    def predict(self, test_ind, model):
        self.ebfe_fitted_test = model(self.xdata[test_ind], *self.params_tr[0])
        self.bde_fitted_test = -(self.ebfe_fitted_test + self.nbfe[test_ind])
        self.mae_test = np.abs(self.bde_fitted_test-self.bde[test_ind]).mean()

        
class BDE_clean(BDE):
    def __init__(self,bfe, bde, ebfe, nbfe, Z1, Z2, xdata):
        self.bfe = bfe
        self.bde = bde
        self.ebfe = ebfe
        self.nbfe = nbfe
        self.Z1 = Z1
        self.Z2 = Z2
        self.xdata = xdata
        
    @classmethod
    def fromdict(cls, datadict):
        # get units
        if 'unit' in datadict.keys():
            e_unit = datadict['unit']
        else:
            e_unit = 1 # use Hartree as energy unit
            
        # bond energy
#         if datadict['bfe type'] == 'isodesmic':
#             bfe = np.array(datadict['bfe isodesmic'])*e_unit
#         else:
        bfe = np.array(datadict['bfe'])*e_unit # homolytic set to default
        bde = -bfe
        
        # nuclear charges
        Z1 = np.array(datadict['Z1'])
        Z2 = np.array(datadict['Z2'])
        
        # nuclear repulsion
        if 'nbfe' in datadict.keys():
            nbfe = np.array(datadict['nbfe'])*e_unit
        else:
            nbfe = Z1*Z2*e_unit
            
        # electronic bond energy
        ebfe = bfe - nbfe
        
        # xdata
        xdata = np.array([Z1, Z2]).T
        return(cls(bfe, bde, ebfe, nbfe, Z1, Z2, xdata))
    
    def get_coeff_mat(self):
        """
        matrix with coefficients for linear fit
        """
        
        unique_Z = np.sort(np.unique(np.concatenate((self.Z1, self.Z2))))
        index_a = dict(zip(unique_Z, np.arange(len(unique_Z))))
        index_b = dict(zip(unique_Z, np.arange(len(unique_Z), 2*len(unique_Z))))
        coeff_mat = np.zeros((len(self.Z1), 2*len(unique_Z))) # there are two parameters for every element
        
        for row, Z1, Z2 in zip(range(len(self.Z1)), self.Z1, self.Z2):
            # coefficients for Z1
            coeff_mat[row, index_a[Z1]] += 1
            coeff_mat[row, index_b[Z1]] += Z2
            # coefficients for Z2
            coeff_mat[row, index_a[Z2]] += 1
            coeff_mat[row, index_b[Z2]] += Z1
            
        self.coeff_mat = coeff_mat
        return(self.coeff_mat)
    

class BDE_set(BDE):
    def __init__(self, energies, nuc_charges_set, unit = 'kcal'):
        if unit == 'kcal':
            ha2kcal = 630
        elif unit == 'atomic':
            ha2kcal = 1
            
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