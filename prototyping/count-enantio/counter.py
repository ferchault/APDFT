from pyscf import gto, scf
from pyscf.symm.geom import detect_symm #change TOLERANCE to higher values
import numpy as np

elements = {'Ghost':0, 'H':1, 'He':2,
'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18,
'K':19, 'Ca':20, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,}

inv_elements = {v: k for k, v in elements.items()}

tolerance = 2 #Rounding to ... digits

def delta(i,j):
    #Kronecker Delta
    if i == j:
        return 1
    else:
        return 0

def center_mole(mole):
    #Centers a molecule
    sum = (0,0,0)
    N = len(mole)
    for i in range(N):
        sum = np.add(mole[i][1],sum)
    sum = np.multiply(sum, 1/N)
    for i in range(N):
        mole[i][1] = np.subtract(mole[i][1],sum)

def Coulomb_matrix(mole):
    #returns the Coulomb matrix of a given molecule
    N = len(mole)
    result = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if (j == i):
                result[i][i] = pow(elements[mole[i][0]], 2.4)
            else:
                result[i][j] = elements[mole[i][0]]*elements[mole[j][0]]/np.linalg.norm(np.subtract(mole[i][1],mole[j][1]))
    return result

def Coulomb_neighborhood(mole):
    '''returns the sum over rows/columns of the Coulomb matrix.
    Thus, each atom is assigned its Coulombic neighborhood'''
    matrix = Coulomb_matrix(mole)
    return matrix.sum(axis = 0)

def CN_inertia_tensor(mole):
    #Calculate an inertia tensor but with Coulomb_neighborhood instead of masses
    N = len(mole)
    CN = Coulomb_neighborhood(mole)
    result_tensor = np.zeros((3,3))
    sum = 0
    for i in range(3):
        for j in range(i+1):
            for k in range(N):
                sum += CN[k]*(np.dot(mole[k][1], mole[k][1])*delta(i,j) - mole[k][1][i]*mole[k][1][j])
            result_tensor[i][j] = sum
            result_tensor[j][i] = sum
            sum = 0
    return result_tensor

def CN_inertia_moment(mole):
    #Calculate the inertia moments of a molecule with charges instead of masses
    #and sort them in ascending order
    w,v = np.linalg.eig(CN_inertia_tensor(mole))
    #Only the eigen values are needed, v is discarded
    moments = np.sort(w)
    #To make life easier, the values in moments are rounded to tolerance for easier comparison
    for i in range(3):
        moments[i] = round(moments[i],tolerance)
    return moments

def num_AEchildren(mole, m = 2, dZ = [+1,-1]):
    '''Returns the number of alchemical enantiomers of mole that can be reached by
    varying m atoms in mole with identical Coulombic neighborhood by the values
    specified in dZ'''
    N = len(mole)
    z = len(dZ)
    if N < m:
        raise ValueError("Number of changing atoms must not exceed number of atoms in molecule")
    if z < 1:
        raise ValueError("Number of elements of dZ must be at least 1")
    if 0 in dZ:
        raise ValueError("0 is not allowed in dZ")
    #Prepare the array mole^
    center_mole(mole)
    mole = np.array(mole, dtype=object)
    #Find the Coulombic neighborhood of each atom in mole
    CN = Coulomb_neighborhood(mole)
    '''To make life easier, the values in CN are rounded to tolerance for easier comparison'''
    for i in range(N):
        CN[i] = round(CN[i],tolerance)
    '''Are there atoms with identical/close Coulombic neighborhood? Find them and store
    their indices'''
    similars = [np.where(CN == i)[0].tolist() for i in np.unique(CN)]
    '''This is the list of all atoms which can be transmuted simultaneously.
    Now, count all AEs which are possible'''
    count = 0
    new_m = m
    #Initalize empty array temp_mole for all configurations. No idea how to that otherwise?
    temp_mole = np.array([['XXXXX', (1,2,3)]], dtype=object)
    temp_mole = np.delete(temp_mole, 0, 0)
    for alpha in range(len(similars)):
        num_sites = len(similars[alpha])
        #Get necessary atoms listed in similars[alpha]
        for i in range(num_sites):
            temp_mole = np.append(temp_mole, [mole[similars[alpha][i]]], axis = 0)
        #Make sure, that m does no exceed length of similars[alpha] = num_sites
        if m > len(similars[alpha]):
            new_m = num_sites
            print("Warning: Number of changing atoms (", m, ") exceeds the number of alchemically similar sites in set", alpha, "which is", num_sites)
        else:
            new_m = m
        '''Now: go through all configurations of changing m_new atoms of set similars[alpha]
        with size num_sites by the values stored in dZ. Then: compare their CN_inertia_moments
        and only count the unique ones'''
        atomwise_config = np.zeros((num_sites, len(dZ)+1))
        standard_config = np.zeros((num_sites))
        #First: All allowed charges for ONE atom at a time
        for i in range(num_sites):
            #no change:
            atomwise_config[i][0] = elements[temp_mole[i][0]]
            standard_config[i] = atomwise_config[i][0]
            #just changes:
            for j in range(len(dZ)):
                atomwise_config[i][j+1] = elements[temp_mole[i][0]]+dZ[j]
        #Second: All possible combinations of those atoms with meshgrid
        #The * passes the arrays element wise
        mole_config = np.array(np.meshgrid(*atomwise_config.tolist())).T.reshape(-1,num_sites)
        #Third: Delete all arrays where number of changes unequal m_new
        config_num = 0
        while config_num < len(mole_config):
            if (num_sites - new_m == (np.subtract(standard_config,mole_config[config_num]) == 0).sum()):
                config_num += 1
            else:
                mole_config = np.delete(mole_config, config_num, axis = 0)
        #Now: check that atoms have not been transmuted to negative charges
        if np.min(mole_config) < 0:
            raise ValueError("Values in dZ lead to negative nuclear charges in alchemically similar site")
        #Of all those configurations, calculate CN_inertia_moments and save them
        CIM = np.zeros((config_num, 3))
        for i in range(config_num):
            for j in range(num_sites):
                temp_mole[j][0] = inv_elements[mole_config[i][j]]
            #print(mole_config[i])
            print(temp_mole)
            CIM[i] = CN_inertia_moment(temp_mole)
            print(CIM[i])
            print('---------------')
        #Every CN_inertia_moment is only to be counted once. No multiples!
        count += len(np.unique(CIM, axis = 0))
        #print(count)
        '''At this point of the algorithm, the np.unique class could be easily used
        to return the unique configurations, but of the ENTIRE molecule, not just
        the similars, by saving all temp_mole in another array and deleting the non-
        unique according to CIM'''
        #Clear temp_mole
        temp_mole = np.array([['XXXXX', (1,2,3)]], dtype=object)
        temp_mole = np.delete(temp_mole, 0, 0)
    return count




benzene = [['C', (0,0,1)], ['C', (0,0.866025,0.5)], ['C', (0,0.866025,-0.5)],
['C', (0,0,-1)], ['C', (0,-0.866025,-0.5)], ['C', (0,-0.866025,0.5)]]

cube = [['Al', (0,0,0)], ['Al', (1,0,0)], ['Al', (0,1,0)], ['Al', (0,0,1)],
['Al', (1,1,0)], ['Al', (1,0,1)], ['Al', (0,1,1)], ['Al', (1,1,1)]]

print(num_AEchildren(cube, m=4, dZ=[+1,-1]))
