import numpy as np
import time

elements = {'Ghost':0, 'H':1, 'He':2,
'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18,
'K':19, 'Ca':20, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,}

inv_elements = {v: k for k, v in elements.items()}

tolerance = 3
'''Rounding to ... digits. Do not go above 3, if you do not want
to write down the structure of your molecule up to the gazillion-th digit. Do
not go below 2, if you do not want the programm to find symmetries where there
are none.'''

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
    #Calculate the inertia moments of a molecule with CN instead of masses
    #and sort them in ascending order
    w,v = np.linalg.eig(CN_inertia_tensor(mole))
    #Only the eigen values are needed, v is discarded
    moments = np.sort(w)
    #To make life easier, the values in moments are rounded to tolerance for easier comparison
    for i in range(3):
        moments[i] = round(moments[i],tolerance)
    return moments

def array_compare(arr1, arr2):
    '''arr1 = [...]
    arr2 = [[...],[...],[...],...]
    Is there an exact copy op arr1 in arr2'''
    within = False
    for i in range(len(arr2)):
        if (arr1 == arr2[i]).all():
            within = True
    return within

def num_AEchildren(mole, m1 = 2, dZ1 = +1, m2 = 2, dZ2 = -1):
    '''Returns the number of alchemical enantiomers of mole that can be reached by
    varying m1 and m2 atoms in mole with identical Coulombic neighborhood by dZ1
    dZ2, respectively.'''
    N = len(mole)
    if N < m1 + m2:
        raise ValueError("Number of changing atoms must not exceed number of atoms in molecule")
    if (m1*dZ1 + m2*dZ2 != 0):
        raise ValueError("Netto change in charge must be 0: m1*dZ1 = -m2*dZ2")
    if (dZ1 == 0) or (dZ2 == 0):
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
    similars = np.array([np.where(CN == i)[0] for i in np.unique(CN)],dtype=object)
    #Delete all similars which include only one atom:
    num_similars = 0
    while num_similars < len(similars):
        if len(similars[num_similars])>1:
            num_similars += 1
        else:
            similars = np.delete(similars, num_similars, axis = 0)
    '''This is the list of all atoms which can be transmuted simultaneously.
    Now, count all molecules which are possible excluding mirrored or rotated versions'''
    count = 0
    #Initalize empty array temp_mole for all configurations. No idea how to do that otherwise?
    temp_mole = np.array([['XXXXX', (1,2,3)]], dtype=object)
    temp_mole = np.delete(temp_mole, 0, 0)
    for alpha in range(len(similars)):
        num_sites = len(similars[alpha])
        #Get necessary atoms listed in similars[alpha]
        for i in range(num_sites):
            temp_mole = np.append(temp_mole, [mole[similars[alpha][i]]], axis = 0)
        #Make sure, that m1+m2 does no exceed length of similars[alpha] = num_sites
        if m1+m2 > len(similars[alpha]):
            raise ValueError("Number of changing atoms m1 + m2 =", m1+m2, "exceeds the number of alchemically similar sites in set", alpha, "which is", num_sites)
        '''Now: go through all configurations of changing m1 + m2 atoms of set similars[alpha]
        with size num_sites by the values stored in dZ. Then: compare their CN_inertia_moments
        and only count the unique ones'''
        atomwise_config = np.zeros((num_sites, 3)) #Three possible states: 0, dZ1, dZ2
        standard_config = np.zeros((num_sites))
        #First: All allowed charges for ONE atom at a time
        for i in range(num_sites):
            #no change:
            atomwise_config[i][0] = elements[temp_mole[i][0]]
            #Initalize standard_config:
            standard_config[i] = atomwise_config[i][0]
            #just changes:
            atomwise_config[i][1] = elements[temp_mole[i][0]]+dZ1
            atomwise_config[i][2] = elements[temp_mole[i][0]]+dZ2
        #Second: All possible combinations of those atoms with meshgrid
        #The * passes the arrays element wise
        mole_config = np.array(np.meshgrid(*atomwise_config.tolist())).T.reshape(-1,num_sites)

        #Third: Delete all arrays where number of changes unequal m1 and dZ1 and m2 and dZ2
        config_num = 0
        while config_num < len(mole_config):
            '''Every configuration has multiple things to fulfill:
            m1 sites need to be changed by dZ1, m2 sites need to be changed by dZ2.
            The netto charge conservation is already baked into the code.'''
            if (m1 == (np.subtract(mole_config[config_num],standard_config) == dZ1).sum()) and (m2 == (np.subtract(mole_config[config_num],standard_config) == dZ2).sum()):
                config_num += 1
            else:
                mole_config = np.delete(mole_config, config_num, axis = 0)
        #Check that atoms have not been transmuted to negative charges
        if np.min(mole_config) < 0:
            raise ValueError("Values in dZ lead to negative nuclear charges in alchemically similar sites")
        #Fourth: All remaining configs, their Coulomb inertia moments and their Delta_Coulomb inertia moments are saved and uniqued
        CIM = np.zeros((config_num, 3))
        '''Delta_CIM calculates the inertia moments of a helper molecule with charge 1 at
        changed sites and 0 elsewhere. In this fashion, we can distinguish alchemically distinct
        but spacially identical sites by their Delta_Coulomb_inertia moments'''
        Delta_CIM = np.zeros((config_num, 3), dtype=object)
        help_mole = np.copy(temp_mole)

        Total_CIM = np.zeros((config_num,3), dtype=object) #Entry 0: Config; entry 1: CN_inertia_moments; entry 2: Delta_CN_inertia_moments
        for i in range(config_num):
            for j in range(num_sites):
                temp_mole[j][0] = inv_elements[mole_config[i][j]]

            CIM[i] = CN_inertia_moment(temp_mole)
            round(CIM[i][0],tolerance)
            round(CIM[i][1],tolerance)
            round(CIM[i][2],tolerance)
            #print(CIM[i])

            #Initalize the helper molecule
            for j in range(num_sites):
                if np.subtract(standard_config[j],mole_config[i][j]) == 0:
                    help_mole[j][0] = inv_elements[0]
                else:
                    help_mole[j][0] = inv_elements[1]
            Delta_CIM[i] = CN_inertia_moment(help_mole)
            round(Delta_CIM[i][0],tolerance)
            round(Delta_CIM[i][1],tolerance)
            round(Delta_CIM[i][2],tolerance)
            #print(Delta_CIM[i])

            Total_CIM[i][0] = np.copy(temp_mole)
            Total_CIM[i][1] = np.copy(CIM[i])
            Total_CIM[i][2] = np.copy(Delta_CIM[i])
        '''Now, all possible configurations are obtained; with two loops, we can
        find all unique ones (uniquing CIM). However, we are not interested in the
        number of unique configurations but in all unique configurations that experienced
        changes in the SAME sites (uniquing Delta_CIM afterwards).
        Then, this molecule and the number of its enantiomers shall be printed!'''

        '''Delete all SPACIALLY equivalent configurations, i.e. all second, third, etc.
        occurences of a spacial configuration'''
        #Initalize array of already seen CIMs. No better way to do this?
        seen = np.array([[1.,2.,3.]])
        seen = np.delete(seen, 0, axis= 0)
        config_num = 0
        while config_num <len(Total_CIM):
            if not array_compare(Total_CIM[config_num][1], seen):
                seen = np.append(seen, [Total_CIM[config_num][1]], axis = 0)
                config_num += 1
            else:
                Total_CIM = np.delete(Total_CIM, config_num, axis = 0)

        '''Delete all ALCHEMICALLY equivalent configurations, i.e. all second, third, etc.
        occurences of an alchemical configuration'''
        #Initalize array of already seen Delta_CIMs. Still no better way to do this?
        seen = np.array([[1.,2.,3.]])
        seen = np.delete(seen, 0, axis= 0)
        config_num = 0
        while config_num <len(Total_CIM):
            #print("seen:", seen)
            #print("Total_CIM:", Total_CIM[config_num][2])
            if not array_compare(Total_CIM[config_num][2], seen):
                seen = np.append(seen, [Total_CIM[config_num][2]], axis = 0)
                config_num += 1
            else:
                Total_CIM = np.delete(Total_CIM, config_num, axis = 0)
        #print(seen)
        #print('---------------')
        #print(Total_CIM)

        '''All is done. Now, print the remaining atoms in sites and their contribution
        to count.'''
        print('---------------')
        for i in range(len(Total_CIM)):
            print(Total_CIM[i][0])
        print('Number of Alchemical Enantiomers from these sites:', len(Total_CIM))
        print('---------------')
        count += len(Total_CIM)
        '''At this point of the algorithm, the np.unique class could be easily used
        to return the unique configurations, but of the ENTIRE molecule.'''

        #Clear temp_mole
        temp_mole = np.array([['XXXXX', (1,2,3)]], dtype=object)
        temp_mole = np.delete(temp_mole, 0, 0)
    return count


#Furthermore: try partitions of m1 and m2 among different sites alpha, try num_AEsibling

benzene = [['C', (0,0,1)], ['C', (0,0.866025,0.5)], ['C', (0,0.866025,-0.5)],
['C', (0,0,-1)], ['C', (0,-0.866025,-0.5)], ['C', (0,-0.866025,0.5)]]

cube = [['Al', (0,0,0)], ['Al', (1,0,0)], ['Al', (1,1,0)], ['Al', (0,1,0)],
['Al', (0,0,1)], ['Al', (1,0,1)], ['Al', (1,1,1)], ['Al', (0,1,1)]]

naphthalene = [['C', (0,0,1)], ['C', (0,0.866025,0.5)], ['C', (0,0.866025,-0.5)],
['C', (0,0,-1)], ['C', (0,-0.866025,-0.5)], ['C', (0,-0.866025,0.5)],
['C', (0,2*0.866025,1)], ['C', (0,3*0.866025,0.5)], ['C', (0,3*0.866025, -0.5)], ['C', (0,2*0.866025,-1)]]


start_time = time.time()
print(num_AEchildren(naphthalene, m1=1, dZ1=+1, m2=1, dZ2=-1))
print("Time:", round((time.time() - start_time),3), "s")
