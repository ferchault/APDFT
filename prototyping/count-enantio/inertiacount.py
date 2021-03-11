import numpy as np
import time
import matplotlib.pyplot as plt
import os
import igraph

elements = {'Ghost':0,'H':1, 'He':2,
'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18,
'K':19, 'Ca':20, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,}

inv_elements = {v: k for k, v in elements.items()}

tolerance = 3
'''Rounding to ... digits for geometry-based method. Do not go above 3, if you do not want
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
                charge = elements[mole[i][0]]
                result[i][i] = 0.5*pow(charge, 2.4)
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
    Is there an exact copy of arr1 in arr2'''
    within = False
    for i in range(len(arr2)):
        if (arr1 == arr2[i]).all():
            within = True
    return within

def sum_formula(array_of_atoms):
    values, counts = np.unique(np.array(array_of_atoms), return_counts=True)
    formula = ''
    for i in range(len(values)):
        if values[i] == 'C':
            if counts[i] > 1:
                formula = str(counts[i]) + formula
            formula = 'C' + formula
        else:
            formula += str(values[i])
            if counts[i] > 1:
                formula += str(counts[i])
    return formula

def geomAE(mole, m=[2,2], dZ=[1,-1], debug = False, chem_formula = True):
    if mole == None:
        raise ValueError("Variable 'mole' must not be 'None'")
    m = np.array(m)
    dZ = np.array(dZ)
    N_m = len(m)
    N_dZ = len(dZ)
    '''Returns the number of alchemical enantiomers of mole that can be reached by
    varying m[i] atoms in mole with identical Coulombic neighborhood by dZ[i].'''

    start_time = time.time()

    N = len(mole)
    if N < np.sum(m):
        raise ValueError("Too less atoms in molecule for sum of to be transmuted atoms.")
    if (np.sum(np.multiply(m,dZ)) != 0):
        raise ValueError("Netto change in charge must be 0")
    if 0 in dZ:
        raise ValueError("0 not allowed in array dZ")
    if N_m != N_dZ:
        raise ValueError("Number of changes and number of charge values do not match!")
    #Prepare the molecule
    center_mole(mole)
    mole = np.array(mole, dtype=object)
    #Find the Coulombic neighborhood of each atom in mole
    CN = Coulomb_neighborhood(mole)
    #print(CN)
    '''To make life easier, the values in CN are rounded to tolerance for easier comparison'''
    for i in range(N):
        CN[i] = round(CN[i],tolerance)
    '''Are there atoms with identical/close Coulombic neighborhood? Find them and store
    their indices in similars'''
    similars = np.array([np.where(CN == i)[0] for i in np.unique(CN)],dtype=object)
    #print(similars)
    #Delete all elements in similars which include only one atom:
    num_similars = 0
    while num_similars < len(similars):
        if len(similars[num_similars])>1:
            num_similars += 1
        else:
            similars = np.delete(similars, num_similars, axis = 0)
    if len(similars) == 0:
        print("No sets of equivalent atoms found.")
        return 0
    '''All of these sites in each set need to be treated simultaneously. Hence, we
    flatten the array. However, we later need to make sure that only each set fulfills
    netto charge conservation. This is why set is initalized'''
    set = np.copy(similars)
    similars = [np.concatenate((similars).tolist())]
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
        if np.sum(m) > len(similars[alpha]):
            print('---------------')
            print("Warning: Number of to be transmuted atoms m = %d exceeds the number of electronically equivalent \n atoms in set %d which is %d. Hence, the returned value is 0 at this site." %(np.sum(m),alpha,num_sites))
            print('Number of Alchemical Enantiomers from set of equivalent atoms with index %d: 0' %alpha)
            print('---------------')
            continue

        '''Now: go through all combinations of transmuting m atoms of set similars[alpha]
        with size num_sites by the values stored in dZ. Then: compare their CN_inertia_moments
        and only count the unique ones'''
        atomwise_config = np.zeros((num_sites, N_dZ+1), dtype='int') #N_dZ+1 possible states: 0, dZ1, dZ2, ...
        standard_config = np.zeros((num_sites))
        #All allowed charges for ONE atom at a time
        for i in range(num_sites):
            #no change:
            atomwise_config[i][0] = elements[temp_mole[i][0]]
            #Initalize standard_config:
            standard_config[i] = atomwise_config[i][0]
            #just changes:
            for j in range(N_dZ):
                atomwise_config[i][j+1] = elements[temp_mole[i][0]]+dZ[j]
        #All possible combinations of those atoms with meshgrid
        #The * passes the arrays element wise
        mole_config_unfiltered = np.array(np.meshgrid(*atomwise_config.tolist(), copy=False)).T.reshape(-1,num_sites)
        mole_config = np.zeros((1,num_sites),dtype='int')
        mole_config = np.delete(mole_config, 0, axis = 0)
        for k in range(len(mole_config_unfiltered)):
            '''m1 sites need to be changed by dZ1, m2 sites need to be changed by dZ2, etc...'''
            if np.array([(m[j] == (np.subtract(mole_config_unfiltered[k],standard_config) == dZ[j]).sum()) for j in range(N_dZ)]).all():
                '''Check that the netto charge change in every set is 0'''
                pos = 0
                for i in range(len(set)):
                    sum = 0
                    #This loop has to start where the last one ended
                    for j in range(pos,pos+len(set[i])):
                        sum += mole_config_unfiltered[k][j] - standard_config[j]
                    pos += len(set[i])
                    if sum != 0:
                        break
                    if (sum == 0) and (i == len(set)-1):
                        #print(mole_config_unfiltered[k])
                        mole_config = np.append(mole_config, [mole_config_unfiltered[k]], axis = 0)
        if len(mole_config) == 0:
            return 0
        if np.min(mole_config) < 0:
            #Check that atoms have not been transmuted to negative charges
            raise ValueError("Values in dZ lead to negative nuclear charges in electronically equivalent atoms.")
        #Fourth: All remaining configs, their Coulomb inertia moments and their Delta_Coulomb inertia moments are saved and uniqued
        CIM = np.zeros((len(mole_config), 3))

        Total_CIM = np.zeros((len(mole_config),2), dtype=object) #Entry 0: Config; entry 1: CN_inertia_moments
        for i in range(len(mole_config)):
            for j in range(num_sites):
                temp_mole[j][0] = inv_elements[mole_config[i][j]]

            CIM[i] = CN_inertia_moment(temp_mole)
            round(CIM[i][0],tolerance)
            round(CIM[i][1],tolerance)
            round(CIM[i][2],tolerance)
            #print(CIM[i])

            Total_CIM[i][0] = np.copy(temp_mole)
            Total_CIM[i][1] = np.copy(CIM[i])

        '''Now, all possible combinations are obtained; with the following loops,
        we can get rid of all the spacial enantiomers: Delete all SPACIALLY
        equivalent configurations, i.e. all second, third, etc. occurences of a
        spacial configuration'''

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
        #print(len(Total_CIM))

        '''Alechemical enantiomers are those molecules which do not transmute
        into themselves under the mirroring in charge, i.e. if one adds the
        inverted configuration of transmutations to twice the molecule,
        its CIM has changed.'''

        config_num = 0
        while config_num <len(Total_CIM):
            current_config = np.zeros(num_sites,dtype=object)
            for i in range(num_sites):
                current_config[i] = elements[Total_CIM[config_num][0][i][0]]
            mirror_config = 2*standard_config - current_config
            #print(current_config)
            #print(mirror_config)
            #print(Total_CIM[config_num][1]) #The CIM of current_config
            for i in range(num_sites):
                temp_mole[i][0] = inv_elements[mirror_config[i]]
            #print(CN_inertia_moment(temp_mole))
            #print('----------')
            if (Total_CIM[config_num][1] == CN_inertia_moment(temp_mole)).all():
                Total_CIM = np.delete(Total_CIM, config_num, axis = 0)
            else:
                config_num += 1

        '''All is done. Now, print the remaining configurations and their contribution
        to count.'''
        count += len(Total_CIM)

        if chem_formula == True:
            random_config = np.zeros((N))
            pos = 0
            for i in range(len(m)):
                for j in range(m[i]):
                    random_config[pos] = dZ[i]
                    pos += 1
            print(sum_formula([inv_elements[elements[mole[j][0]]+random_config[j]] for j in range(N)]))

        if debug == True:
            print('---------------')
            print("Time:", (time.time() - start_time),"s")
            print('---------------')
            for i in range(len(Total_CIM)):
                #This prints the current configuration
                print(Total_CIM[i][0])
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                #Fill the points into xs,ys,zs
                xs = np.zeros((num_sites))
                ys = np.zeros((num_sites))
                zs = np.zeros((num_sites))
                n = np.zeros((num_sites),dtype=object)
                for j in range(num_sites):
                    xs[j] = Total_CIM[i][0][j][1][0]
                    ys[j] = Total_CIM[i][0][j][1][1]
                    zs[j] = Total_CIM[i][0][j][1][2]
                    n[j] = Total_CIM[i][0][j][0]
                ax.scatter(xs, ys, zs, marker='o', facecolor='black')
                #print(Total_CIM[i][0][1][0])
                for j, txt in enumerate(n):
                    ax.text(xs[j], ys[j], zs[j], n[j])
                #ax.set_xlabel('X')
                #ax.set_ylabel('Y')
                #ax.set_zlabel('Z')
            plt.show()
            print('Number of molecules to be considered one part of a pair of Alchemical Enantiomers \nfrom set of electronically equivalent atoms with index %d: %d' %(alpha,len(Total_CIM)))
            print('---------------')
        #Clear temp_mole
        temp_mole = np.array([['XXXXX', (1,2,3)]], dtype=object)
        temp_mole = np.delete(temp_mole, 0, 0)
    return count
