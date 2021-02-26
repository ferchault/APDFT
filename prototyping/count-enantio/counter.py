import numpy as np
import time
import matplotlib.pyplot as plt

elements = {'-C': -6, '-B': -5, '-Be': -4, '-Li': -3, '-He': -2, '-H': -1, 'Ghost':0,
'H':1, 'He':2,
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
                charge = elements[mole[i][0]]
                result[i][i] = 0.5*np.sign(charge)*pow(np.abs(charge), 2.4)
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

def num_AEchildren(mole, m1 = 2, dZ1 = +1, m2 = 2, dZ2 = -1, partition = False, debug = False):
    '''Returns the number of alchemical enantiomers of mole that can be reached by
    varying m1 and m2 atoms in mole with identical Coulombic neighborhood by dZ1
    dZ2, respectively.'''

    '''Partition allows the changed atoms to be of more than one set of points with
    identical Coulombic neighborhood.'''

    '''In case that the same number of nuclear charges are increased as they are
    decreased, the set of AE at the end include their own mirror images. Depending
    on your input constraints, the returned number has to be halfed!'''

    N = len(mole)
    if N < m1 + m2:
        raise ValueError("Number of changing atoms must not exceed number of atoms in molecule")
    if (m1*dZ1 + m2*dZ2 != 0):
        raise ValueError("Netto change in charge must be 0: m1*dZ1 = -m2*dZ2. You entered: %d = %d" %(m1*dZ1, -m2*dZ2))
    if (dZ1 == 0) or (dZ2 == 0):
        raise ValueError("0 is not allowed in dZ")
    if (partition != True) and (partition != False):
        raise ValueError("Partition must be True or False")
    #Prepare the molecule
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
    '''If partitoning is allowed, all of these sites in each set need to be treated
    simultaneously. Hence, we flatten the array. However, we later need to make sure
    that only each set fulfills netto charge change = 0. This is why set is Initalized'''
    if partition == True:
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
        if m1+m2 > len(similars[alpha]):
            print('---------------')
            print("Warning: Number of changing atoms m1 + m2 = %d exceeds the number of alchemically \nsimilar sites in set %d which is %d. Hence, the returned value is 0 at this site." %(m1+m2,alpha,num_sites))
            print('Number of Alchemical Enantiomers from site with index %d: 0' %alpha)
            print('---------------')
            continue

        '''Now: go through all configurations of changing m1 + m2 atoms of set similars[alpha]
        with size num_sites by the values stored in dZ. Then: compare their CN_inertia_moments
        and only count the unique ones'''
        atomwise_config = np.zeros((num_sites, 3)) #Three possible states: 0, dZ1, dZ2
        standard_config = np.zeros((num_sites))
        #All allowed charges for ONE atom at a time
        for i in range(num_sites):
            #no change:
            atomwise_config[i][0] = elements[temp_mole[i][0]]
            #Initalize standard_config:
            standard_config[i] = atomwise_config[i][0]
            #just changes:
            atomwise_config[i][1] = elements[temp_mole[i][0]]+dZ1
            atomwise_config[i][2] = elements[temp_mole[i][0]]+dZ2
        #All possible combinations of those atoms with meshgrid
        #The * passes the arrays element wise
        mole_config = np.array(np.meshgrid(*atomwise_config.tolist())).T.reshape(-1,num_sites)
        config_num = 0
        while config_num < len(mole_config):
            '''m1 sites need to be changed by dZ1, m2 sites need to be changed by dZ2.
            The netto charge conservation is already baked into the code for partition == False.'''
            if (m1 == (np.subtract(mole_config[config_num],standard_config) == dZ1).sum()) and (m2 == (np.subtract(mole_config[config_num],standard_config) == dZ2).sum()):
                if partition == True:
                    '''Check that the netto charge change in every set is 0 (work in progress!!!!)'''
                    for i in range(len(set)):
                        sum = 0
                        #This loop has to start where the last one ended
                        pos = 0
                        for j in range(pos,pos+len(set[i])):
                            sum += mole_config[config_num][j] - standard_config[j]
                        pos += len(set[i])
                        if sum != 0:
                            mole_config = np.delete(mole_config, config_num, axis = 0)
                            break
                        if (sum == 0) and (i == len(set)-1):
                            config_num += 1
                if partition == False:
                    config_num += 1
            else:
                mole_config = np.delete(mole_config, config_num, axis = 0)
        #Check that atoms have not been transmuted to negative charges
        if np.min(mole_config) < 0:
            raise ValueError("Values in dZ lead to negative nuclear charges in alchemically similar sites")
        #Fourth: All remaining configs, their Coulomb inertia moments and their Delta_Coulomb inertia moments are saved and uniqued
        CIM = np.zeros((config_num, 3))

        Total_CIM = np.zeros((config_num,2), dtype=object) #Entry 0: Config; entry 1: CN_inertia_moments
        for i in range(config_num):
            for j in range(num_sites):
                temp_mole[j][0] = inv_elements[mole_config[i][j]]

            CIM[i] = CN_inertia_moment(temp_mole)
            round(CIM[i][0],tolerance)
            round(CIM[i][1],tolerance)
            round(CIM[i][2],tolerance)
            #print(CIM[i])

            Total_CIM[i][0] = np.copy(temp_mole)
            Total_CIM[i][1] = np.copy(CIM[i])

        '''Now, all possible configurations are obtained; with the following loops,
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
        #print(Total_CIM)

        '''ALCHEMICALLY symmetric molecules are those which do not transmute
        into themselves under the mirroring in charge, i.e. if one adds the
        inverted (minus sign) configuration of charge changes to twice the molecule,
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

        '''All is done. Now, print the remaining atoms in sites and their contribution
        to count. We only need one half as we know every AEs partner immediatly.'''
        count += len(Total_CIM)

        if debug == True:
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
                for j in range(num_sites):
                    xs[j] = Total_CIM[i][0][j][1][0]
                    ys[j] = Total_CIM[i][0][j][1][1]
                    zs[j] = Total_CIM[i][0][j][1][2]
                #print(xs,ys,zs)
                ax.scatter(xs, ys, zs)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.show()
            print('Number of molecules to be consider Alchemical Enantiomers from site with index %d: %d' %(alpha,len(Total_CIM)))
            print('---------------')
        #Clear temp_mole
        temp_mole = np.array([['XXXXX', (1,2,3)]], dtype=object)
        temp_mole = np.delete(temp_mole, 0, 0)
    return count


'''Furthermore: validate with brute force method, try num_AEsibling'''

benzene = [['C', (0,0,1)], ['C', (0,0.8660254037844386467637231707,0.5)], ['C', (0,0.8660254037844386467637231707,-0.5)],
['C', (0,0,-1)], ['C', (0,-0.8660254037844386467637231707,-0.5)], ['C', (0,-0.8660254037844386467637231707,0.5)]]

cube = [['Al', (0,0,0)], ['Al', (1,0,0)], ['Al', (1,1,0)], ['Al', (0,1,0)],
['Al', (0,0,1)], ['Al', (1,0,1)], ['Al', (1,1,1)], ['Al', (0,1,1)]]

naphthalene = [['C', (0,0,1)], ['C', (0,0.8660254037844386467637231707,0.5)], ['C', (0,0.8660254037844386467637231707,-0.5)],
['C', (0,0,-1)], ['C', (0,-0.8660254037844386467637231707,-0.5)], ['C', (0,-0.8660254037844386467637231707,0.5)],
['C', (0,2*0.8660254037844386467637231707,1)], ['C', (0,3*0.8660254037844386467637231707,0.5)], ['C', (0,3*0.8660254037844386467637231707, -0.5)], ['C', (0,2*0.8660254037844386467637231707,-1)]]

triangle = [['C', (0,0,1)], ['C', (0,1,0)], ['C', (1,0,0)]]

metal_octa = [['Al', (0,0.5,0.5)], ['Al', (0,0.5,-0.5)], ['Al', (0,-0.5,-0.5)], ['Al', (0,-0.5,0.5)],
['C', (0,0,1)],['C', (0,1,0)],['C', (0,0,-1)],['C', (0,-1,0)]]

start_time = time.time()
print(num_AEchildren(benzene, m1=2, dZ1=+1, m2=1, dZ2=-2, partition = True, debug = True))
print("Time:", (time.time() - start_time))
