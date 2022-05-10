import numpy as np
from pyscf import gto, scf, qmmm
import os
import itertools
from copy import deepcopy as dc
import basis_set_exchange as bse

basis = 'def2-TZVP'
PathToQM9XYZ = '/home/simon/QM9/XYZ/'


# The main part of the program: Given a molecule in a list of lists, how many AEs are
# there if one considers the input molecule as the alchemical mirror plane? And what
# do they look like?

# How to split the big datasets into small chunks
# Split QM9:              python -c "for i,c in enumerate(open('dsgdb9nsd.xyz').read()[:-1].split('\n\n')): open(f'dsgdb9nsd_{i+1:06d}.xyz', 'w').write(c+'\n')"
# Split ZINC/named:       python -c "for i,c in enumerate(open('named.mol2').read()[:-1].split('@<TRIPOS>MOLECULE')): open(f'ZINC_named_{i:05d}.mol2', 'w').write(c+'\n')"
# Split ZINC/in-vivo:     python -c "for i,c in enumerate(open('in-vivo.mol2').read()[:-1].split('@<TRIPOS>MOLECULE')): open(f'ZINC_in-vivo_{i:05d}.mol2', 'w').write(c+'\n')"


elements = {'ghost':0, 'H':1, 'He':2,
'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18,
'K':19, 'Ca':20, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,}

inv_elements = {v: k for k, v in elements.items()}

# Below are all the partitions of splitting m_tot = np.sum(dZ_all[i])
# atoms in a pure (i.e. uncolored/isoatomic) molecule in n=len(dZ_all[i]) partitions
# for dZ_max = 3 up to m_tot = 8 and n = 2 and 3
m_possibilities = np.array([
[1,1],[1,1],[1,1],
[2,1],
[2,2],[2,2],[2,2],[3,1],
[2,3],
[3,3],[3,3],[3,3],[2,4],
[2,6],[4,4],[4,4],[4,4],
[1,1,1],
[2,1,1],
[1,1,3],[1,2,2],
[2,2,2],[1,1,4],[1,2,3],
[1,2,5],[1,2,5],[1,3,4],[1,3,4],[2,2,4],[2,3,3]
], dtype=object)
dZ_possibilities = np.array([
[1,-1],[2,-2],[3,-3],
[-1,2],
[1,-1],[2,-2],[3,-3],[-1,3],
[3,-2],
[1,-1],[+2,-2],[3,-3],[2,-1],
[3,-1],[1,-1],[2,-2],[3,-3],
[3,-2,-1],
[2,-1,-3],
[2,1,-1],[-2,2,-1],
[3,-2,-1],[3,1,-1],[3,-3,1],
[3,1,-1],[1,2,-1],[-2,2,-1],[-1,3,-2],[1,3,-2],[-3,3,-1]
],dtype=object)

#Monkey patching PySCF's qmmm:
def add_qmmm(calc, mol, Z):
    mf = qmmm.mm_charge(calc, mol.atom_coords()*0.52917721067, Z)
    def energy_nuc(self):
        q = mol.atom_charges().astype(np.float)
        q += Z
        return mol.energy_nuc(q)
    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)
    return mf

#Dictionary which works as a dump for all previously computed energies
already_compt = {}

#ALL BASIC FUNCTIONS WITHOUT ANY DEPENDENCIES-----------------------------------

def geom_hash(input_geometry, Z):
    #Assigns a hash value to a geometry to store it for later
    hash = ''
    N = len(input_geometry)
    for i in range(N):
        hash += '___'
        hash += input_geometry[i][0]+str(Z[i])
        hash += '___'
        for j in [1,2,3]:
            hash += str(round(input_geometry[i][j],3))
    return hash


def delta(i,j):
    #Kronecker Delta
    if i == j:
        return 1
    else:
        return 0


def are_close_scalars(a,b,looseness):
    value = False
    if abs(a-b) < looseness:
        value = True
    return value


def are_close_lists(a,b):
    value = True
    for i in range(len(a)):
        if abs(a[i]-b[i]) > 0.05*np.sqrt(len(a)): #The prefactor allows less closer arrays to appear so.
            value = False
    return value


def array_compare(arr1, arr2):
    # arr1 = [...]
    # arr2 = [[...],[...],[...],...]
    # Is there an approximate copy of arr1 in arr2
    within = False
    for i in range(len(arr2)):
        if are_close_lists(arr1, arr2[i]):
            within = True
    return within


def center_mole(mole):
    #Centers a molecule
    sum = [0,0,0]
    N = len(mole)
    for i in range(N):
        sum[0] = sum[0] + mole[i][1]
        sum[1] = sum[1] + mole[i][2]
        sum[2] = sum[2] + mole[i][3]
    sum[0] = sum[0]/N
    sum[1] = sum[1]/N
    sum[2] = sum[2]/N
    result = dc(mole)
    for i in range(N):
        result[i][1] = result[i][1] - sum[0]
        result[i][2] = result[i][2] - sum[1]
        result[i][3] = result[i][3] - sum[2]
    return result


def Coulomb_matrix(mole):
    #returns the Coulomb matrix of a given molecule
    N = len(mole)
    result = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if (j == i):
                charge = elements[mole[i][0]]
                summand = 0.5*pow(charge, 2.4)
                result[i][i] = summand
            else:
                summand = elements[mole[i][0]]*elements[mole[j][0]]/np.linalg.norm(np.subtract(mole[i][1:],mole[j][1:]))
                result[i][j] = summand
    return result


def atomrep(mole):
    # Find a good representation for the molecule, which is just the sum over the rows or
    # columns of the Coulomb matrix
    N = len(mole)
    matrix = Coulomb_matrix(mole)
    #Calculate the norm:
    sum = 0
    result = []
    for i in range(N):
        for j in range(N):
            sum = sum + matrix[i][j]**2
        sum = sum**0.5
        result.append(sum)
        sum = 0
    return result


def atomrep_inertia_moment(mole):
    #Calculate the inertia moments of a molecule with atomrep instead of masses
    #and sort them in ascending order
    # First: The intertia tensor
    N = len(mole)
    CN = atomrep(mole)
    result_tensor = np.zeros((3,3))
    sum = 0
    for i in range(3):
        for j in range(i+1):
            for k in range(N):
                sum = sum + CN[k]*(np.dot(mole[k][1:], mole[k][1:])*delta(i,j) - mole[k][1+i]*mole[k][1+j])
            result_tensor[i][j] = dc(sum)
            result_tensor[j][i] = dc(sum)
            sum = 0
    w,v = np.linalg.eig(result_tensor)
    #Only the eigen values are needed, v is discarded
    moments = np.sort(w)
    return moments


def get_equi_atoms_from_geom(mole, with_hydrogen = True):
    #Spits out all atoms that have similar environment
    #Sort CN and group everything together that is not farther than tolerance
    CN = atomrep(center_mole(mole))
    #In atomrep, the hydrogens have been considered. Now, get rid of them again
    indices = dc(np.argsort(CN).tolist())
    indices2 = dc(indices)
    lst = []
    similars = []
    for i in range(len(indices)-1):
        if i not in indices2:
            continue
        lst.append(indices[i])
        indices2.remove(i)
        for j in range(i+1,len(indices)):
            if are_close_scalars(CN[indices[i]],CN[indices[j]],0.05):
                lst.append(indices[j])
                indices2.remove(j)
        similars.append(lst)
        lst = []
    #If without hydrogen: Delete all hydrogens
    if not with_hydrogen:
        num_similars = 0
        while num_similars < len(similars):
            num_members = 0
            while num_members < len(similars[num_similars]):
                if mole[similars[num_similars][num_members]][0] != 'H':
                    num_members += 1
                else:
                    del similars[num_similars][num_members]
            num_similars += 1
    #Delete all similars which include only one atom or none at all:
    num_similars = 0
    while num_similars < len(similars):
        if len(similars[num_similars])>1:
            num_similars += 1
        else:
            del similars[num_similars]
    return similars


def nuclear_energy(mole, Z=[]):
    # Calculate the nuclear energy of the molecule mole with charges Z stacked on top of
    # the atoms
    sum = 0
    N = len(mole)
    if len(Z) == 0:
        Z = np.zeros((N)).tolist()
    for i in range(N):
        for j in range(i+1,N):
            sum += (elements[mole[i][0]]+Z[i])*(elements[mole[j][0]]+Z[j])/np.linalg.norm(np.subtract(mole[i][1:],mole[j][1:]))
    #Result needs to be in Ha, and the length has been in Angstrom
    return sum*0.529177210903


def total_energy(mole, Z=[], basis=basis, method = 'HF'):
    N = len(mole)
    if len(Z) == 0:
        Z = np.zeros((N)).tolist()
    if geom_hash(mole, Z) in already_compt:
        return already_compt[geom_hash(mole, Z)]
    else:
        #Z are additional charges; electrons are accounted for Z
        atom_string = ''
        overall_charge = 0
        for i in range(N): #get the atoms and their coordinates
            atom_string += mole[i][0]
            overall_charge += elements[mole[i][0]]+Z[i]
            for j in [1,2,3]:
                atom_string += ' ' + str(mole[i][j])
            atom_string += '; '
        # Initialize the molecule
        mol = gto.M(atom = atom_string[:-2], basis = bse.get_basis(basis, fmt='nwchem'), unit='Angstrom', charge=0, spin=0, verbose=0)
        if method == 'HF':
            calc = add_qmmm(scf.UHF(mol).density_fit(), mol, Z)
            hfe = calc.density_fit().kernel(verbose=0)
            total_energy = calc.e_tot
            already_compt.update({geom_hash(mole, Z):total_energy})
            return total_energy
        if method == 'DFT':
            mf = dft.UKS(mol)
            mf.xc = 'pbe,pbe'
            #THE CRUCIALL BIT!!! NO IDEA IF THIS EVEN WORKS...
            calc = add_qmmm(dft.UKS(mol), mol, Z)
            mf = mf.newton() # second-order algortihm
            mf.kernel()
            if spin == 0:
                dm = mf.make_rdm1()
            if spin == 1:
                dm = mf.make_rdm1()[0] + mf.make_rdm1()[1]
            total_energy = mf.e_tot
            already_compt.update({geom_hash(mole, Z):total_energy})
            return total_energy


def electronic_energy(mole, Z=[], basis=basis):
    return total_energy(mole, Z, basis=basis) - nuclear_energy(mole,Z)


#PARSER FUNCTION FOR QM9--------------------------------------------------------
def parse_XYZtoMAG(input_PathToFile):
    #check if file is present
    if os.path.isfile(input_PathToFile):
        #open text file in read mode
        f = open(input_PathToFile, "r")
        data = f.read()
        f.close()
    else:
        print('\nFile', input_PathToFile, 'not found!!!', end='\n\n')
    #Get the name of the molecule, or at least the file's name
    MAG_name = input_PathToFile.split('/')[-1].split('.')[0]
    N = int(data.splitlines(False)[0]) #number of atoms including hydrogen
    #Get the geometry of the molecule
    mole = []
    for i in range(2,N+2): #get the atoms and their coordinates
        line = data.splitlines(False)[i]
        symbol = line.split('\t')[0]
        x = float(line.split('\t')[1].strip())
        y = float(line.split('\t')[2].strip())
        z = float(line.split('\t')[3].strip())
        mole.append([symbol,x,y,z])
    # Sort the atoms such that the hydrogens are always the last ones
    for i in range(N):
        if mole[i][0] == 'H':
            mole[i], mole[-1] = mole[-1], mole[i]
    mole = center_mole(mole)
    return mole, MAG_name


#ALL HIGHER LEVEL FUNCTIONS WITH VARIOUS DEPENDENCIES---------------------------
def dlambda_electronic_energy(mole, Z, dlambda, order):
    #Z is the deviation of the molecule from integer nuclear charges
    #dlambda is needed as the basis vector for the parameter lambda and is the change of nuclear charges at lamda=1
    step = 0.02
    if order < 1:
        if geom_hash(mole, Z) in already_compt:
            return already_compt[geom_hash(mole, Z)]
        else:
            result = electronic_energy(mole, Z)
            already_compt.update({geom_hash(mole, Z):result})
            return result
    else:
        def f(b):
            return dlambda_electronic_energy(mole, [x+b*step*y for x,y in zip(Z,dlambda)], dlambda, order-1)
        # Various differentiation stencils below, pick one:
        #return (-f(4)/280 + 4*f(3)/105 - f(2)/5 + 4*f(1)/5 - 4*f(-1)/5 + f(-2)/5 - 4*f(-3)/105 + f(-4)/280)/step
        return (-f(2)/12 + 2*f(1)/3 - 2*f(-1)/3 + f(-2)/12)/step
        #return (f(1)-2*f(0)+f(-1))/(step)

def lambda_taylorseries_electronic_energy(mole, Z, dlambda, order):
    # dlambda is a list with the desired difference in nuclear charge of the endpoints
    # compared to the current state of the molecule (so the difference transmuted for
    # lambda = 1)
    return dlambda_electronic_energy(mole, Z, dlambda, order)/math.factorial(order)


def geomAE(mole, m=[2,2], dZ=[1,-1], with_electronic_energy_difference = False, with_hydrogen = False):
    # Returns the number of alchemical enantiomers of mole that can be reached by
    # varying m[i] atoms in mole with identical Coulombic neighborhood by dZ[i].
    # Alternatively, the hydrogens can also be considered as transmutable
    N_m = len(m)
    N_dZ = len(dZ)
    equi_atoms = get_equi_atoms_from_geom(mole, with_hydrogen=with_hydrogen)
    original_mole = dc(mole)

    if not with_hydrogen:
        # Remove the hydrogens
        mole = [mole[i] for i in range(len(mole)) if mole[i][0] != 'H']
    N = len(mole)

    if (np.sum(np.multiply(m,dZ)) != 0):
        raise ValueError("Netto change in charge must be 0")
    if 0 in dZ:
        raise ValueError("0 not allowed in array dZ")
    if N_m != N_dZ:
        raise ValueError("Number of changes and number of charge values do not match!")
    if len(equi_atoms) == 0:
        return 0
    # All of these sites in each set need to be treated simultaneously. Hence, we
    # flatten the array. However, we later need to make sure that only each set fulfills
    # netto charge conservation. This is why similars is initalized
    similars = list(itertools.chain(*equi_atoms))
    #Make sure, that m1+m2 does no exceed length of similars
    if np.sum(m) > len(similars):
        return 0
    # This is the list of all atoms which can be transmuted simultaneously.
    # Now, count all molecules which are possible excluding mirrored or rotated versions
    count = 0
    #Initalize empty array temp_mole for all configurations.
    temp_mole = []
    #Get necessary atoms listed in equi_atoms
    for i in range(len(similars)):
        temp_mole.append(mole[similars[i]])
    # Now: go through all combinations of transmuting m atoms of set similars
    # by the values stored in dZ. Then: compare their atomrep_inertia_moments
    # and only count the unique ones (i.e. discard spatial enantiomers)
    # This is the generation and first filtering step
    atomwise_config = np.zeros((len(similars), N_dZ+1), dtype='int') #N_dZ+1 possible states: 0, dZ1, dZ2, ...
    standard_config = np.zeros((len(similars)))
    #All allowed charges for ONE atom at a time
    for i in range(len(similars)):
        #no change:
        atomwise_config[i][0] = elements[temp_mole[i][0]]
        #Initalize standard_config:
        standard_config[i] = elements[temp_mole[i][0]]
        #just changes:
        for j in range(N_dZ):
            atomwise_config[i][j+1] = elements[temp_mole[i][0]]+dZ[j]
    #All possible combinations of those atoms with meshgrid; the * passes the arrays element-wise
    mole_config_unfiltered = np.array(np.meshgrid(*atomwise_config.tolist(), copy=False),dtype='int').T.reshape(-1,len(similars))
    #Initalize a molecule configuration:
    mole_config = np.zeros((1,len(similars)),dtype='int')
    mole_config = np.delete(mole_config, 0, axis = 0)

    for k in range(len(mole_config_unfiltered)):
        # m1 sites need to be changed by dZ1, m2 sites need to be changed by dZ2, etc...
        if np.array([(m[v] == (np.subtract(mole_config_unfiltered[k],standard_config) == dZ[v]).sum()) for v in range(N_dZ)]).all():
            # Check that the netto charge change in every set of equivalent atoms is 0
            pos = 0
            for i in range(len(equi_atoms)):
                sum = 0
                #This loop has to start where the last one ended:
                for j in range(pos,pos+len(equi_atoms[i])):
                    sum += mole_config_unfiltered[k][j] - standard_config[j]
                pos += len(equi_atoms[i])
                if sum != 0:
                    break
                if (sum == 0) and (i == len(equi_atoms)-1):
                    mole_config = np.append(mole_config, [mole_config_unfiltered[k]], axis = 0)
    if len(mole_config) == 0:
        return 0
    if np.min(mole_config) < 0:
        #Check that atoms have not been transmuted to negative charges
        raise ValueError("Values in dZ lead to negative nuclear charges in electronically equivalent atoms.")
    #Fourth: All remaining configs, their Coulomb inertia moments and their Delta_Coulomb inertia moments are saved and uniqued
    CIM = np.zeros((len(mole_config), 3))

    Total_CIM = np.zeros((len(mole_config),2), dtype=object) #Entry 0: Config; entry 1: atomrep_inertia_moments
    for i in range(len(mole_config)):
        for j in range(len(similars)):
            temp_mole[j][0] = inv_elements[mole_config[i][j]]
        CIM[i] = atomrep_inertia_moment(temp_mole)
        Total_CIM[i][0] = dc(temp_mole)
        Total_CIM[i][1] = dc(CIM[i])
    mole = dc(original_mole)
    # Now, all possible combinations are obtained; with the following loops,
    # we can get rid of all the spacial enantiomers: Delete all SPATIALLY
    # equivalent configurations, i.e. all second, third, etc. occurences of a
    # spacial configuration

    #Initalize array of already seen CIMs.
    seen = np.array([[1.,2.,3.]])
    seen = np.delete(seen, 0, axis= 0)

    config_num = 0
    while config_num < len(Total_CIM):
        if not array_compare(Total_CIM[config_num][1], seen):
            seen = np.append(seen, [Total_CIM[config_num][1]], axis = 0)
            config_num += 1
        else:
            Total_CIM = np.delete(Total_CIM, config_num, axis = 0)

    # Alchemical enantiomers are those molecules which do not transmute
    # into themselves (or its spatial enantiomer) under the mirroring in charge,
    # i.e. if one adds the inverted configuration of transmutations to twice the molecule,
    # its CIM has changed. This is the second filtering step
    config_num = 0
    while config_num < len(Total_CIM):
        current_config = np.zeros(len(similars),dtype=object)
        for i in range(len(similars)):
            current_config[i] = elements[Total_CIM[config_num][0][i][0]]
        mirror_config = 2*standard_config - current_config
        for i in range(len(similars)):
            temp_mole[i][0] = inv_elements[mirror_config[i]]
        if are_close_lists(Total_CIM[config_num][1], atomrep_inertia_moment(temp_mole)):
            Total_CIM = np.delete(Total_CIM, config_num, axis = 0)
        else:
            config_num += 1
    # All is done. Now, print the remaining configurations and their contribution to count.
    count += len(Total_CIM)

    if with_electronic_energy_difference and len(Total_CIM) > 0:
        # Explicitly calculate the energies of all the configurations in Total_CIM[0]
        # and their mirrors, then print their energy difference
        for i in range(len(Total_CIM)):
            #print('--------------------------------')
            Z_new = np.zeros((len(original_mole))).tolist()
            num = 0
            #Initalize the two strings to discriminate the AEs
            AE_string1 = ''
            AE_string2 = ''
            for j in similars:
                Z_new[j] = elements[Total_CIM[i][0][num][0]]-elements[original_mole[j][0]]
                num += 1
            #Fill both strings
            for j in range(len(original_mole)):
                AE_string1 += inv_elements[elements[original_mole[j][0]]+Z_new[j]]
                AE_string2 += inv_elements[elements[original_mole[j][0]]-Z_new[j]]
            diff = electronic_energy(original_mole, Z = Z_new) - electronic_energy(original_mole, Z = [-x for x in Z_new])
            print(AE_string1+" minus "+AE_string2+str(diff))

    # print(Total_CIM)
    return count


def Find_AEfromref(mole, dZ_max = 3, with_electronic_energy_difference = False, with_hydrogen = False):
    dZ_all = dc(dZ_possibilities)
    m_all = dc(m_possibilities)
    original_mole = dc(mole)
    N = len(mole)
    #Get rid of all dZs > dZ_max and all m's > N:
    num = 0
    while num < len(dZ_all):
        if np.max(dZ_all[num]) > dZ_max or sum(m_all[num]) > N:
            m_all = np.delete(m_all, num, axis = 0)
            dZ_all = np.delete(dZ_all, num, axis = 0)
        else:
            num += 1

    total_number = 0

    for i in range(len(m_all)):
        mole = dc(original_mole)
        random_config = np.zeros((N))
        pos = 0
        for j in range(len(m_all[i])):
            for k in range(m_all[i][j]):
                random_config[pos] = dZ_all[i][j]
                pos += 1
        x = geomAE(mole, m_all[i], dZ_all[i], with_electronic_energy_difference = with_electronic_energy_difference, with_hydrogen=with_hydrogen)
        total_number += x
    return total_number

tot = Find_AEfromref(parse_XYZtoMAG('/home/simon/QM9/XYZ/dsgdb9nsd_000214.xyz')[0], dZ_max = 1)
print(tot)
