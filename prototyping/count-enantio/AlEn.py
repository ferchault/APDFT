#ALL IMPORT STATEMENTS----------------------------------------------------------
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import os
import igraph
import itertools
from pyscf import gto, scf
from pysmiles import read_smiles
import networkx as nx

#ALL CONFIGURATIONS AND GLOBAL VARIABLES----------------------------------------
original_stdout = sys.stdout # Save a reference to the original standard output
tolerance = 0.05 #Rounding error in geometry-based method
performance_use = 0.50 #portion of cpu cores to be used
gate_threshold = 0 #Cutoff threshold in Coulomb matrix
basis = 'ccpvdz'#'def2tzvp' #Basis set for QM calculations
PathToNauty27r1 = '/home/simon/nauty27r1/'
PathToQM9XYZ = '/home/simon/QM9/XYZ/'

elements = {'Ghost':0,'H':1, 'He':2,
'Li':3, 'Be':4, 'B':5, 'C':6, 'C-N':6.5, 'N':7, 'O':8, 'F':9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18,
'K':19, 'Ca':20, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,}

inv_elements = {v: k for k, v in elements.items()}

'''Below are all the partitions of splitting m_tot = np.sum(dZ_all[i])
atoms in a pure (i.e. uncolored/isoatomic) molecule in n=len(dZ_all[i]) partitions
for dZ_max = 3 up to m_tot = 8 and n = 2 and 3'''
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


#ALL BASIC FUNCTIONS WITHOUT ANY DEPENDENCIES-----------------------------------
def delta(i,j):
    #Kronecker Delta
    if i == j:
        return 1
    else:
        return 0

def are_close_scalars(a,b):
    value = False
    if abs(a-b) < tolerance:
        value = True
    return value

def are_close_lists(a,b):
    value = True
    for i in range(len(a)):
        if abs(a[i]-b[i]) > tolerance:
            value = False
    return value

def center_mole(mole):
    #Centers a molecule
    sum = [0,0,0]
    N = len(mole)
    result = mole
    for i in range(N):
        sum[0] += result[i][1]
        sum[1] += result[i][2]
        sum[2] += result[i][3]
    sum = np.multiply(sum, 1/N)
    #print(sum)
    for i in range(N):
        result[i][1:] = np.subtract(result[i][1:],sum)
    return result

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
                summand = elements[mole[i][0]]*elements[mole[j][0]]/np.linalg.norm(np.subtract(mole[i][1:],mole[j][1:]))
                #The offdiagonal elements are gated, such that any discussion of electronic similarity can be restricted to an atoms direct neighborhood
                if summand > gate_threshold:
                    result[i][j] = summand
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
                sum += CN[k]*(np.dot(mole[k][1:], mole[k][1:])*delta(i,j) - mole[k][1+i]*mole[k][1+j])
            result_tensor[i][j] = sum
            result_tensor[j][i] = sum
            sum = 0
    return result_tensor

def atomrep_inertia_moment(mole, representation='atomic_Coulomb'):
    if representation == 'atomic_Coulomb':
        #Calculate the inertia moments of a molecule with CN instead of masses
        #and sort them in ascending order
        w,v = np.linalg.eig(CN_inertia_tensor(mole))
        #Only the eigen values are needed, v is discarded
        moments = np.sort(w)
        return moments
    else:
        return 0


def array_compare(arr1, arr2):
    '''arr1 = [...]
    arr2 = [[...],[...],[...],...]
    Is there an approximate copy of arr1 in arr2'''
    within = False
    for i in range(len(arr2)):
        if are_close_lists(arr1, arr2[i]):
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

def bond_count(edge_layout, elements_at_index):
    '''Count all the bonds of edge_layout with the vertices defined in
    elements_at_index and return them'''
    #First: Translate all edges into bond-strings
    collect_bonds = []
    for edge in edge_layout:
        bond_vertices = [inv_elements[i] for i in np.sort([elements[elements_at_index[edge[0]]], elements[elements_at_index[edge[1]]]])]
        bond_name = bond_vertices[0]+bond_vertices[1]
        collect_bonds = np.append(collect_bonds, bond_name)
    return collect_bonds

def bestest(list, dx):
    '''Given a list and distance dx, find an integer x such that the interval [x-dx,x+dx]
    has the most elements of list in it and return the x and the indices of those elements.'''
    min_val = min(list)
    max_val = max(list)
    x_range = [i for i in range(int(min_val), int(max_val)+1)]
    count = [0 for i in x_range]
    for i in range(len(x_range)):
        for j in range(len(list)):
            if (list[j] <= x_range[i]+dx) and (list[j] >= x_range[i]-dx):
                count[i] += 1
    best_x = x_range[count.index(max(count))]
    result = []
    for i in range(len(list)):
        if (list[i] <= best_x+dx) and (list[i] >= best_x-dx):
            result.append(i)
    return best_x, result

#CLASS DEFINITION OF MoleAsGraph------------------------------------------------

class MoleAsGraph:
    def __init__(self, name, edge_layout, elements_at_index, geometry):
        '''Caution: For all methods so far, the indexing of the geometry and the graphs (i.e. all
        remaining attributes) does not have to match! Keep this in mind: They may have different
        indices!!!!!! They do not even have to be the same molecules but then the Constructor may
        print a Warning'''
        self.name = name
        self.edge_layout = edge_layout #edge_layout = [[site_index, connected site index (singular!!!)], [...,...], [...,...]]
        self.elements_at_index = elements_at_index #Which element is at which vertex number
        self.geometry = center_mole(geometry) #The usual xyz representation
        if len(np.unique(edge_layout)) == len(elements_at_index):
            self.number_atoms = len(elements_at_index)
        else:
            print('Number of vertices and number of elements do not match!')
        if len(edge_layout) != 0:
            self.max_index = max([max(sublist) for sublist in edge_layout])
        else:
            self.max_index = 0
        if len(elements_at_index) != self.max_index+1:
            print("Number of atoms does not match naming of vertices: enumerate the vertices with integers without omissions!")
        self.orbits = self.get_orbits_from_graph()
        self.equi_atoms =  self.get_equi_atoms_from_geom()
        #This part can only be used if you are sure that the indexing of atoms in the graph is the same as in the xyz
        #self.orbits = np.array(self.get_equi_atoms_from_geom(), dtype=object)

        #If the indexing of vertices does not match the indexing of geometry:
        #if not np.array([elements_at_index[i] == geometry[i][0] for i in range(self.number_atoms)]).all():
        #    print("Warning in MoleAsGraph: Indexing of atoms in attribute 'geometry' does not match indexing in attribute 'elements_at_index' of molecule "+name)

    def get_site(self,site_number):
        if site_number >= self.number_atoms:
            raise ValueError("Cannot return site with index "+str(site_number)+". Molecule only has "+str(self.number_atoms)+" atoms.")
        else:
            return self.elements_at_index(site_number)

    def get_number_automorphisms(self):
        #Prepare the graph
        g = igraph.Graph([tuple(v) for v in self.edge_layout])
        #Get all automorphisms with colored vertices according to self.elements_at_index
        automorphisms = g.get_automorphisms_vf2(color=[elements[self.elements_at_index[i]] for i in range(self.number_atoms)])
        #Get rid of all molecules without any orbits (the identity does not count):
        return len(automorphisms)


    def get_orbits_from_graph(self):
        #Prepare the graph
        g = igraph.Graph([tuple(v) for v in self.edge_layout])
        #Get all automorphisms with colored vertices according to self.elements_at_index
        automorphisms = g.get_automorphisms_vf2(color=[elements[self.elements_at_index[i]] for i in range(self.number_atoms)])
        #Get rid of all molecules without any orbits (the identity does not count):
        if len(automorphisms) == 1:
            similars = [[]]
            return similars
        else:
            '''Any element of the list of vertices = set of elements, that moves along
            a fixed path upon the group acting on the set is part of an orbit. Since we know
            all allowed automorphisms, we can check for all the possible vertices at positions
            that are reached through the permutations = automorphisms'''
            #print('----------------')
            #print(self.name)
            #print(automorphisms)
            #print('----------------')
            similars = np.array(np.zeros(self.number_atoms), dtype=object)
            for i in range(self.number_atoms):
                orbit = []
                for j in range(len(automorphisms)):
                    orbit = np.append(orbit, automorphisms[j][i])
                orbit = np.unique(orbit)
                #print(orbit)
                similars[i] = orbit
            #Delete all similars which include only one atom:
            num_similars = 0
            while num_similars < len(similars):
                if len(similars[num_similars])>1:
                    num_similars += 1
                else:
                    similars = np.delete(similars, num_similars, axis = 0)
            #Unique the list:
            sites = [list(map(int,x)) for x in set(tuple(x) for x in similars)]
        return sites

    def get_equi_atoms_from_geom(self):
        mole = self.geometry.copy()
        #Sort CN and group everything together that is not farther than tolerance
        CN = Coulomb_neighborhood(center_mole(mole))
        indices = np.argsort(CN).tolist()
        indices2 = np.copy(indices).tolist()
        lst = []
        similars = []
        for i in range(len(indices)-1):
            if i not in indices2:
                continue
            lst.append(indices[i])
            indices2.remove(i)
            for j in range(i+1,len(indices)):
                if are_close_scalars(CN[indices[i]],CN[indices[j]]):
                    lst.append(indices[j])
                    indices2.remove(j)
            similars.append(lst)
            lst = []
        #Delete all similars which include only one atom:
        num_similars = 0
        while num_similars < len(similars):
            if len(similars[num_similars])>1:
                num_similars += 1
            else:
                del similars[num_similars]
        return similars

    def get_energy_NN(self):
        #Calculate the nuclear energy of the molecule
        sum = 0
        for i in range(self.number_atoms):
            for j in range(i+1,self.number_atoms):
                sum += elements[self.geometry[i][0]]*elements[self.geometry[j][0]]/np.linalg.norm(np.subtract(self.geometry[i][1:],self.geometry[j][1:]))
        return sum*0.529177210903 #Result needs to be in Ha, and the length has been in Angstrom

    def get_molecular_norm(self):
        return np.linalg.norm(CN_inertia_tensor(self.geometry.copy()))

    def print_atomic_norms(self, input_PathToFile):
        N = self.number_atoms
        file = open(input_PathToFile, 'r')
        data = file.read()
        file.close()
        for i in range(N):
            result = 0 #=norm of the row/column of the Coulomb matrix
            for j in range(N):
                if (j == i):
                    result += 0.5*pow(elements[self.geometry[i][0]], 2.4)
                else:
                    result += elements[self.geometry[i][0]]*elements[self.geometry[j][0]]/np.linalg.norm(np.subtract(self.geometry[i][1:],self.geometry[j][1:]))
            #Get the number of bonds (including hydrogen for hybridisation purposes)
            Num = int(data.splitlines(False)[0]) #number of atoms including hydrogen
            smiles = data.splitlines(False)[Num+3].split('\t')[0]
            #mol = read_smiles(smiles) #include hydrogen for hybridisation counting
            #graph = nx.to_networkx_graph(mol)
            #degree = graph.degree([i])[1]
            print(self.name+'\t'+self.geometry[i][0]+'\t'+str(i)+'\t'+smiles+'\t'+str(result))
            #Name   Chemical Element    Index   SMILES  Norm

    def get_total_energy(self, basis=basis):
        #Make sure that the hydrogens are parsed, too!!!!!!
        atom_string = ''
        for i in range(self.number_atoms): #get the atoms and their coordinates
            atom_string += self.geometry[i][0]
            for j in [1,2,3]:
                atom_string += ' ' + str(self.geometry[i][j])
            atom_string += '; '
        mol = gto.M(
            verbose = 0,
            atom = atom_string[:-2],  #Last '; ' was removed; in Angstrom
            basis = basis,
            symmetry = False,
            unit = 'Angstrom'
        )
        mf = scf.HF(mol).run()
        energy = mf.e_tot
        return energy

    def get_Hessian(self):
        #Make sure that the hydrogens are parsed, too!!!!!!
        atom_string = ''
        for i in range(self.number_atoms): #get the atoms and their coordinates
            atom_string += self.geometry[i][0]
            for j in [1,2,3]:
                atom_string += ' ' + str(self.geometry[i][j])
            atom_string += '; '
        mol = gto.M(
            verbose = 0,
            atom = atom_string[:-2],  #Last '; ' was removed; in Angstrom
            basis = basis,
            symmetry = False,
            unit = 'Angstrom'
        )
        mf = mol.RHF().run()
        Hessian = mf.Hessian().kernel()
        return Hessian

    def fill_hydrogen_valencies(self, input_PathToFile):
        '''If the xyz file from which this molecule originates is known,
        the valencies can be filled with the hydrogens as given in the file.
        Add the geometric information line by line with an offset subtracted!!!!
        and for each line, make one vertex and one edge to the closest heavy atom'''
        #check if file is present
        if os.path.isfile(input_PathToFile):
            #open text file in read mode
            f = open(input_PathToFile, "r")
            data = f.read()
            f.close()
        else:
            print('File', input_PathToFile, 'not found.')
            return 0
        #Initalize
        name = self.name
        new_geometry = self.geometry.copy()
        new_elements_at_index = self.elements_at_index.copy()
        new_edge_layout = self.edge_layout.copy()
        N_heavy = self.number_atoms #number of previously give atoms, all heavy
        N = int(data.splitlines(False)[0]) #number of atoms including hydrogen
        #Compare self.geometry to the first entry of the file (= line 2) -> obtain offset
        line2 = data.splitlines(False)[2].split('\t')
        coord_orig = [float(line2[1]),float(line2[2]),float(line2[3])]
        offset = [0,0,0]
        for i in [0,1,2]:
            offset[i] = coord_orig[i] - self.geometry[0][i+1]
        for i in range(2,N+2): #get only the hydrogens and their coordinates
            line = data.splitlines(False)[i]
            #Check for hydrogen specifically
            x = line.split('\t')
            if x[0] != 'H':
                continue
            else:
                new_geometry.append(['H', float(x[1])-offset[0],float(x[2])-offset[1],float(x[3])-offset[2]])
                new_elements_at_index.append('H')
            #Now: find the index of the atom with the shortest distance
            shortest_distance = 100000
            for j in range(N_heavy):
                distance = np.linalg.norm(np.subtract(self.geometry[j][1:],new_geometry[-1][1:]))
                if distance < shortest_distance:
                    shortest_distance = distance
                    index_of_shortest = j
            new_edge_layout.append([int(index_of_shortest),len(new_geometry)-1])
        return MoleAsGraph(name, new_edge_layout, new_elements_at_index, new_geometry)


#MoleAsGraph EXAMPLES-----------------------------------------------------------
anthracene = MoleAsGraph('Anthracene',
                        [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[6,7],[7,8],[8,9],[9,5],[7,10],[10,11],[11,12],[12,13],[13,8]],
                        ['C','C','C','C','C','C','C','C','C','C','C','C','C','C'],
                        [['C', 0,0,0.5], ['C', 0,0.8660254037844386467637231707,1], ['C', 0,2*0.8660254037844386467637231707,0.5],
                        ['C', 0,2*0.8660254037844386467637231707,-0.5], ['C', 0,0.8660254037844386467637231707,-1], ['C', 0,0,-0.5],
                        ['C', 0,-0.8660254037844386467637231707,1], ['C', 0,-2*0.8660254037844386467637231707,0.5], ['C', 0,-2*0.8660254037844386467637231707, -0.5], ['C', 0,-0.8660254037844386467637231707,-1],
                        ['C', 0,-3*0.8660254037844386467637231707,1], ['C', 0,-4*0.8660254037844386467637231707,0.5], ['C', 0,-4*0.8660254037844386467637231707, -0.5], ['C', 0,-3*0.8660254037844386467637231707,-1]])
benzene = MoleAsGraph(  'Benzene',
                        [[0,1],[1,2],[2,3],[3,4],[4,5],[0,5]],
                        ['C','C','C','C','C','C'],
                        [['C', 0,0,1], ['C', 0,0.8660254037844386467637231707,0.5], ['C', 0,0.8660254037844386467637231707,-0.5],
                        ['C', 0,0,-1], ['C', 0,-0.8660254037844386467637231707,-0.5], ['C', 0,-0.8660254037844386467637231707,0.5]])

cube = MoleAsGraph(     'Cube',
                        [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],
                        ['C','C','C','C','C','C','C','C'],
                        [['C',0,0,0],['C',0,1,0],['C',0,1,1],['C',0,0,1],['C',1,0,0],['C',1,1,0],['C',1,1,1],['C',1,0,1]])

naphthalene = MoleAsGraph(  'Naphthalene',
                            [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[6,7],[7,8],[8,9],[9,5]],
                            ['C','C','C','C','C','C','C','C','C','C'],
                            [['C', 0,0,0.5], ['C', 0,0.8660254037844386467637231707,1], ['C', 0,2*0.8660254037844386467637231707,0.5],
                            ['C', 0,2*0.8660254037844386467637231707,-0.5], ['C', 0,0.8660254037844386467637231707,-1], ['C', 0,0,-0.5],
                            ['C', 0,-0.8660254037844386467637231707,1], ['C', 0,-2*0.8660254037844386467637231707,0.5], ['C', 0,-2*0.8660254037844386467637231707, -0.5], ['C', 0,-0.8660254037844386467637231707,-1]])

#PARSER FUNCTIONS FOR QM9-------------------------------------------------------
def energy_PySCF_from_QM9(input_PathToFile, basis=basis):
    if os.path.isfile(input_PathToFile):
        #open text file in read mode
        f = open(input_PathToFile, "r")
        data = f.read()
        f.close()
    else:
        print('File', input_PathToFile, 'not found.')
        return 0
    N = int(data.splitlines(False)[0]) #number of atoms including hydrogen
    atom_string = ''
    for i in range(2,N+2): #get the atoms and their coordinates
        line = data.splitlines(False)[i]
        x = line.split('\t')
        atom_string += x[0] + ' ' + x[1] + ' ' + x[2] + ' '  + x[3] + '; '
    #print(atom_string[:-2])
    mol = gto.M(
        verbose = 0,
        atom = atom_string[:-2],  #Last '; ' was removed; in Angstrom
        basis = basis,
        symmetry = False,
    )
    mf = scf.HF(mol)
    energy = mf.kernel()
    return energy

def parse_QM9toMAG(input_PathToFile, with_hydrogen = False):
    '''MoleAsGraph instance returned'''
    #check if file is present
    if os.path.isfile(input_PathToFile):
        #open text file in read mode
        f = open(input_PathToFile, "r")
        data = f.read()
        f.close()
    else:
        print('File', input_PathToFile, 'not found.')
        return 0
    #Get the name of the molecule
    MAG_name = input_PathToFile.split('/')[-1].split('.')[0]
    N = int(data.splitlines(False)[0]) #number of atoms including hydrogen
    #Get the geometry of the molecule
    mole = []
    N_heavyatoms = N
    for i in range(2,N+2): #get the atoms and their coordinates
        line = data.splitlines(False)[i]
        if not with_hydrogen and line.split('\t')[0] == 'H':
            N_heavyatoms -= 1
            continue
        else:
            symbol = line.split('\t')[0]
            x = float(line.split('\t')[1].strip())
            y = float(line.split('\t')[2].strip())
            z = float(line.split('\t')[3].strip())
            mole.append([symbol,x,y,z])
    #print(mole)
    #Get the edges of the molecule as a graph
    network = read_smiles(data.splitlines(False)[N+3].split('\t')[0])
    edge_layout = [list(v) for v in network.edges()]
    elements_at_index = [v[1] for v in network.nodes(data='element')]
    return MoleAsGraph(MAG_name, edge_layout, elements_at_index, mole)


#ALL HIGHER LEVEL FUNCTIONS WITH VARIOUS DEPENDENCIES---------------------------
def geomAE(graph, m=[2,2], dZ=[1,-1], debug = False, chem_formula = True, get_all_energies = False,  take_hydrogen_data_from=''):
    '''Returns the number of alchemical enantiomers of mole that can be reached by
    varying m[i] atoms in mole with identical Coulombic neighborhood by dZ[i].
    In case of method = 'geom', log = 'verbose', the path for the xyz data of the hydrogens is
    needed to fill the valencies.'''
    N_m = len(m)
    N_dZ = len(dZ)
    start_time = time.time()
    mole = np.copy(np.array(graph.geometry, dtype=object))
    N = graph.number_atoms
    equi_atoms = np.copy(np.array(graph.equi_atoms, dtype=object))

    if N < np.sum(m):
        raise ValueError("Too less atoms in molecule for sum of to be transmuted atoms.")
    if (np.sum(np.multiply(m,dZ)) != 0):
        raise ValueError("Netto change in charge must be 0")
    if 0 in dZ:
        raise ValueError("0 not allowed in array dZ")
    if N_m != N_dZ:
        raise ValueError("Number of changes and number of charge values do not match!")
    if len(equi_atoms) == 0:
        return 0

    '''All of these sites in each set need to be treated simultaneously. Hence, we
    flatten the array. However, we later need to make sure that only each set fulfills
    netto charge conservation. This is why similars is initalized'''
    similars = list(itertools.chain(*equi_atoms))
    '''This is the list of all atoms which can be transmuted simultaneously.
    Now, count all molecules which are possible excluding mirrored or rotated versions'''
    count = 0
    #Initalize empty array temp_mole for all configurations.
    temp_mole = []
    #Get necessary atoms listed in equi_atoms
    for i in range(len(similars)):
        temp_mole.append(mole[similars[i]])
    #Make sure, that m1+m2 does no exceed length of similars
    if np.sum(m) > len(similars):
        return 0

    '''Now: go through all combinations of transmuting m atoms of set similars
    by the values stored in dZ. Then: compare their atomrep_inertia_moments
    and only count the unique ones'''
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
    mole_config_unfiltered = np.array(np.meshgrid(*atomwise_config.tolist(), copy=False)).T.reshape(-1,len(similars))
    #Initalize a molecule configuration:
    mole_config = np.zeros((1,len(similars)),dtype='int')
    mole_config = np.delete(mole_config, 0, axis = 0)

    for k in range(len(mole_config_unfiltered)):
        # m1 sites need to be changed by dZ1, m2 sites need to be changed by dZ2, etc...
        if np.array([(m[v] == (np.subtract(mole_config_unfiltered[k],standard_config) == dZ[v]).sum()) for v in range(N_dZ)]).all():
            '''Check that the netto charge change in every set of equivalent atoms is 0'''
            pos = 0
            for i in range(len(equi_atoms)):
                sum = 0
                #This loop has to start where the last one ended
                for j in range(pos,pos+len(equi_atoms[i])):
                    sum += mole_config_unfiltered[k][j] - standard_config[j]
                pos += len(equi_atoms[i])
                if sum != 0:
                    break
                if (sum == 0) and (i == len(equi_atoms)-1):
                    #print(mole_config_unfiltered[k])
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

    '''Alchemical enantiomers are those molecules which do not transmute
    into themselves (or its spatial enantiomer) under the mirroring in charge,
    i.e. if one adds the inverted configuration of transmutations to twice the molecule,
    its CIM has changed.'''

    config_num = 0
    while config_num <len(Total_CIM):
        current_config = np.zeros(len(similars),dtype=object)
        for i in range(len(similars)):
            current_config[i] = elements[Total_CIM[config_num][0][i][0]]
        mirror_config = 2*standard_config - current_config
        #print(current_config)
        #print(mirror_config)
        #print(Total_CIM[config_num][1]) #The CIM of current_config
        for i in range(len(similars)):
            temp_mole[i][0] = inv_elements[mirror_config[i]]
        #print(atomrep_inertia_moment(temp_mole))
        #print('----------')
        if are_close_lists(Total_CIM[config_num][1], atomrep_inertia_moment(temp_mole)):
            Total_CIM = np.delete(Total_CIM, config_num, axis = 0)
        else:
            config_num += 1

    '''All is done. Now, print the remaining configurations and their contribution
    to count.'''
    count += len(Total_CIM)

    if get_all_energies and len(Total_CIM) > 0 and take_hydrogen_data_from != '':
        '''Explicitly calculate the energies of all the configurations in Total_CIM[0],
        but do so by returning a list of MoleAsGraph objects'''
        for i in range(len(Total_CIM)):
            #Create temporary MoleAsGraph object:
            dummy_elements_at_index = np.array(graph.elements_at_index, copy=True, dtype=object)
            dummy_geometry = np.array(graph.geometry, copy=True, dtype=object)
            num = 0
            for j in similars:
                dummy_elements_at_index[j] = Total_CIM[i][0][num][0]
                dummy_geometry[j][0] = Total_CIM[i][0][num][0]
                num += 1
            dummy_mole = MoleAsGraph('dummy', graph.edge_layout, dummy_elements_at_index.tolist(), dummy_geometry.tolist()).fill_hydrogen_valencies(take_hydrogen_data_from)
            #print(dummy_mole.geometry)
            dummy_energy_total = dummy_mole.get_total_energy()
            dummy_energy_NN = dummy_mole.get_energy_NN()
            print("Total energy: "+str(dummy_energy_total)+"\tNuclear energy: "+str(dummy_energy_NN)+"\tElectronic energy: "+str(dummy_energy_total-dummy_energy_NN))

    if debug == True:
        num_sites = len(similars)
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
                xs[j] = Total_CIM[i][0][j][1]
                ys[j] = Total_CIM[i][0][j][2]
                zs[j] = Total_CIM[i][0][j][3]
                n[j] = Total_CIM[i][0][j][0]
            ax.scatter(xs, ys, zs, marker='o', facecolor='black')
            #print(Total_CIM[i][0][1][0])
            for j, txt in enumerate(n):
                ax.text(xs[j], ys[j], zs[j], n[j])
            #ax.set_xlabel('X')
            #ax.set_ylabel('Y')
            #ax.set_zlabel('Z')
        plt.show()
    return count

def nautyAE(graph, m = [2,2], dZ=[+1,-1], debug = False, chem_formula = True, bond_energy_rules = False):
    #graph is an instance of the MoleAsGraph class
    #dZ and m are each arrays that include the transmutations and their number
    start_time = time.time()
    m = np.array(m)
    dZ = np.array(dZ)
    N_m = len(m)
    N_dZ = len(dZ) #number of transmutations = number of colors-1
    max_node_number = np.amax(graph.edge_layout)+1
    N = len(np.unique(graph.edge_layout))
    if len(graph.orbits) == 0:
        #There are no orbits, so there are no AE
        return 0
    if 0 in dZ:
        raise ValueError("0 not allowed in array dZ")
    if N_dZ < 2:
        raisorbitsor("Number of transmutations must be at least 2")
    if N_m != N_dZ:
        raise ValueError("Number of transmutations does not match their number!")
    #Check for overall charge conservation
    if (np.sum(np.multiply(m,dZ)) != 0):
        raise ValueError("Netto change in charge must be 0")
    if N == 1:
        raise ValueError("Graph needs to have at least 2 vertices.")
    if N != max_node_number:
        raise ValueError("Enumerate the vertices with integers without omissions")
    if N_dZ != len(np.unique(dZ)):
        raise ValueError("Equal values in multiple entries")
    if N < np.sum(m):
        raise ValueError("Too less vertices in graph for total number of to be transmuted atoms.")

    #Use graph-based algorithm nauty27r1; build the string command to be passed to the bash
    #Command for the standard case looks like:
    #echo 'n=10;0:1;1:2;2:3;3:4;4:5;5:0;0:6;6:7;7:8;8:9;9:5;' | /home/simon/Desktop/nauty27r1/dretog -q | /home/simon/Desktop/nauty27r1/vcolg -q -T -m3 |
    #awk '{count1 = 0; count2 = 0; for (i=3; i<13; i++){if ($i == 1) count1++; else if ($i == 2) count2++;} if ((count1 == 2) && (count2 == 2)) print}'
    #This immediatly checks wether charge conservation and the correct number of colors are given.
    command = "echo 'n=" + str(N) + ";"
    for i in range(len(graph.edge_layout)):
        command += str(graph.edge_layout[i][0]) +":" + str(graph.edge_layout[i][1]) + ";"
    command += "' | "+PathToNauty27r1+"dretog -q | "+PathToNauty27r1+"vcolg -q -T -m"
    command += str(N_dZ+1)
    command += " | awk '{"
    for i in range(N_dZ):
        command += "count"+str(i+1)+"=0; "
    command += "for (i=3; i<" + str(N+3) + "; i++){if ($i == 1) count1++; else if ($i == 2) count2++;"
    for color in range(2,N_dZ):
        command += " else if ($i == " + str(color+1) + ") count" + str(color+1) + "++;"
    command += "} if ((count1 == " + str(m[0]) + ") && (count2 == " + str(m[1]) + ")"
    for color in range(2,N_dZ):
        command += " && (count" + str(color+1) + " == " + str(m[color]) + ")"
    command += ") print}'"
    output = os.popen(command).read()
    #print(command,"\n")
    #Color 0 is the standard, colors 1,2,3,etc. are the transmutations
    #Parse output to an array
    num_lines = output.count('\n')
    graph_config = np.zeros((num_lines),dtype=object)
    for i in range(num_lines):
        line = output.splitlines(False)[i]
        #Get rid of everything after '  ':
        line = line.split('  ')[0]
        #Parse numbers to integer array:
        numbers = [int(j) for j in line.split(' ')]
        #Delete first two elements
        numbers = np.delete(numbers, (0,1), axis = 0)
        graph_config[i] = numbers

    '''The parsed array needs to fulfill two things:
    1) Is the netto charge within all orbits conserved? An orbit is the set of all
    equivalent atoms in the geometry-based method.
    2) Is the alchemically mirrored graph not itself?
    This is the same as asking if graph and mirror are isomorphic which can only happen
    if and only if for each pair of m[i],dZ[i] there exists a pair m[j],-dZ[j]
    that is equal.'''

    #print(graph_config)
    #Answering question one:
    config_num = 0
    while config_num < len(graph_config):
        for i in range(len(graph.orbits)):
            sum = 0
            for j in graph.orbits[i]:
                #Avoid getting the last element
                if graph_config[config_num][j] != 0:
                    sum += dZ[graph_config[config_num][j]-1]
            if sum != 0:
                graph_config = np.delete(graph_config, config_num, axis = 0)
                break
            if i == len(graph.orbits)-1:
                config_num += 1
    #print(graph_config)
    #Prepare some dicts
    color2dZ = {0:0}
    for i in range(N_dZ):
        color2dZ[i+1] = dZ[i]
    dZ2color = {v: k for k, v in color2dZ.items()}
    #Answering the second question:
    '''Find out if all those graphs are able to self mirror'''
    self_mirrorable = np.array([-i in dZ for i in dZ]).all()
    if self_mirrorable:
        '''Use igraph's isomorphic-function to delete graphs which are isomorphic
        after transmutation'''
        #Prepare the graph
        g1 = igraph.Graph([tuple(v) for v in graph.edge_layout])
        config_num = 0
        while config_num < len(graph_config):
            if g1.isomorphic_vf2(color1=graph_config[config_num], color2=[dZ2color[-color2dZ[graph_config[config_num][i]]] for i in range(N)]):
                graph_config = np.delete(graph_config, config_num, axis = 0)
            else:
                config_num += 1
    count = len(graph_config)


    #-----------------------------Very optional, very prelimenary-------------------------
    #Find all the rules for bond energies between the AEs in one pair:
    if bond_energy_rules == True:
        #Initalize vector to hold all coefficients of all possible bonds
        #Find all possible nuclear charges:
        chem_elem = np.copy(graph.elements_at_index)
        transmutations = dZ
        transmutations = np.append(transmutations,0)
        available_charges = np.unique([elements[chem_elem[i]] for i in range(len(chem_elem))])
        possible_charges = np.unique([[i+j for i in available_charges] for j in transmutations])
        #Sort the vector
        possible_charges = np.sort(possible_charges)
        #Create the vector of all possible combinations
        possible_bonds = []
        for i in range(len(possible_charges)):
            for j in range(i, len(possible_charges)):
                possible_bonds = np.append(possible_bonds, inv_elements[possible_charges[i]]+inv_elements[possible_charges[j]])
        #print(possible_bonds)
        ref_bonds = bond_count(graph.edge_layout, graph.elements_at_index)
        multip = np.zeros((len(possible_bonds)), dtype='int')
        for j in range(len(ref_bonds)):
            for k in range(len(possible_bonds)):
                if ref_bonds[j] == possible_bonds[k]:
                    multip[k] += 1

        rule = []
        for i in range(count):
            #We need the bonds of both molecules of the pair of AE!!!
            bonds1 = bond_count(graph.edge_layout, [inv_elements[elements[graph.elements_at_index[j]]+color2dZ[graph_config[i][j]]] for j in range(N)])
            bonds2 = bond_count(graph.edge_layout, [inv_elements[elements[graph.elements_at_index[j]]-color2dZ[graph_config[i][j]]] for j in range(N)])
            #Count the multiplicities of each bond and store them in a possible_bonds-like vector:
            multip1 = np.zeros((len(possible_bonds)), dtype='int')
            multip2 = np.zeros((len(possible_bonds)), dtype='int')
            for j in range(len(bonds1)):
                for k in range(len(possible_bonds)):
                    if bonds1[j] == possible_bonds[k]:
                        multip1[k] += 1
                    if bonds2[j] == possible_bonds[k]:
                        multip2[k] += 1
            #Prelimenary printout:
            out =''
            sign = 1
            first_nonzero = False
            for i in range(len(multip1)):
                if multip1[i]-multip2[i] != 0:
                    if first_nonzero == False:
                        first_nonzero = True
                        if multip1[i]-multip2[i] < 0:
                            sign = -1
                    out += '  '+str(sign*(multip1[i]-multip2[i]))+' E_'+str(possible_bonds[i])
            if len(out.strip()) != 0:
                rule = np.append(rule, out+' = 0')
        final_set = np.unique(rule)
        for r in final_set:
            print(r)
    if debug == True:
        print('---------------')
        print("Time:", (time.time() - start_time),"s")
        print('---------------')
        print('Alchemical Enantiomers:')
        for i in range(count):
            x = [inv_elements[elements[graph.elements_at_index[j]]+color2dZ[graph_config[i][j]]] for j in range(N)]
            print(str(sum_formula(x)) + ": " + str(x))
    return count

def uncolor(graph):
    '''Find the most symmetric reference molecule by setting all atoms to the
    rounded average integer charge.'''
    tot_nuc_charge = 0
    for i in range(graph.number_atoms):
        tot_nuc_charge += elements[graph.elements_at_index[i]]
    average_element = inv_elements[int(tot_nuc_charge/graph.number_atoms)]
    new_elements = np.empty((graph.number_atoms), dtype='str')
    new_geometry = graph.geometry
    #Explicitly set the average element:
    #average_element = 'C'
    for i in range(graph.number_atoms):
        new_elements[i] = average_element
        new_geometry[i][0] = average_element
    return MoleAsGraph('isoatomic'+graph.name, graph.edge_layout,new_elements,new_geometry)

def Find_theoAEfromgraph(N, dZ_max):
    '''Find the theoretical possible upper limit for the number of possible molecules. Use
    nauty's geng to generate all possible connected graphs with at least 1 degree and at most
    4 (organic molecules only) and pipe that into vcolg'''
    count = 0
    N_dZ = 1+2*dZ_max
    command = PathToNauty27r1+"geng -c "+str(N)+" -d1 -D4 -q | "+PathToNauty27r1+"vcolg -q -T -m"+str(N_dZ)
    #print(command)
    output = os.popen(command).read()
    #--------------------------------------------------------------------------
    num_lines = output.count('\n')
    element_config = np.empty((num_lines),dtype='str')
    #Use dynamic stepsizes for nice random batching
    batching = 1
    if N > 4 or num_lines > 5000:
        batching = N*N*(int(num_lines/10000)+1)
    for i in range(0,num_lines, batching):
        line = output.splitlines(False)[i]
        #Split at '  ':
        colors = line.split('  ')[0]
        noding = line.split('  ')[1]

        #Parse numbers to integer array and create a fictional molecule
        chem_elements = np.array([inv_elements[int(j)+6] for j in colors.split(' ')],dtype=object)
        edges = [int(j) for j in noding.split(' ')]
        #Delete first two elements
        chem_elements = np.delete(chem_elements, (0,1), axis = 0)
        #Initalize the edge_layout of the fictional molecule
        edge_layout = [[edges[2*j], edges[2*j+1]] for j in range(int(len(edges)/2))]
        #print('-------------------')
        #print(chem_elements)
        #print(edge_layout)
        #Create the fictional molecule as MoleAsGraph object and call Find_AEfromref
        fict_mole = MoleAsGraph('spamandeggs',edge_layout ,chem_elements.tolist(), None)
        num_AE = Find_AEfromref(fict_mole, dZ_max=dZ_max, log = 'quiet', method = 'graph', bond_energy_rules = False)
        if num_AE > 0:
            count += 1
    print('Number of atoms: '+str(N)+'\tdZ_max: '+str(dZ_max)+'\tPossibles / % : '+str(count*batching*100/(num_lines)))

def Find_AEfromref(graph, dZ_max = 3, log = 'normal', method = 'graph', take_hydrogen_data_from = ''):
    '''In case of method = 'geom', log = 'verbose', the path for the xyz data of the hydrogens is
    needed to fill the valencies.'''
    with_energies = False
    with_bond_energy_rules = False
    if method == 'graph' and log == 'verbose':
        with_bond_energy_rules = True
        print('----------------------------')
        print('Bond energy rules:\n')
    if method == 'geom' and log == 'verbose':
        with_energies = True
        print('----------------------------')
        print('Energies of AEs in Eh:\n')
    dZ_all = np.copy(dZ_possibilities)
    m_all = np.copy(m_possibilities)
    start_time = time.time()
    #Get rid of all dZs > dZ_max:
    num = 0
    while num < len(dZ_all):
        if np.max(dZ_all[num]) > dZ_max:
            m_all = np.delete(m_all, num, axis = 0)
            dZ_all = np.delete(dZ_all, num, axis = 0)
        else:
            num += 1
    #Check if they are overall netto charge is conserved
    #for i in range(len(m_all)):
    #    print(np.array([dZ_all[i][j]*m_all[i][j] for j in range(len(m_all[i]))]).sum())
    #Get rid of all m's with more changes than atoms in the molecule:
    if len(graph.orbits) != 0:
        num = 0
        available_sites = len(np.hstack(graph.orbits).ravel())
        while num < len(m_all):
            if np.sum(m_all[num]) > available_sites:
                m_all = np.delete(m_all, num, axis = 0)
                dZ_all = np.delete(dZ_all, num, axis = 0)
            else:
                num += 1
    else:
        m_all = [[]]
        dZ_all = [[]]
    #For plotting: save number of transmuted atoms num_trans and time
    #print(m_all)
    #print(dZ_all)
    num_trans = []
    times = []
    total_number = 0
    if log == 'normal':
        print('\n'+ graph.name + '; method = ' + method + '\n------------------------------')

    for i in range(len(m_all)):
        random_config = np.zeros((graph.number_atoms))
        pos = 0
        for j in range(len(m_all[i])):
            for k in range(m_all[i][j]):
                random_config[pos] = dZ_all[i][j]
                pos += 1
        chem_form = str(sum_formula([inv_elements[elements[graph.elements_at_index[v]]+random_config[v]] for v in range(graph.number_atoms)]))
        if log == 'normal' or log == 'verbose':
            print(chem_form)
        m_time = time.time()
        if method == 'graph':
            x = nautyAE(graph, m_all[i], dZ_all[i], debug= False, chem_formula = True, bond_energy_rules = with_bond_energy_rules)
        if method == 'geom':
            x = geomAE(graph, m_all[i], dZ_all[i], debug= False, chem_formula = True, get_all_energies = with_energies, take_hydrogen_data_from= take_hydrogen_data_from)
        if log == 'normal' or log == 'verbose':
            print('Time:', time.time()-m_time)
            print('Number of AEs:', x,'\n')
        num_trans.append(np.sum(m_all[i]))
        times.append(time.time()-m_time)
        total_number += x
    if log == 'verbose' or log == 'normal':
        print('----------------------------')
        print(graph.name)
        print('Total time:', time.time()-start_time)
        print('Total number of AEs:', total_number)
        print('Number of transmuted atoms:', list(num_trans))
        print('Time:', list(times))
        print('----------------------------')
    if log == 'sparse':
        print(graph.name + '\t' + str(time.time()-start_time) + '\t' + str(graph.number_atoms) + '\t' + str(total_number))
    if log == 'quiet':
        return total_number

def Find_reffromtar(graph, dZ_max = 3, method = 'graph', log = 'normal'):
    '''Find the most symmetric reference molecule, not all of them. Most symmetric
    means here the least amount of atoms are not part of an orbit/equivalent set. Less
    symmetric is always possible and included in most symmetric (and appears specifically
    when searching for AEs)'''
    #Initalize original state:
    if method == 'graph':
        chem_config = np.copy(graph.elements_at_index)
    if method == 'geom':
        chem_config = np.array([graph.geometry[i][0] for i in range(graph.number_atoms)], copy=True)
    Geom = np.array(graph.geometry, copy=True, dtype=object)
    '''Find all orbits/equivalent sets if the molecule is colorless/isoatomic
    and save them in a list of lists called sites. This is the reason why we
    dropped the more precise terms "orbit" and "similars"'''
    if method == 'graph':
        #This is basically identical with MoleAsGraph's get_orbits_from_graph method
        g = igraph.Graph([tuple(v) for v in graph.edge_layout])
        #Get all automorphisms with uncolored vertices
        automorphisms = g.get_automorphisms_vf2()
        if len(automorphisms) == 1:
            sites = [[]]
        else:
            similars = np.array(np.zeros(graph.number_atoms), dtype=object)
            for i in range(graph.number_atoms):
                all_orbits = []
                for j in range(len(automorphisms)):
                    all_orbits = np.append(all_orbits, automorphisms[j][i])
                all_orbits = np.unique(all_orbits)
                #print(all_orbits)
                similars[i] = all_orbits
            #Unique the list and obtain the orbits of the uncolored graph
            unique_similars = np.array([list(x) for x in set(tuple(x) for x in similars)], dtype=object)
            sites = np.array([np.array(v) for v in unique_similars], dtype=object)
            #Delete all similars which include only one atom:
            num_similars = 0
            while num_similars < len(sites):
                if len(sites[num_similars])>1:
                    num_similars += 1
                else:
                    sites = np.delete(sites, num_similars, axis = 0)
    if method == 'geom':
        #This is basically the same as MoleAsGraph's get_equi_atoms_from_geom method
        #Change all atoms to be of the same chemical element
        for i in range(len(Geom)):
            Geom[i][0] = 'C'
        #print(Geom)
        CN = Coulomb_neighborhood(Geom)
        indices = np.argsort(CN).tolist()
        indices2 = np.copy(indices).tolist()
        lst = []
        sites = []
        for i in range(len(indices)-1):
            if i not in indices2:
                continue
            lst.append(indices[i])
            indices2.remove(i)
            for j in range(i+1,len(indices)):
                if are_close_scalars(CN[indices[i]],CN[indices[j]]):
                    lst.append(indices[j])
                    indices2.remove(j)
            sites.append(lst)
            lst = []
        #Delete all similars which include only one atom:
        num_similars = 0
        while num_similars < len(sites):
            if len(sites[num_similars])>1:
                num_similars += 1
            else:
                del sites[num_similars]
    '''We want to maximize the number of elements per orbit/equivalent set. Use bestest()'''
    for alpha in sites:
        if len(alpha) == 0:
            break
        #Get the colors/chemical elements of this orbit/equivalent set
        if method == 'graph':
            nodes = [elements[graph.elements_at_index[int(i)]] for i in alpha]
        if method == 'geom':
            nodes = [elements[graph.geometry[int(i)][0]] for i in alpha]
        #print(alpha)
        #print(nodes)
        #Initalize optimum in this orbit/equivalent set and the indices
        opt, indices = bestest(nodes, dZ_max)
        vertices = [int(alpha[i]) for i in indices] #We do not need the internal indexing of bestest
        #print(opt)
        #print(vertices)
        #Update chem_config
        if len(vertices) > 1:
            for i in vertices:
                chem_config[i] = inv_elements[opt]
    '''Now discard everything that transmutes into itself upon mirroring of the charges'''
    if method == 'graph':
        #Initalize graph
        g = igraph.Graph([tuple(v) for v in graph.edge_layout])
        #Initalize target, reference and mirror configuration and netto charge
        reference_config = [elements[np.copy(chem_config)[i]] for i in range(graph.number_atoms)]
        target_config = [elements[np.copy(graph.elements_at_index)[i]] for i in range(graph.number_atoms)]
        mirror_config = np.copy(target_config)
        netto_charge = 0
        #print(reference_config)
        #print(target_config)
        for i in range(graph.number_atoms):
            mirror_config[i] = 2*reference_config[i] - target_config[i]
            netto_charge += reference_config[i] - target_config[i]
        #print(mirror_config)
        if g.isomorphic_vf2(color1=target_config, color2=mirror_config) or (netto_charge != 0):
            #If yes, wipe chem_config such that the original molecule is returned
            chem_config = np.copy(graph.elements_at_index)
    if method == 'geom':
        #Initalize target, reference and mirror configuration and netto chage
        reference_config = np.array(graph.geometry, copy=True, dtype=object)
        target_config = np.array(graph.geometry, copy=True, dtype=object)
        mirror_config = np.array(graph.geometry, copy=True, dtype=object)
        netto_charge = 0
        for i in range(graph.number_atoms):
            reference_config[i][0] = chem_config[i]
            mirror_config[i][0] = inv_elements[2*elements[reference_config[i][0]] - elements[target_config[i][0]]]
            netto_charge += elements[reference_config[i][0]] - elements[target_config[i][0]]
        #Test if the target is its own mirror or charge is not conserved:
        if are_close_lists(atomrep_inertia_moment(target_config), atomrep_inertia_moment(mirror_config)) or (netto_charge != 0):
            #If yes, wipe chem_config such that the original molecule is returned
            chem_config = np.array([graph.geometry[i][0] for i in range(graph.number_atoms)], copy=True)
    #Return a MoleAsGraph object
    for i in range(len(chem_config)):
        Geom[i][0] = chem_config[i]
    if log == 'normal':
        print('Name: reffrom'+str(graph.name))
        print('Edge layout: '+str(graph.edge_layout))
        print('Elements at index: '+str(chem_config))
        print('GEOMETRY:\n' +str(Geom))
    return MoleAsGraph('reffrom'+graph.name, graph.edge_layout,chem_config.tolist(), Geom.tolist())
