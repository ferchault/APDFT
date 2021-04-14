import numpy as np
import igraph
from inertiacount import Coulomb_neighborhood, array_compare, CN_inertia_tensor, center_mole
import os
from pysmiles import read_smiles
from config import *
import networkx as nx
from pyscf import gto, scf


class MoleAsGraph:
    def __init__(self, name, edge_layout, elements_at_index, geometry):
        '''Caution: For all methods so far, the indexing of the geometry and the graphs (i.e. all
        remaining attributes) does not have to match! Keep this in mind: They may have different
        indices!!!!!! They do not even have to be the same molecules but then the Constructor may
        print a Warning'''
        self.name = name
        self.edge_layout = edge_layout #edge_layout = [[site_index, connected site index (singular!!!)], [...,...], [...,...]]
        self.elements_at_index = elements_at_index #Which element is at which vertex number
        self.geometry = geometry #The usual xyz representation
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

    def count_automorphisms(self):
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

    def get_equi_atoms_from_geom(self, gate_threshold=0, tolerance=rounding_tolerance):
        CN = Coulomb_neighborhood(center_mole(self.geometry), gate_threshold=gate_threshold)
        for i in range(len(self.geometry)):
            CN[i] = round(CN[i],tolerance)
        similars = np.array([np.where(CN == i)[0] for i in np.unique(CN)],dtype=object)
        #Delete all similars which include only one atom:
        num_similars = 0
        while num_similars < len(similars):
            if len(similars[num_similars])>1:
                num_similars += 1
            else:
                similars = np.delete(similars, num_similars, axis = 0)
        return similars

    def get_energy_NN(self):
        #Calculate the nuclear energy of the molecule
        sum = 0
        for i in range(self.number_atoms):
            for j in range(i+1,self.number_atoms):
                sum += elements[self.geometry[i][0]]*elements[self.geometry[j][0]]/np.linalg.norm(np.subtract(self.geometry[i][1:],self.geometry[j][1:]))
        return 0.5*sum

    def get_molecular_norm(self):
        return np.linalg.norm(CN_inertia_tensor(self.geometry))

    def print_atomic_norms(self, input_file):
        N = self.number_atoms
        file = open(input_file, 'r')
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

    def energy_PySCF(self, basis='ccpvdz'):
        #Make sure that the hydrogens are psrsed, too!!!!!!
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
        )
        mf = scf.HF(mol)
        energy = mf.kernel()
        return energy

    def fill_hydrogen_valencies(self, input_path, input_file):
        '''If the xyz file from which this molecule originates is known,
        the valencies can be filled with the hydrogens as given in the file.
        Add the geometric information line by line, and for each line, make
        one vertex and one edge to the closest heavy atom'''
        #check if file is present
        if os.path.isfile(input_path+input_file):
            #open text file in read mode
            f = open(input_path+input_file, "r")
            data = f.read()
            f.close()
        else:
            print('File', input_file, 'not found.')
            return 0
        #Initalize
        name = self.name
        new_geometry = self.geometry
        new_elements_at_index = self.elements_at_index
        new_edge_layout = self.edge_layout
        N_heavy = self.number_atoms #number of previously give atoms, all heavy
        N = int(data.splitlines(False)[0]) #number of atoms including hydrogen
        for i in range(2,N+2): #get only the hydrogens and their coordinates
            line = data.splitlines(False)[i]
            #Check for hydrogen specifically
            x = line.split('\t')
            if x[0] != 'H':
                continue
            else:
                new_geometry.append(['H', float(x[1]),float(x[2]),float(x[3])])
                new_elements_at_index.append('H')
            #Now: find the index of the atom with the shortest distance
            shortest_distance = 100000
            for j in range(N_heavy):
                distance = np.linalg.norm(np.subtract(self.geometry[j][1:],new_geometry[-1][1:]))
                if distance < shortest_distance:
                    shortest_distance = distance
                    index_of_shortest = j
            new_edge_layout.append([int(index_of_shortest),len(new_geometry)-1])
        return MoleAsGraph(name, self.edge_layout, self.elements_at_index, self.geometry)


def energy_PySCF_from_QM9(input_path, input_file, basis='ccpvdz'):
    if os.path.isfile(input_path+input_file):
        #open text file in read mode
        f = open(input_path+input_file, "r")
        data = f.read()
        f.close()
    else:
        print('File', input_file, 'not found.')
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

def parse_QM9toMAG(input_path, input_file, with_hydrogen = False):
    '''MoleAsGraph instance returned'''
    #check if file is present
    if os.path.isfile(input_path+input_file):
        #open text file in read mode
        f = open(input_path+input_file, "r")
        data = f.read()
        f.close()
    else:
        print('File', input_file, 'not found.')
        return 0
    #Get the name of the molecule
    MAG_name = input_file.split('.')[0]
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


def get_energy_const_atoms(input_path, input_file):
    if os.path.isfile(input_path+input_file):
        #open text file in read mode
        f = open(input_path+input_file, "r")
        data = f.read()
        f.close()
    else:
        print('File', input_file, 'not found.')
        return 0
    N = int(data.splitlines(False)[0])
    sum = 0
    for i in range(2,N+2): #get the atoms one by one
        line = data.splitlines(False)[i]
        #if line.split('\t')[0] != 'H':
        sum += atomref_U[line.split('\t')[0]]
    return sum


#Test-molecules-----------------------------------------------------------------
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
naphthalene = MoleAsGraph(  'Naphthalene',
                            [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[6,7],[7,8],[8,9],[9,5]],
                            ['C','C','C','C','C','C','C','C','C','C'],
                            [['C', 0,0,0.5], ['C', 0,0.8660254037844386467637231707,1], ['C', 0,2*0.8660254037844386467637231707,0.5],
                            ['C', 0,2*0.8660254037844386467637231707,-0.5], ['C', 0,0.8660254037844386467637231707,-1], ['C', 0,0,-0.5],
                            ['C', 0,-0.8660254037844386467637231707,1], ['C', 0,-2*0.8660254037844386467637231707,0.5], ['C', 0,-2*0.8660254037844386467637231707, -0.5], ['C', 0,-0.8660254037844386467637231707,-1]])
