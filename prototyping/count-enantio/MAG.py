import numpy as np
import igraph
from inertiacount import Coulomb_neighborhood, array_compare
import os
from pysmiles import read_smiles

tolerance = 3

class MoleAsGraph:
    def __init__(self, name, edge_layout, elements_at_index, geometry, orbits=None):
        '''Caution: For all methods so far, the indexing of the geometry and the graphs (i.e. all
        remaining attributes) does not have to match! Keep this in mind: They may have different
        indices!!!!!!'''
        self.name = name
        self.edge_layout = np.array(edge_layout, dtype=object) #edge_layout = [[site_index, connected site index (singular!!!)], [...,...], [...,...]]
        self.elements_at_index = np.array(elements_at_index, dtype=object) #Which element is at which vertex number
        self.geometry = geometry #The usual xyz representation
        if edge_layout != None:
            self.number_atoms = len(np.unique(self.edge_layout))
        elif elements_at_index != None:
            self.number_atoms = len(self.elements_at_index)
        else:
            print(self, "is underdefined.")
        if len(self.edge_layout) != 0:
            self.max_index = np.amax(self.edge_layout)
        else:
            self.max_index = 0
        if self.number_atoms != self.max_index+1:
            print("Number of atoms does not match naming of vertices: enumerate the vertices with integers without omissions!")
        if orbits != None:
            #Specific orbits can be forced!
            self.orbits = np.array(orbits, dtype=object) # orbits = [[equivalent sites of type 1],[equivalent sites of type 2],[...]]
        #This part can only be used if you are sure that the indexing of atoms in the graph is the same as in the xyz
        #elif geometry != None:
        #    self.orbits = np.array(self.get_equi_atoms_from_geom(), dtype=object)
        elif edge_layout != None:
            self.orbits = np.array(self.get_orbits_from_graph(), dtype=object)
        else:
            print(self, "is underdefined.")

    def site(self,site_number):
        if site_number >= self.number_atoms:
            raise ValueError("Cannot return site with index "+str(site_number)+". Molecule only has "+str(self.number_atoms)+" atoms.")
        else:
            return self.elements_at_index(site_number)






    def get_orbits_from_graph(self):
        #Prepare the graph
        g = igraph.Graph([tuple(v) for v in self.edge_layout])
        #THIS DOES NOT WHAT IT SHOULD DO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print(g.get_automorphisms_vf2())
        '''
        bws = [round(g.betweenness(vertices=i),tolerance) for i in range(self.number_atoms)]
        similars = np.array([np.where(bws == i)[0] for i in np.unique(bws)],dtype=object)
        print(similars)
        #Delete all sets of equivalent atoms which include only one atom:
        num_similars = 0
        while num_similars < len(similars):
            if len(similars[num_similars])>1:
                num_similars += 1
            else:
                similars = np.delete(similars, num_similars, axis = 0)
        return similars
        '''





    def get_equi_atoms_from_geom(self):
        CN = Coulomb_neighborhood(self.geometry)
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

def parse_QM9toMAG(input_path, input_file):
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
    #Get the geomeztry of the molecule
    mole = np.array([['C', (1,2,3)]], dtype=object) #Initalize array for molecule (geometric information)
    mole = np.delete(mole, 0, axis=0)
    N_heavyatoms = N
    for i in range(2,N+2): #get the atoms and their coordinates
        line = data.splitlines(False)[i]
        if line.split('\t')[0] == 'H':
            N_heavyatoms -= 1
            continue
        else:
            symbol = line.split('\t')[0]
            x = float(line.split('\t')[1].strip())
            y = float(line.split('\t')[2].strip())
            z = float(line.split('\t')[3].strip())
            mole = np.append(mole, np.array([[symbol,(x,y,z)]],dtype=object), axis=0)
    #Get the edges of the molecule as a graph
    network = read_smiles(data.splitlines(False)[N+3].split('\t')[0])
    edge_layout = [list(v) for v in network.edges()]
    elements_at_index = [v[1] for v in network.nodes(data='element')]
    return MoleAsGraph(MAG_name, edge_layout, elements_at_index, mole)


#Test-molecules-----------------------------------------------------------------
anthracene = MoleAsGraph('Anthracene',
                        [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[6,7],[7,8],[8,9],[9,5],[7,10],[10,11],[11,12],[12,13],[13,8]],
                        ['C','C','C','C','C','C','C','C','C','C','C','C','C','C'],
                        [['C', (0,0,0.5)], ['C', (0,0.8660254037844386467637231707,1)], ['C', (0,2*0.8660254037844386467637231707,0.5)],
                        ['C', (0,2*0.8660254037844386467637231707,-0.5)], ['C', (0,0.8660254037844386467637231707,-1)], ['C', (0,0,-0.5)],
                        ['C', (0,-0.8660254037844386467637231707,1)], ['C', (0,-2*0.8660254037844386467637231707,0.5)], ['C', (0,-2*0.8660254037844386467637231707, -0.5)], ['C', (0,-0.8660254037844386467637231707,-1)],
                        ['C', (0,-3*0.8660254037844386467637231707,1)], ['C', (0,-4*0.8660254037844386467637231707,0.5)], ['C', (0,-4*0.8660254037844386467637231707, -0.5)], ['C', (0,-3*0.8660254037844386467637231707,-1)]])
benzene = MoleAsGraph(  'Benzene',
                        [[0,1],[1,2],[2,3],[3,4],[4,5],[0,5]],
                        ['C','C','C','C','C','C'],
                        [['C', (0,0,1)], ['C', (0,0.8660254037844386467637231707,0.5)], ['C', (0,0.8660254037844386467637231707,-0.5)],
                        ['C', (0,0,-1)], ['C', (0,-0.8660254037844386467637231707,-0.5)], ['C', (0,-0.8660254037844386467637231707,0.5)]])
isochrysene = MoleAsGraph(  'Isochrysene',
                            [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[6,7],[7,8],[8,9],[9,5],[8,10],[10,11],[11,12],[12,13],[13,9],[6,14],[14,15],[15,16],[16,17],[17,7]],
                            ['C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C'],
                            [['C', (0,1,0.8660254037844386467637231707)], ['C', (0,2,0.8660254037844386467637231707)], ['C', (0,2.5,0)],
                            ['C', (0,2,-0.8660254037844386467637231707)], ['C', (0,1,-0.8660254037844386467637231707)], ['C', (0,0.5,0)],
                            ['C', (0,0.5,2*0.8660254037844386467637231707)], ['C', (0,-0.5,2*0.8660254037844386467637231707)], ['C', (0,-1,0.8660254037844386467637231707)], ['C', (0,-0.5,0)],
                            ['C', (0,-2,0.8660254037844386467637231707)], ['C', (0,-2.5,0)], ['C', (0,-2,-0.8660254037844386467637231707)], ['C', (0,-1,-0.8660254037844386467637231707)],
                            ['C', (0,1,3*0.8660254037844386467637231707)], ['C', (0,0.5,4*0.8660254037844386467637231707)], ['C', (0,-0.5,4*0.8660254037844386467637231707)], ['C', (0,-1,3*0.8660254037844386467637231707)]])
naphthalene = MoleAsGraph(  'Naphthalene',
                            [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[6,7],[7,8],[8,9],[9,5]],
                            ['C','C','C','C','C','C','C','C','C','C'],
                            [['C', (0,0,0.5)], ['C', (0,0.8660254037844386467637231707,1)], ['C', (0,2*0.8660254037844386467637231707,0.5)],
                            ['C', (0,2*0.8660254037844386467637231707,-0.5)], ['C', (0,0.8660254037844386467637231707,-1)], ['C', (0,0,-0.5)],
                            ['C', (0,-0.8660254037844386467637231707,1)], ['C', (0,-2*0.8660254037844386467637231707,0.5)], ['C', (0,-2*0.8660254037844386467637231707, -0.5)], ['C', (0,-0.8660254037844386467637231707,-1)]])
phenanthrene = MoleAsGraph( 'Phenanthrene',
                            [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[6,7],[7,8],[8,9],[9,5],[8,10],[10,11],[11,12],[12,13],[13,9]],
                            ['C','C','C','C','C','C','C','C','C','C','C','C','C','C'],
                            [['C', (0,1,0.8660254037844386467637231707)], ['C', (0,2,0.8660254037844386467637231707)], ['C', (0,2.5,0)],
                            ['C', (0,2,-0.8660254037844386467637231707)], ['C', (0,1,-0.8660254037844386467637231707)], ['C', (0,0.5,0)],
                            ['C', (0,0.5,2*0.8660254037844386467637231707)], ['C', (0,-0.5,2*0.8660254037844386467637231707)], ['C', (0,-1,0.8660254037844386467637231707)], ['C', (0,-0.5,0)],
                            ['C', (0,-2,0.8660254037844386467637231707)], ['C', (0,-2.5,0)], ['C', (0,-2,-0.8660254037844386467637231707)], ['C', (0,-1,-0.8660254037844386467637231707)]])

heptagon = MoleAsGraph('Heptagon',
                        [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,0]],
                        ['C','C','C','C','C','C','C'],
                        None)
