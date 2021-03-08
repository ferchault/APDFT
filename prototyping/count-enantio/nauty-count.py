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

class MoleAsGraph:
    def __init__(self, edge_layout, equi_sites, elements_at_index):
        self.edge_layout = np.array(edge_layout, dtype=object) #edge_layout = [[site_index, connected site index (singular!!!)], [...,...], [...,...]]
        self.equi_sites = np.array(equi_sites, dtype=object) # equi_sites = [[equivalent sites of type 1],[equivalent sites of type 2],[...]]
        self.elements_at_index = np.array(elements_at_index, dtype=object) #Which element is at which index number
        self.number_atoms = len(np.unique(self.edge_layout))
        self.max_index = np.amax(self.edge_layout)
        if self.number_atoms != self.max_index+1:
            print("Number of atoms does not match naming of nodes: enumerate the nodes with integers without omissions!")
    def site(self,site_number):
        if site_number >= self.number_atoms:
            raise ValueError("Cannot return site with index "+str(site_number)+". Molecule only has "+str(self.number_atoms)+" atoms.")
        else:
            return self.elements_at_index(site_number)
    def equi_sites(self):
        print('Under construction')
    def close_equi_sites(self):
        print('Under construction')

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

anthracene = MoleAsGraph(   [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[6,7],[7,8],[8,9],[9,5],[7,10],[10,11],[11,12],[12,13],[13,8]],
                            [[6,9],[0,5,7,8],[1,4,10,13],[2,3,11,12]],
                            ['C','C','C','C','C','C','C','C','C','C','C','C','C','C'])
benzene = MoleAsGraph(  [[0,1],[1,2],[2,3],[3,4],[4,5],[0,5]],
                        [[0,1,2,3,4,5]],
                        ['C','C','C','C','C','C'])
isochrysene = MoleAsGraph(  [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[2,9],[1,6],[6,7],[7,8],[8,9],[6,10],[10,11],[11,12],[12,13],[13,7],[8,14],[14,15],[15,16],[16,17],[17,9]],
                            [[1,2,6,7,8,9],[0,3,10,13,14,17],[4,5,11,12,15,16]],
                            ['C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C','C'])
naphthalene = MoleAsGraph(  [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[6,7],[7,8],[8,9],[9,5]],
                            [[0,5],[2,3,7,8],[1,4,6,9]],
                            ['C','C','C','C','C','C','C','C','C','C'])
phenanthrene = MoleAsGraph( [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[2,9],[1,6],[6,7],[7,8],[8,9],[6,10],[10,11],[11,12],[12,13],[13,7]],
                            [[1,6],[2,7],[3,13],[4,12],[5,11],[0,10],[8,9]],
                            ['C','C','C','C','C','C','C','C','C','C','C','C','C','C'])

def nautyAE(graph, m = [2,2], dZ=[+1,-1], debug = False, chem_formula = True):
    #graph is an instance of the MoleAsGraph class
    #m and dZ are each arrays that include the number and amount of change in nuclear charge
    start_time = time.time()
    m = np.array(m)
    dZ = np.array(dZ)
    N_m = len(m)
    N_dZ = len(dZ) #number of different charge differences = number of colors-1
    max_node_number = np.amax(graph.edge_layout)+1
    N = len(np.unique(graph.edge_layout))
    if 0 in dZ:
        raise ValueError("0 not allowed in array dZ")
    if N_dZ < 2:
        raise ValueError("Number of changes in charge must be at least 2")
    if N_m != N_dZ:
        raise ValueError("Number of changes and number of change values do not match!")
    #Check for overall charge conservation
    if (np.sum(np.multiply(m,dZ)) != 0):
        raise ValueError("Netto change in charge must be 0")
    if N == 1:
        raise ValueError("Graph needs to have at least 2 atoms.")
    if N != max_node_number:
        raise ValueError("Enumerate the nodes with integers without omissions")
    if N_dZ != len(np.unique(dZ)):
        raise ValueError("Equal values in multiple entries")
    if N < np.sum(m):
        raise ValueError("Too less atoms in molecule for sum of to be transmuted atoms.")

    #Use graph-based algorithm nauty27r1; build the string command to be passed to the bash
    #Command for the standard case looks like:
    #echo 'n=10;0:1;1:2;2:3;3:4;4:5;5:0;0:6;6:7;7:8;8:9;9:5;' | /home/simon/Desktop/nauty27r1/dretog -q | /home/simon/Desktop/nauty27r1/vcolg -q -T -m3 |
    #awk '{count1 = 0; count2 = 0; for (i=3; i<13; i++){if ($i == 1) count1++; else if ($i == 2) count2++;} if ((count1 == 2) && (count2 == 2)) print}'
    #This immediatly checks wether charge conservation and the correct number of colors are given.
    command = "echo 'n=" + str(N) + ";"
    for i in range(len(graph.edge_layout)):
        command += str(graph.edge_layout[i][0]) +":" + str(graph.edge_layout[i][1]) + ";"
    command += "' | /home/simon/Desktop/nauty27r1/dretog -q | /home/simon/Desktop/nauty27r1/vcolg -q -T -m"
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
    #Color 0 is the standard, colors 1,2,3,etc. are the deviations
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
    1) Is the netto charge within equi_sites conserved? Can this be done in vcolg?
    2) Is the graph not its own alchemical mirror image?
    This is the same as asking if graph and mirror are isomorphic but can only happen
    if and only if for each pair of m[i],dZ[i] there exists a pair m[j],-dZ[j]
    that is equal for i != j'''

    #print(graph_config)
    #Answering question one:
    config_num = 0
    while config_num < len(graph_config):
        for i in range(len(graph.equi_sites)):
            sum = 0
            for j in graph.equi_sites[i]:
                #Avoid getting the last element
                if graph_config[config_num][j] != 0:
                    sum += dZ[graph_config[config_num][j]-1]
            if sum != 0:
                graph_config = np.delete(graph_config, config_num, axis = 0)
                break
            if i == len(graph.equi_sites)-1:
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
        '''Use igraph's isomorphic-function to delete graphs which are themselves
        upon transmutation'''
        #Prepare the graph
        g1 = igraph.Graph([tuple(v) for v in graph.edge_layout])
        config_num = 0
        while config_num < len(graph_config):
            if g1.isomorphic_vf2(color1=graph_config[config_num], color2=[dZ2color[-color2dZ[graph_config[config_num][i]]] for i in range(N)]):
                graph_config = np.delete(graph_config, config_num, axis = 0)
            else:
                config_num += 1
    count = len(graph_config)
    if debug == True:
        print('---------------')
        print("Time:", (time.time() - start_time),"s")
        print('Alchemical Enantiomers:')
        for i in range(count):
            x = [inv_elements[elements[graph.elements_at_index[j]]+color2dZ[graph_config[i][j]]] for j in range(N)]
            print(str(sum_formula(x)) + ": " + str(x))
    if chem_formula == True:
        if count == 0:
            random_config = np.zeros((N))
            pos = 0
            for i in range(len(m)):
                for j in range(m[i]):
                    random_config[pos] = dZ[i]
                    pos += 1
            print(str(sum_formula([inv_elements[elements[graph.elements_at_index[j]]+random_config[j]] for j in range(N)])))
        else:
            print(str(sum_formula([inv_elements[elements[graph.elements_at_index[j]]+color2dZ[graph_config[0][j]]] for j in range(N)])))
    return count

def FindAE(graph, dZ_max = 3):
    '''Below are all the partitions of splitting m_tot = np.sum(dZ_all[i])
    atoms in a pure (i.e. uncolored) molecule in n=len(dZ_all[i]) partitions
    for dZ_max <= 1 or 2 or 3 up to m_tot = 8 and n = 2 and 3'''
    if dZ_max == 3:
        m_all = np.array([
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
        dZ_all = np.array([
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
    if dZ_max == 2:
        m_all = np.array([
        [1,1],[1,1],
        [2,1],
        [2,2],[2,2],
        [3,3],[3,3],[2,4],
        [4,4],[4,4],
        [1,1,3],[1,2,2],
        [1,2,5],[1,3,4]
        ], dtype=object)
        dZ_all = np.array([
        [1,-1],[2,-2],
        [-1,2],
        [1,-1],[2,-2],
        [1,-1],[+2,-2],[2,-1],
        [1,-1],[2,-2],
        [2,1,-1],[-2,2,-1],
        [1,2,-1],[-2,2,-1]
        ],dtype=object)
    if dZ_max == 1:
        m_all = np.array([
        [1,1],
        [2,2],
        [3,3],
        [4,4]
        ], dtype=object)
        dZ_all = np.array([
        [1,-1],
        [1,-1],
        [1,-1],
        [1,-1]
        ],dtype=object)
    #Check if they are overall netto charge is conserved
    #for i in range(len(m_all)):
    #    print(np.array([dZ_all[i][j]*m_all[i][j] for j in range(len(m_all[i]))]).sum())
    start_time = time.time()
    #Parallelize this for-loop:
    total_number = 0
    for i in range(len(m_all)):
        x = nautyAE(graph, m_all[i], dZ_all[i], debug= False, chem_formula = True)
        total_number += x
        print(x)
    print('Total time:', time.time()-start_time)
    print('Total number of AEs:', total_number)

#Function that gives equi_sites for arbitrary molecule
#Function that finds AEs from target molecule, not just reference (brute force with close_equi_sites)
#Optional: Take vcolg, rewrite it such that filtering happens in C, not awk or python

print('Naphthalene\n------------------------------')
FindAE(naphthalene)
print('Phenanthrene\n------------------------------')
FindAE(phenanthrene)
print('Anthracene\n------------------------------')
FindAE(anthracene)
print('Isochrysene\n------------------------------')
FindAE(isochrysene)
