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

def nautyAE(graph, equi_sites, m = [2,2], dZ=[+1,-1], debug = False):
    '''graph = [[site_index, connected site index (singular!!!)], [...,...], [...,...]]
    equi_sites = [[equivalent sites of type 1],[equivalent sites of type 2],[...]]
    m and dZ are each arrays that include the number and amount of change in nuclear charge'''
    start_time = time.time()
    N_m = len(m)
    N_dZ = len(dZ)
    max_node_number = np.amax(graph)+1
    N = len(np.unique(graph))
    if 0 in dZ:
        raise ValueError("0 not allowed in array dZ")
    if N_m != N_dZ:
        raise ValueError("Number of changes and number of change values do not match!")
    #Check for overall charge conservation
    if ((N_m*N_dZ).sum() != 0):
        raise ValueError("Netto change in charge must be 0")
    if N == 1:
        raise ValueError("Graph needs to have at least 2 atoms.")
    if N != max_node_number:
        raise ValueError("Enumerate the nodes with integers without omissions")
    if N_dZ != np.unique(dZ):
        raise ValueError("Equal values in multiple entries")
    #Use graph-based algorithm nauty27r1; build the string command to be passed to the bash
    command = "echo 'n=" + str(N) + ";"
    for i in range(len(graph)):
        command += str(graph[i][0]) +":" + str(graph[i][1]) + ";"
    command += "' | /home/simon/Desktop/nauty27r1/dretog -q | /home/simon/Desktop/nauty27r1/vcolg -q -T -m3 | awk '{if (($3"
    for i in range(4,N+3):
        command += "+$" + str(i)
    sum = 0
    for i in range(N_m):
        sum += i*m[i]
    command += ") == " + str(sum) + ") print}'"
    #This last command is no check for charge conservation; it gets rid of a lot of configurations though
    output = os.popen(command).read()
    print(command)
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

    '''The parsed array needs to fulfill three things:
    1) Is the number of charged sites correct?
    2) Is the netto charge within equi_sites conserved?
    3) Is the graph not its own alchemical mirror image?
    This is the same as asking if graph and mirror are isomorphic but can only happen
    if and only if for each pair of m[i],dZ[i] there exists a pair m[j],-dZ[j]
    that is equal for i != j'''
    #Answering questions one and two:
    config_num = 0
    while config_num < len(graph_config):
        if [(graph_config[config_num] == v).sum() for v in range(N_m)].all():
            for i in range(len(equi_sites)):
                sum = 0
                for j in equi_sites[i]:
                    sum += dZ[graph_config[config_num][j]-1]
                if sum != 0:
                    graph_config = np.delete(graph_config, config_num, axis = 0)
                    break
                if i == len(equi_sites)-1:
                    config_num += 1
        else:
            graph_config = np.delete(graph_config, config_num, axis = 0)





    #Answering the third question:
    '''Find our if graph is able to self mirror'''
    '''Use igraph's isomorphic-function to delete graphs which are themselves
    upon transmutation'''
    g1 = igraph.Graph([tuple(v) for v in graph])
    config_num = 0
    while config_num < len(graph_config):
        if g1.isomorphic_vf2(color1=graph_config[config_num], color2=[2-graph_config[config_num][i] for i in range(N)]):
            graph_config = np.delete(graph_config, config_num, axis = 0)
        else:
            config_num += 1






    count = len(graph_config)
    if debug == True:
        print('---------------')
        print("Time:", (time.time() - start_time),"s")
        print('---------------')
        #Here, a nice drawing part would be awesome!!!
        print(graph_config) #prints the number of the respective color along all equivalent sites
    return count

benzene_topol = [[0,1],[1,2],[2,3],[3,4],[4,5],[0,5]]
benzene_equi_sites = [[0,1,2,3,4,5]]

naphthalene_topol = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[6,7],[7,8],[8,9],[9,5]]
naphthalene_equi_sites = [[0,5],[2,3,7,8],[1,4,6,9]]

triangle_topol = [[0,1],[1,2],[2,0]]
triangle_equi_sites = [[0,1,2]]

print(nautyAE(naphthalene_topol, naphthalene_equi_sites, m = [2,2], dZ = [1,-1]))
