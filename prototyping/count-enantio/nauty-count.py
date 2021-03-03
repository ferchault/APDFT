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
    m = np.array(m)
    dZ = np.array(dZ)
    N_m = len(m)
    N_dZ = len(dZ) #number of different charge differences = number of colors-1
    max_node_number = np.amax(graph)+1
    N = len(np.unique(graph))
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

    #Use graph-based algorithm nauty27r1; build the string command to be passed to the bash
    #Command for the standard case looks like:
    #echo 'n=10;0:1;1:2;2:3;3:4;4:5;5:0;0:6;6:7;7:8;8:9;9:5;' | /home/simon/Desktop/nauty27r1/dretog -q | /home/simon/Desktop/nauty27r1/vcolg -q -T -m3 |
    #awk '{count1 = 0; count2 = 0; for (i=3; i<13; i++){if ($i == 1) count1++; else if ($i == 2) count2++;} if ((count1 == 2) && (count2 == 2)) print}'
    #This immediatly checks wether charge conservation and the correct number of colors are given.
    command = "echo 'n=" + str(N) + ";"
    for i in range(len(graph)):
        command += str(graph[i][0]) +":" + str(graph[i][1]) + ";"
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
    1) Is the netto charge within equi_sites conserved?
    2) Is the graph not its own alchemical mirror image?
    This is the same as asking if graph and mirror are isomorphic but can only happen
    if and only if for each pair of m[i],dZ[i] there exists a pair m[j],-dZ[j]
    that is equal for i != j'''

    #print(graph_config)
    #Answering question one:
    config_num = 0
    while config_num < len(graph_config):
        for i in range(len(equi_sites)):
            sum = 0
            for j in equi_sites[i]:
                #Avoid getting the last element
                if graph_config[config_num][j] != 0:
                    sum += dZ[graph_config[config_num][j]-1]
            if sum != 0:
                graph_config = np.delete(graph_config, config_num, axis = 0)
                break
            if i == len(equi_sites)-1:
                config_num += 1
    #print(graph_config)

    #Answering the second question:
    '''Find out if all those graphs are able to self mirror'''
    self_mirrorable = np.array([-i in dZ for i in dZ]).all()
    if self_mirrorable:
        '''Use igraph's isomorphic-function to delete graphs which are themselves
        upon transmutation'''
        #Prepare some dicts
        color2dZ = {0:0}
        for i in range(N_dZ):
            color2dZ[i+1] = dZ[i]
        dZ2color = {v: k for k, v in color2dZ.items()}
        #Prepare the graph
        g1 = igraph.Graph([tuple(v) for v in graph])
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

print(nautyAE(naphthalene_topol, naphthalene_equi_sites, m = [2,2], dZ = [-1,1], debug = True))
