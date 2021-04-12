import numpy as np
import time
import matplotlib.pyplot as plt
import os
import igraph
import sys
from config import *

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
