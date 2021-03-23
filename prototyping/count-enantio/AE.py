from nautycount import *
from inertiacount import *
from MAG import *
from config import *

def Find_AEfromref(graph, dZ_max = 3, log = True, method = 'graph'):
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
        m_all = np.array([[]], dtype=object)
        dZ_all = np.array([[]], dtype=object)
    #Parallelize this for-loop:
    #For plotting: save number of transmuted atoms num_trans and time
    num_trans = np.array([])
    times = np.array([])
    total_number = 0
    log_name = graph.name + '_' + method + ".txt"
    if log == True:
        with open(log_name, 'w') as f:
            sys.stdout = f # Change the standard output to the created file
            print('\n'+ graph.name + '; method = ' + method + '\n------------------------------')
            sys.stdout = original_stdout # Reset the standard output to its original value
    if log == False:
        print('\n'+ graph.name + '; method = ' + method + '\n------------------------------')

    for i in range(len(m_all)):
        random_config = np.zeros((graph.number_atoms))
        pos = 0
        for j in range(len(m_all[i])):
            for k in range(m_all[i][j]):
                random_config[pos] = dZ_all[i][j]
                pos += 1
        chem_form = str(sum_formula([inv_elements[elements[graph.elements_at_index[v]]+random_config[v]] for v in range(graph.number_atoms)]))
        m_time = time.time()
        if method == 'graph':
            x = nautyAE(graph, m_all[i], dZ_all[i], debug= False, chem_formula = True)
        if method == 'geom':
            x = geomAE(graph.geometry, m_all[i], dZ_all[i], debug= False, chem_formula = True)
        if log == True:
            with open(log_name, 'a') as f:
                sys.stdout = f # Change the standard output to the created file
                print(chem_form)
                print('Time:', time.time()-m_time)
                print(x)
                sys.stdout = original_stdout # Reset the standard output to its original value
        if log == False:
            print(chem_form)
            print('Time:', time.time()-m_time)
            print(x)
        num_trans = np.append(num_trans, np.sum(m_all[i]))
        times = np.append(times, time.time()-m_time)
        total_number += x
    if log == True:
        with open(log_name, 'a') as f:
            sys.stdout = f # Change the standard output to the created file
            print('Total time:', time.time()-start_time)
            print('Total number of AEs:', total_number)
            print('Number of transmuted atoms:', list(num_trans))
            print('Time:', list(times))
            sys.stdout = original_stdout # Reset the standard output to its original value
    if log == False:
        print('Total time:', time.time()-start_time)
        print('Total number of AEs:', total_number)
        print('Number of transmuted atoms:', list(num_trans))
        print('Time:', list(times))
    if log == 'sparse':
        print(graph.name + '\t' + str(time.time()-start_time) + '\t' + str(graph.number_atoms) + '\t' + str(total_number))
    if log == 'quiet':
        return total_number

def Find_reffromtar(graph, dZ_max = 3, method = 'graph'):
    '''Find the most symmetric reference molecule, not all  of them. Most symmetric
    means here the least amount of atoms not part of a orbis/equivalent sets. Less
    symmetric is always possible and included in most symmetric (and appears specifically,
    when searching for AEs)

    Find all orbits/equivalent sets if the molecule is colorless/isoatomic
    and save them in a list of lists called sites. This is the reason why we
    dropped the more precise terms "orbit" and "similars"'''
    if method == 'graph':
        #This is basically identical with MoleAsGraph's get_orbits_from_graph method
        g = igraph.Graph([tuple(v) for v in graph.edge_layout])
        #Get all automorphisms with uncolored vertices
        automorphisms = g.get_automorphisms_vf2()
        if len(automorphisms) == 1:
            orbit = [[]]
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
    if method == 'geom':
        #This is basically the same as MoleAsGraph's get_equi_atoms_from_geom method
        CN = Coulomb_neighborhood(graph.geometry)
        for i in range(len(graph.geometry)):
            CN[i] = round(CN[i],tolerance)
        sites = np.array([np.where(CN == i)[0] for i in np.unique(CN)],dtype=object)
        #Delete all similars which include only one atom:
        num_similars = 0
        while num_similars < len(sites):
            if len(sites[num_similars])>1:
                num_similars += 1
            else:
                sites = np.delete(sites, num_similars, axis = 0)
    #Initalize array for deviations; this can be a flat array, we only need a mapping of atom_ID
    #to the deviation
    dev = np.zeros((len(np.hstack(sites).ravel()),2))
    #Initalize mu
    mu = np.zeros(len(sites))
    '''Initalize elements_at_index-like vector with all entries mu[alpha] except the atoms
    which are not in sites, they are their original chemical element.'''
    chem_config = np.copy(graph.elements_at_index)
    pos = 0
    for alpha in range(len(sites)):
        #Find average; this is a little tricky for mu should be an integer:
        mu[alpha] = int(np.sum([elements[graph.elements_at_index[int(sites[alpha][i])]] for i in range(len(sites[alpha]))])/len(sites[alpha]))
        #Find deviations:
        for i in sites[alpha]:
            #print(mu-elements[graph.elements_at_index[int(sites[alpha][int(i)])]])
            dev[pos][0] = int(i) #Index
            dev[pos][1] = mu[alpha]-elements[graph.elements_at_index[int(i)]]
            chem_config[int(i)] = inv_elements[int(mu[alpha])]
            pos += 1
    #Special case: All is fine and within dZ_max:
    if np.max(dev[...][1])-np.min(dev[...][1]) <= 2*dZ_max:
        print(chem_config)
        print(dev)
        return MoleAsGraph('reffrom'+graph.name, graph.edge_layout, chem_config, graph.geometry)


if __name__ == "__main__":
    '''with open('QM9_log01.txt', 'a') as f:
        #Skip everything with only one heavy atom: water, methane, ammonia. Start at index 4, end at index 133885
        for i in range(4,10000):
            pos = '000000'[:(6-len(str(i)))] + str(i)
            sys.stdout = f # Change the standard output to the created file
            Find_AEfromref(parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_' + pos + '.xyz'), log='sparse', dZ_max=2)
            sys.stdout = original_stdout # Reset the standard output to its original value
            print(pos)'''
    #Testing
    #parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_022079.xyz')
    #Find_AEfromref(naphthalene, log='sparse', dZ_max=1)
    Find_reffromtar(naphthalene, method = 'graph')
    #print(naphthalene.get_energy_NN())

#TODOS:
#Function that finds references from target molecule, not just reference (brute force with close_orbits)
#Optional: Take vcolg, rewrite it such that filtering happens in C, not awk or python
#Optional: Parallelize the for-loop in Find_AEfromref
