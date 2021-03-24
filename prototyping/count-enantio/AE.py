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


def Find_reffromtar(graph, dZ_max = 3, method = 'graph', log = False):
    '''Find the most symmetric reference molecule, not all of them. Most symmetric
    means here the least amount of atoms are not part of an orbit/equivalent set. Less
    symmetric is always possible and included in most symmetric (and appears specifically
    when searching for AEs)'''
    #Initalize original state:
    if method == 'graph':
        chem_config = np.copy(graph.elements_at_index)
    if method == 'geom':
        chem_config = np.array([graph.geometry[i][0] for i in range(graph.number_atoms)])
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
        for i in range(len(Geom)):
            CN[i] = round(CN[i],tolerance)
        sites = np.array([np.where(CN == i)[0] for i in np.unique(CN)],dtype=object)
        #Delete all similars which include only one atom:
        num_similars = 0
        while num_similars < len(sites):
            if len(sites[num_similars])>1:
                num_similars += 1
            else:
                sites = np.delete(sites, num_similars, axis = 0)
    if len(sites) == 1:
        if len(sites[0]) == 0:
            if log == True:
                print('NAME:\nreffrom'+str(graph.name))
                print('EDGE LAYOUT:\n'+str(graph.edge_layout))
                print('ELEMENTS AT INDEX:\n'+str(graph.elements_at_index))
                print('GEOMETRY:\n' +str(graph.geometry))
            return MoleAsGraph('reffrom'+graph.name, graph.edge_layout, chem_config, graph.geometry)
    else:
        '''We want to maximize the number of elements per orbit/equivalent set. Use bestest()'''
        for alpha in sites:
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
        #Return a MoleAsGraph object
        for i in range(len(chem_config)):
            Geom[i][0] = chem_config[i]
        if log == True:
            print('NAME:\nreffrom'+str(graph.name))
            print('EDGE LAYOUT:\n'+str(graph.edge_layout))
            print('ELEMENTS AT INDEX:\n'+str(chem_config))
            print('GEOMETRY:\n' +str(Geom))
        return MoleAsGraph('reffrom'+graph.name, graph.edge_layout,chem_config, Geom)

if __name__ == "__main__":
    #with open('QM9_log01.txt', 'a') as f:
        #Skip everything with only one heavy atom: water, methane, ammonia. Start at index 4, end at index 133885
    #for i in range(4,10000):
    #    pos = '000000'[:(6-len(str(i)))] + str(i)
        #sys.stdout = f # Change the standard output to the created file
    #    Find_AEfromref(parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_' + pos + '.xyz'), log='sparse', dZ_max=2)
        #sys.stdout = original_stdout # Reset the standard output to its original value
        #print(str(pos)+' -> Done')
    '''with open('QM9_target_log01.txt', 'a') as f:
        for i in range(4,10000):
            pos = '000000'[:(6-len(str(i)))] + str(i)
            sys.stdout = f # Change the standard output to the created file
            Find_AEfromref(Find_reffromtar(parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_' + pos + '.xyz'), dZ_max=2), log='sparse', dZ_max=2)
            sys.stdout = original_stdout # Reset the standard output to its original value
            print(str(pos)+' -> Done')'''
    #Testing
    #parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_022079.xyz')
    Find_AEfromref(naphthalene, log='sparse', dZ_max=2)
    #Find_reffromtar(naphthalene, method = 'geom', dZ_max = 2, log= True)
    #print(naphthalene.get_energy_NN())

#TODOS:
#Optional: Take vcolg, rewrite it such that filtering happens in C, not awk or python
#Optional: Parallelize the for-loop in Find_AEfromref
