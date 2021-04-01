from nautycount import *
from inertiacount import *
from MAG import *
from config import *
import os
import multiprocessing as mp

def Find_AEfromref(graph, dZ_max = 3, log = False, method = 'graph', bond_energy_rules = False):
    if method == 'geom' and bond_energy_rules != False:
        print("Warning: argument 'bond_energy_rules' is only supported for method 'geom'")
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
            x = nautyAE(graph, m_all[i], dZ_all[i], debug= False, chem_formula = True, bond_energy_rules = bond_energy_rules)
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
        if (CN_inertia_moment(target_config) == CN_inertia_moment(mirror_config)).all() or (netto_charge != 0):
            #If yes, wipe chem_config such that the original molecule is returned
            chem_config = np.array([graph.geometry[i][0] for i in range(graph.number_atoms)], copy=True)
    #Return a MoleAsGraph object
    for i in range(len(chem_config)):
        Geom[i][0] = chem_config[i]
    if log == True:
        print('NAME:\nreffrom'+str(graph.name))
        print('EDGE LAYOUT:\n'+str(graph.edge_layout))
        print('ELEMENTS AT INDEX:\n'+str(chem_config))
        print('GEOMETRY:\n' +str(Geom))
    return MoleAsGraph('reffrom'+graph.name, graph.edge_layout,chem_config.tolist(), Geom)


def uncolor(graph):
    '''Find the most symmetric reference molecule by setting all atoms to the
    rounded average integer charge.'''
    tot_nuc_charge = 0
    for i in range(graph.number_atoms):
        tot_nuc_charge += elements[graph.elements_at_index[i]]
    average_element = inv_elements[int(tot_nuc_charge/graph.number_atoms)]
    new_elements = np.empty((graph.number_atoms), dtype='str')
    for i in range(graph.number_atoms):
        new_elements[i] = average_element
    return MoleAsGraph('isoatomic'+graph.name, graph.edge_layout,new_elements,graph.geometry)


def Find_theoAEfromgraph(N = 3, dZ_max=1):
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
    print('Number of atoms: '+str(N)+'\tdZ_max: '+str(dZ_max)+'\tNumber of possibilities: '+str(num_lines)+'\tAEs within there: '+str(count)+'\tStepsize: '+str(batching))


def multicore_QM9(i):
    pos = '000000'[:(6-len(str(i)))] + str(i)
    Find_AEfromref(parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_' + pos + '.xyz'), log='sparse', dZ_max=1)
    #UNCOLOR: Find_AEfromref(uncolor(parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_' + pos + '.xyz')), log='sparse', dZ_max=1)
    #TARGET SEARCH: Find_AEfromref(Find_reffromtar(parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_' + pos + '.xyz'), dZ_max=2), log='sparse', dZ_max=2)


if __name__ == "__main__":
    #Going through QM9 and counting AEs (or whatever)
    for count in range(1,14+1):
        #Start at 1, end at 14+1
        start_tag = (count-1)*10000
        end_tag = count*10000
        if count == 1:
            #Skip everything with only one heavy atom: water, methane, ammonia. Start at index 4
            start_tag = 4
        if count == 14:
            end_tag = 133885+1
        batch_index = '00'[:(2-len(str(count)))] + str(count)


        #---------------------------Mutliprocessing-----------------------------
        pool = mp.Pool(mp.cpu_count()-2) #Keep 2 cores out of the picture
        with open('QM9_log'+batch_index+'_dZ1.txt', 'a') as f:
        #UNCOLOR: with open('QM9_uncolored_log'+batch_index+'_dZ1.txt', 'a') as f:
        #TARGET SEARCH: with open('QM9_target_log'+batch_index+'_dZ2.txt', 'a') as f:
            sys.stdout = f # Change the standard output to the created file
            results = pool.map(multicore_QM9, [i for i in range(start_tag,end_tag)])
            sys.stdout = original_stdout # Reset the standard output to its original value
            print(str(pos)+' -> Done')
        pool.close()


        #-----------------------------Find orbits-------------------------------
        '''with open('QM9_orbit_log'+batch_index+'_dZ2.txt', 'a') as f:
            for i in range(start_tag,end_tag):
                pos = '000000'[:(6-len(str(i)))] + str(i)
                sys.stdout = f # Change the standard output to the created file
                result = parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_' + pos + '.xyz').get_orbits_from_graph()
                if len(result[0]) == 0:
                    num_orbits = 0
                else:
                    num_orbits = len(result)
                num_vertices = 0
                max_sized_orbit = 2
                for i in range(len(result)):
                    num_vertices += len(result[i])
                    if len(result[i]) > max_sized_orbit:
                        max_sized_orbit = len(result[i])
                print(pos+'\t'+str(num_orbits)+'\t'+str(num_vertices)+'\t'+str(max_sized_orbit))
                sys.stdout = original_stdout # Reset the standard output to its original value
                print(str(pos)+' -> Done')'''


    #------------------------------Testing--------------------------------------
    #parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_022079.xyz')
    #Find_AEfromref(parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_133885.xyz'), log='sparse', dZ_max=2)
    #print(Find_reffromtar(benzene, method = 'geom', dZ_max = 1, log= True).elements_at_index)
    #print(naphthalene.get_energy_NN())
    #Find the sweetspots
    #Find_theoAEfromgraph(N=10, dZ_max=1)

#---------------------------Available functions---------------------------------

#in AE.py
#Find_AEfromref(graph, dZ_max = 3, log = False, method = 'graph', bond_energy_rules = False)
#Find_reffromtar(graph, dZ_max = 3, method = 'graph', log = False)
#uncolor(graph)
#Find_theoAEfromgraph(N = 3, dZ_max=1)

#in MAG.py
#parse_QM9toMAG(input_path, input_file)


#-------------------------------------TODOS-------------------------------------
#Bond energy rules, already implemented in Find_AEfromref, continue with that
    #Checking for bond energy rules
    #for i in range(4,48):
    #    pos = '000000'[:(6-len(str(i)))] + str(i)
    #    Find_AEfromref(parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_' + pos + '.xyz'), bond_energy_rules = True, dZ_max=2, log = 'quiet')
#Optional: Take vcolg, rewrite it such that filtering happens in C, not awk or python
