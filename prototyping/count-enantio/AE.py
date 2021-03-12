from nautycount import *
from inertiacount import *
from MAG import *

def FindAE_fromref(graph, dZ_max = 3, log = True, method = 'graph'):
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


#TODOS:
#Function that finds AEs from target molecule, not just reference (brute force with close_orbits)
#Optional: Take vcolg, rewrite it such that filtering happens in C, not awk or python OR take pynauty
#Optional: Parallelize the for-loop in FindAE_fromref

#FindAE_fromref(heptagon, log='sparse')
print(parse_QM9toMAG('/home/simon/Desktop/QM9/XYZ/', 'dsgdb9nsd_002430.xyz').get_orbits_from_graph())
#FindAE_fromref(benzene, log = 'quiet')
#FindAE_fromref(benzene, method = 'geom', log = 'quiet')
#FindAE_fromref(naphthalene)
#FindAE_fromref(naphthalene, method = 'geom')
#FindAE_fromref(phenanthrene)
#FindAE_fromref(phenanthrene, method = 'geom')
#FindAE_fromref(anthracene)
#FindAE_fromref(anthracene, method = 'geom')
#FindAE_fromref(isochrysene)
#FindAE_fromref(isochrysene, method = 'geom')

with open('QM9_log.txt', 'a') as f:
    #Skip everything with only one heavy atom: water, methane, ammonia. Start at index 4
    for i in range(4,10001):
        pos = '000000'[:(6-len(str(i)))] + str(i)
        sys.stdout = f # Change the standard output to the created file
        FindAE_fromref(parse_QM9toMAG('/home/simon/Desktop/QM9/XYZ/', 'dsgdb9nsd_' + pos + '.xyz'), log='sparse', dZ_max=2)
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(pos)
