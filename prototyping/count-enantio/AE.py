from nautycount import *
from inertiacount import *
from MAG import *

def FindAE(graph, dZ_max = 3, log = True, method = 'graph'):
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
    num = 0
    available_sites = len(np.hstack(graph.equi_sites).ravel())
    while num < len(m_all):
        if np.sum(m_all[num]) > available_sites:
            m_all = np.delete(m_all, num, axis = 0)
            dZ_all = np.delete(dZ_all, num, axis = 0)
        else:
            num += 1
    #Parallelize this for-loop:
    total_number = 0
    if log == True:
        log_name = graph.name + '_' + method + ".txt"
        with open(log_name, 'w') as f:
            sys.stdout = f # Change the standard output to the created file
            print('\n'+ graph.name + '; method = ' + method + '\n------------------------------')
            for i in range(len(m_all)):
                m_time = time.time()
                if method == 'graph':
                    x = nautyAE(graph, m_all[i], dZ_all[i], debug= False, chem_formula = True)
                if method == 'geom':
                    x = geomAE(graph.geometry, m_all[i], dZ_all[i], debug= False, chem_formula = True)
                print('Time:', time.time()-m_time)
                print(x)
                total_number += x
            print('Total time:', time.time()-start_time)
            print('Total number of AEs:', total_number)
            sys.stdout = original_stdout # Reset the standard output to its original value
    else:
        print('\n'+ graph.name + '; method = ' + method + '\n------------------------------')
        for i in range(len(m_all)):
            m_time = time.time()
            if method == 'graph':
                x = nautyAE(graph, m_all[i], dZ_all[i], debug= False, chem_formula = True)
            if method == 'geom':
                x = geomAE(graph.geometry, m_all[i], dZ_all[i], debug= False, chem_formula = True)
            print('Time:', time.time()-m_time)
            print(x)
            total_number += x
        print('Total time:', time.time()-start_time)
        print('Total number of AEs:', total_number)

#TODOS:
#Function that gives equi_sites for arbitrary molecule
#Function that finds AEs from target molecule, not just reference (brute force with close_equi_sites)
#Optional: Take vcolg, rewrite it such that filtering happens in C, not awk or python
#Optional: Parallelize the for-loop in FindAE

FindAE(benzene)
FindAE(benzene, method = 'geom')
FindAE(naphthalene)
FindAE(naphthalene, method = 'geom')
#FindAE(phenanthrene)
#FindAE(phenanthrene, method = 'geom')
#FindAE(anthracene)
#FindAE(anthracene, method = 'geom')
#FindAE(isochrysene)
#FindAE(isochrysene, method = 'geom')
