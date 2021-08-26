from AlEn import *
import multiprocessing as mp
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
performance_use = 0.95

#TAGS:
#benzene: dsgdb9nsd_000214
#cyclooctatraene: dsgdb9nsd_017954

def multicore_QM9(tag_number, batch_index, dZ_max):
    pos = '000000'[:(6-len(str(tag_number)))] + str(tag_number)
    #Check for directory /logs and create one if necessary:
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    #-----------------------------Count AEs-------------------------------------
    #RAW:
    with open('logs/QM9_log'+batch_index+'_dZ'+str(dZ_max)+'_geom.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        Find_AEfromref(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz'), log='sparse', dZ_max=dZ_max, method = 'geom')
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')

    #UNCOLOR:
    """
    with open('logs/QM9_uncolored_log'+batch_index+'_dZ'+str(dZ_max)+'_geom.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        Find_AEfromref(uncolor(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz')), log='sparse', dZ_max=dZ_max, method = 'geom')
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')
    """

    #TARGET SEARCH:
    '''with open('logs/QM9_target_log'+batch_index+'_dZ'+str(dZ_max)+'_geom.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        Find_AEfromref(Find_reffromtar(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz'), dZ_max=dZ_max), log='sparse', dZ_max=dZ_max, method = 'graph')
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')'''

    #-----------------------------Find orbits-----------------------------------
    '''with open('logs/QM9_orbit_log'+batch_index+'_dZ'+str(dZ_max)+'_graph.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        orbs = parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz').get_orbits_from_graph()
        if len(orbs[0]) == 0:
            num_orbits = 0
        else:
            num_orbits = len(orbs)
        num_vertices = 0
        max_sized_orbit = 2
        for j in range(len(orbs)):
            num_vertices += len(orbs[j])
            if len(orbs[j]) > max_sized_orbit:
                max_sized_orbit = len(orbs[j])
        print(pos+'\t'+str(num_orbits)+'\t'+str(num_vertices)+'\t'+str(max_sized_orbit))
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')'''

    #-----------------------------Find norms-----------------------------------
    '''with open('logs/QM9_norm_log'+batch_index+'_dZ'+str(dZ_max)+'_geom.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        norm = np.linalg.norm(CN_inertia_tensor(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz').geometry))
        print(pos+'\t'+str(norm))
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')'''

    #------------Complete information aboout number and energy of AEs-----------
    '''with open('logs/QM9_log'+batch_index+'_dZ'+str(dZ_max)+'_geom_energy.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        inputpath = PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz'
        Find_AEfromref(parse_XYZtoMAG(inputpath), log='verbose', dZ_max=dZ_max, method = 'geom', take_hydrogen_data_from=inputpath)
        print('\n')
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')'''

    #------------------Energy differences of pairs of AE------------------------
    """
    with open('logs/QM9_log_energydiff_dZ'+str(dZ_max)+'_range'+str(batch_index)+'.txt', 'a') as f: #batch index is the yukawa mass here
        sys.stdout = f # Change the standard output to the created file
        inputpath = PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz'
        Find_AEfromref(parse_XYZtoMAG(inputpath, with_hydrogen = False), dZ_max=dZ_max, log = 'quiet', with_electronic_energy_difference = True, method = 'geom', take_hydrogen_data_from=inputpath)
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')
    """

def multicore_ZINC(tag_number, batch_index, dZ_max):
    pos = '00000'[:(5-len(str(tag_number)))] + str(tag_number)
    #Check for directory /logs and create one if necessary:
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    #-----------------------------Count AEs-------------------------------------
    #RAW:
    """
    with open('logs/ZINC_log'+batch_index+'_dZ'+str(dZ_max)+'_geom.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        Find_AEfromref(parse_MOL2toMAG(PathToZINC+'ZINC_named_' + pos + '.mol2'), log='sparse', dZ_max=dZ_max, method = 'geom')
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')

    """
    #------------------Energy differences of pairs of AE------------------------
    with open('logs/ZINC_log_energydiff_dZ'+str(dZ_max)+'_range'+str(batch_index)+'.txt', 'a') as f: #batch index is the yukawa mass here
        sys.stdout = f # Change the standard output to the created file
        inputpath = PathToZINC+'ZINC_named_' + pos + '.mol2'
        Find_AEfromref(parse_MOL2toMAG(inputpath, with_hydrogen = False), dZ_max=dZ_max, log = 'quiet', with_electronic_energy_difference = True, method = 'geom', take_hydrogen_data_from=inputpath)
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')

#-------------------------------MAIN PROGRAM------------------------------------
print("main.py started")

#----------------------Going through QM9------------------------------------
"""
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
    pool = mp.Pool(int(performance_use*mp.cpu_count()))
    pool.starmap(multicore_QM9, [(i,batch_index,1) for i in range(start_tag,end_tag)])
    pool.close()

    pool = mp.Pool(int(performance_use*mp.cpu_count()))
    pool.starmap(multicore_QM9, [(i,batch_index,2) for i in range(start_tag,end_tag)])
    pool.close()
"""

#----------------------Going through ZINC------------------------------------


"""
for count in [1,2,3,4,5,6]:
    start_tag = (count-1)*10000
    end_tag = count*10000
    if count == 1:
        #The first file is empty
        start_tag = 1
    if count == 6:
        end_tag = 59986+1
    batch_index = '00'[:(2-len(str(count)))] + str(count)

    #---------------------------Mutliprocessing-----------------------------
    pool = mp.Pool(int(performance_use*mp.cpu_count()))
    pool.starmap(multicore_ZINC, [(i,batch_index,1) for i in range(start_tag,end_tag)])
    pool.close()
    pool = mp.Pool(int(performance_use*mp.cpu_count()))
    pool.starmap(multicore_ZINC, [(i,batch_index,2) for i in range(start_tag,end_tag)])
    pool.close()
"""


#----------------------Finding Yukawa mass by going through QM9-----------------
"""
print('----------------------------------')
print(standard_yukawa_range)
start_tag = 4
end_tag = 133885+1
pool = mp.Pool(int(performance_use*mp.cpu_count()))
pool.starmap(multicore_QM9, [(i,standard_yukawa_range,1) for i in range(start_tag,end_tag,100)])
pool.close()
"""

#----------------------Finding Yukawa mass by going through ZINC----------------

print('----------------------------------')
print(standard_yukawa_range)
start_tag = 1
end_tag = 59986+1
pool = mp.Pool(int(performance_use*mp.cpu_count()))
pool.starmap(multicore_ZINC, [(i,standard_yukawa_range,1) for i in range(start_tag,end_tag,50)])
pool.close()



#---------------------------theoretically possible graphs-------------------
"""
pool = mp.Pool(int(performance_use*mp.cpu_count()))
pool.starmap(Find_theoAEfromgraph, [(i,z) for i,z in itertools.product([2,3,4,5,6,7,8,9],[1,2])])
pool.close()
"""

#------------------------------Testing--------------------------------------
#print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_007404.xyz', with_hydrogen=True).get_electronic_energy([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
#print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_040004.xyz', with_hydrogen=False).equi_atoms)
#print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_040004.xyz', with_hydrogen=False).orbits)
#print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_000214.xyz', with_hydrogen=False).geometry)
#parse_MOL2toMAG(PathToZINC+'ZINC_named_00004.mol2', with_hydrogen=False)

#print(geomAE(parse_XYZtoMAG('cyclooctatetraene_13halfs.xyz'), m=[4,4], dZ=[0.5,-0.5], debug = True, get_all_energies=True, take_hydrogen_data_from='cyclooctatetraene_13halfs.xyz'))
#print(geomAE(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_010839.xyz'), m=[2,2], dZ=[1,-1], debug = True, get_all_energies=True, chem_formula = True, take_hydrogen_data_from=PathToQM9XYZ+'dsgdb9nsd_010839.xyz'))
#print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_010839.xyz').get_energy_NN(take_hydrogen_data_from=PathToQM9XYZ+'dsgdb9nsd_010839.xyz'))
#print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_010839.xyz').fill_hydrogen_valencies(PathToQM9XYZ+'dsgdb9nsd_010839.xyz').energy_PySCF())
#print(geomAE(parse_XYZtoMAG('cubane_13halfs.xyz'), m=[4,4], dZ=[0.5,-0.5], debug = True, get_all_energies=True, chem_formula = True, take_hydrogen_data_from='cubane_13halfs.xyz'))
#print(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).get_energy_NN())
#print(geomAE(parse_XYZtoMAG('benzene.xyz'), m=[2,2], dZ=[1,-1], get_all_energies=False, get_electronic_energy_difference=True, take_hydrogen_data_from='benzene.xyz', debug=False))

#Find_AEfromref(parse_XYZtoMAG('adamantane.xyz'), log='normal', method='geom', dZ_max=1, take_hydrogen_data_from='adamantane.xyz', with_Taylor_expansion = True)
#print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_040004.xyz', with_hydrogen=True).get_electronic_energy([0,0,0,-1,0,1,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0]))
#print(Find_AEfromref(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_040004.xyz', with_hydrogen = False), dZ_max=1, log = 'quiet', with_electronic_energy_difference = True, method = 'geom', take_hydrogen_data_from=PathToQM9XYZ+'dsgdb9nsd_040004.xyz'))
#print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_040004.xyz').equi_atoms)
#Find_AEfromref(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_040004.xyz'), log='normal', method='geom', dZ_max=1, take_hydrogen_data_from='benzene.xyz')

#print(total_energy_with_dZ(parse_XYZtoMAG('cyclooctatetraene_13halfs.xyz', with_hydrogen=True).geometry, dZ=[0,-0.01,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))

#print(total_energy_with_dZ(parse_XYZtoMAG('cyclooctatetraene_13halfs.xyz', with_hydrogen=True).geometry, dZ=[0,0.01,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
#print(parse_XYZtoMAG('cyclooctatetraene_13halfs.xyz', with_hydrogen=True).get_total_energy())

#inputpath = PathToQM9XYZ+'dsgdb9nsd_001733.xyz'
#mole = parse_XYZtoMAG(inputpath, with_hydrogen=False)
#print(mole.elements_at_index)
#geomAE(mole, dZ=[1,-1], m=[1,1], get_electronic_energy_difference=True, take_hydrogen_data_from=inputpath)
"""
start_tag = 4
end_tag = 100 #133885+1
all_norms = []
for i in range(start_tag, end_tag):
    pos = '000000'[:(6-len(str(i)))] + str(i)
    current_norms = parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz', with_hydrogen = True).print_atomic_norms(tolist=True, with_hydrogen=False)
    for j in range(len(current_norms)):
        all_norms.append(current_norms[j])
all_norms.sort(key = lambda k: k[3]) #sort by norm
print(np.array(all_norms, dtype=object))
"""
#print(parse_XYZtoMAG('benzene.xyz', with_hydrogen = True).geometry)

#parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_000018.xyz', with_hydrogen=True, angle=[0,0,-60]).plot_rho_3D()
#print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_000024.xyz', with_hydrogen=True).geometry)
#for i in range(30,40):
#    print(i)
#    parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_'+'000000'[:(6-len(str(i)))] + str(i)+'.xyz', with_hydrogen=True).plot_rho_2D()

#parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).plot_rho_2D()
#parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).plot_delta_rho_2D([1,-1,-1,1,0,0,0,0,0,0,0,0], [-1,1,1,-1,0,0,0,0,0,0,0,0])


#parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True, angle=[-145,-21,-63]).plot_rho_3D(Z=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0])
#parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True, angle=[-145,-21,-63]).plot_delta_rho_3D([0.5,0.5,-0.5,0.5,0.5,-0.5,-0.5,-0.5,0,0,0,0,0,0,0,0], [-0.5,-0.5,0.5,-0.5,-0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0], Z = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0])
#test = MoleAsGraph('H2', [[0,1]], ['H','H'], [['H', 0,0,0],['H',0,0,1.4]])

#test.get_total_energy_arbprec()
#water = parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_000003.xyz', with_hydrogen=True)


"""
print("-------------------------NNCNNCCC-------------------------")
sum=0
print("Actual")
print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True).get_electronic_energy([1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0]))
for i in range(0,7):
    print("Taylor series of energy, order "+str(i)+" :")
    res = lambda_taylorseries_electronic_energy(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True), [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0], [0.5,0.5,-0.5,0.5,0.5,-0.5,-0.5,-0.5,0,0,0,0,0,0,0,0], i)
    print(res)
    sum += res
    print("Sum = "+str(sum))
print("-------------------------CCNCCNNN-------------------------")
sum = 0
print("Actual")
print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True).get_electronic_energy([0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0]))
for i in range(0,7):
    print("Taylor series of energy, order "+str(i)+" :")
    res = lambda_taylorseries_electronic_energy(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True), [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0], [-0.5,-0.5,0.5,-0.5,-0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0], i)
    print(res)
    sum += res
    print("Sum = "+str(sum))
"""


"""
print("-------------------------NBBNCC-------------------------")
sum=0
print("Actual")
print(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).get_electronic_energy([1,-1,-1,1,0,0,0,0,0,0,0,0]))
for i in range(0,7):
    print("Taylor series of energy, order "+str(i)+" :")
    res = lambda_taylorseries_electronic_energy(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True), [0,0,0,0,0,0,0,0,0,0,0,0], [1,-1,-1,1,0,0,0,0,0,0,0,0], i)
    print(res)
    sum += res
    print("Sum = "+str(sum))
print("-------------------------BNNBCC-------------------------")
sum = 0
print("Actual")
print(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).get_electronic_energy([-1,1,1,-1,0,0,0,0,0,0,0,0]))
for i in range(0,7):
    print("Taylor series of energy, order "+str(i)+" :")
    res = lambda_taylorseries_electronic_energy(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True), [0,0,0,0,0,0,0,0,0,0,0,0], [-1,1,1,-1,0,0,0,0,0,0,0,0], i)
    print(res)
    sum += res
    print("Sum = "+str(sum))
"""

#print(Find_reffromtar(benzene, method = 'geom', dZ_max = 1, log= 'normal').get_energy_NN())
"""
for tag_number in range(10,133885+1):
    pos = '000000'[:(6-len(str(tag_number)))] + str(tag_number)
    f = open(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz', "r")
    data = f.read()
    f.close()
    N = int(data.splitlines(False)[0])
    smiles = data.splitlines(False)[N+3].split('\t')[0].strip()
    inchi = data.splitlines(False)[N+4].split('\t')[0].strip()
    #print(smiles)
    #This case: Benzene
    if smiles == 'C1C3CC2CC(CC1C2)C3':
        print(pos)
    #if inchi == 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H':
    #    print(pos)'''
    #Find_AEfromref(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz'), log='verbose', dZ_max=2, method = 'geom', take_hydrogen_data_from=PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz')
    #print('\n')
    #Find_AEfromref(uncolor(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz')), log='sparse', dZ_max=2, method = 'geom')
    #print(pos)
    #print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz', with_hydrogen=False).fill_hydrogen_valencies(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz').energy_PySCF())
"""
