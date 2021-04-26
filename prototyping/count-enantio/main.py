from AlEn import *
import multiprocessing as mp
import os

def multicore_QM9(tag_number, batch_index, dZ_max):
    pos = '000000'[:(6-len(str(tag_number)))] + str(tag_number)
    #Check for directory /logs and create one if necessary:
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    #-----------------------------Count AEs-------------------------------------
    #RAW:
    '''with open('logs/QM9_log'+batch_index+'_dZ'+str(dZ_max)+'_geom.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        Find_AEfromref(parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz'), log='sparse', dZ_max=dZ_max, method = 'geom')
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')'''

    #UNCOLOR:
    '''with open('logs/QM9_uncolored_log'+batch_index+'_dZ'+str(dZ_max)+'_geom.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        Find_AEfromref(uncolor(parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz')), log='sparse', dZ_max=dZ_max, method = 'geom')
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')'''

    #TARGET SEARCH:
    '''with open('logs/QM9_target_log'+batch_index+'_dZ'+str(dZ_max)+'_geom.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        Find_AEfromref(Find_reffromtar(parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz'), dZ_max=dZ_max), log='sparse', dZ_max=dZ_max, method = 'graph')
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')'''

    #-----------------------------Find orbits-----------------------------------
    '''with open('logs/QM9_orbit_log'+batch_index+'_dZ'+str(dZ_max)+'_graph.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        orbs = parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz').get_orbits_from_graph()
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
        norm = np.linalg.norm(CN_inertia_tensor(parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz').geometry))
        print(pos+'\t'+str(norm))
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')'''

    #------------Complete information aboout number and energy of AEs-----------
    '''with open('logs/QM9_log'+batch_index+'_dZ'+str(dZ_max)+'_geom_energy.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        inputpath = PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz'
        Find_AEfromref(parse_QM9toMAG(inputpath), log='verbose', dZ_max=2, method = 'geom', take_hydrogen_data_from=inputpath)
        print('\n')
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')'''


#-------------------------------MAIN PROGRAM------------------------------------

#----------------------Going through QM9------------------------------------
'''for count in range(1,14+1):
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
    pool.close()'''

#---------------------------theoretically possible graphs-------------------
'''pool = mp.Pool(int(performance_use*mp.cpu_count()))
pool.starmap(Find_theoAEfromgraph, [(i,z) for i,z in itertools.product([2,3,4,5,6,7,8,9],[1,2])])
pool.close()'''


#------------------------------Testing--------------------------------------
#mole2 = parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True)
#print(mole2.get_total_energy(), mole2.get_nuclear_energy(), mole2.get_total_energy()-mole2.get_nuclear_energy())
#mole1 = parse_QM9toMAG('cyclooctatetraene_13halfs.xyz', with_hydrogen=True)
#print(mole1.get_total_energy(), mole1.get_nuclear_energy(), mole1.get_total_energy()-mole1.get_nuclear_energy())
#mole3 = parse_QM9toMAG('cyclooctatetraene_14halfs.xyz', with_hydrogen=True)
#print(mole3.get_total_energy(), mole3.get_nuclear_energy(), mole3.get_total_energy()-mole3.get_nuclear_energy())
#print(parse_QM9toMAG('benzene.xyz', with_hydrogen=True).get_total_energy())


#print(geomAE(parse_QM9toMAG('cyclooctatetraene_13halfs.xyz'), m=[4,4], dZ=[0.5,-0.5], debug = True, get_all_energies=True, take_hydrogen_data_from='cyclooctatetraene_13halfs.xyz'))
#print(geomAE(parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_010839.xyz'), m=[2,2], dZ=[1,-1], debug = True, get_all_energies=True, chem_formula = True, take_hydrogen_data_from=PathToQM9XYZ+'dsgdb9nsd_010839.xyz'))
#print(parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_010839.xyz').get_energy_NN(take_hydrogen_data_from=PathToQM9XYZ+'dsgdb9nsd_010839.xyz'))
#print(parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_010839.xyz').fill_hydrogen_valencies(PathToQM9XYZ+'dsgdb9nsd_010839.xyz').energy_PySCF())
#print(geomAE(parse_QM9toMAG('cubane_13halfs.xyz'), m=[4,4], dZ=[0.5,-0.5], debug = True, get_all_energies=True, chem_formula = True, take_hydrogen_data_from='cubane_13halfs.xyz'))
#print(parse_QM9toMAG('benzene.xyz', with_hydrogen=True).get_energy_NN())
#print(geomAE(parse_QM9toMAG('benzene.xyz'), m=[2,2], dZ=[1,-1], get_all_energies=False, get_electronic_energy_difference=True, take_hydrogen_data_from='benzene.xyz', debug=False))
#Find_AEfromref(parse_QM9toMAG('benzene.xyz'), log='normal', method='geom', dZ_max=2, take_hydrogen_data_from='benzene.xyz')
#print(total_energy_with_dZ(parse_QM9toMAG('cyclooctatetraene_13halfs.xyz', with_hydrogen=True).geometry, dZ=[0,-0.01,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))

#print(total_energy_with_dZ(parse_QM9toMAG('cyclooctatetraene_13halfs.xyz', with_hydrogen=True).geometry, dZ=[0,0.01,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
#print(parse_QM9toMAG('cyclooctatetraene_13halfs.xyz', with_hydrogen=True).get_total_energy())
print(parse_QM9toMAG('benzene_AE1.xyz', with_hydrogen=True).get_total_energy())
print(parse_QM9toMAG('benzene_AE2.xyz', with_hydrogen=True).get_total_energy())
for i in range(0,5):
    print("Taylor series of energy, order "+str(i)+" :")
    print(taylorseries_energy(parse_QM9toMAG('benzene.xyz', with_hydrogen=True).geometry, [1,-1,-1,1,0,0], i))
#print(parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True).get_total_energy())
#print(parse_QM9toMAG('cyclooctatetraene_13halfs.xyz', with_hydrogen=True).get_total_energy())
#print(parse_QM9toMAG('cyclooctatetraene_14halfs.xyz', with_hydrogen=True).get_total_energy())


#print(Find_reffromtar(benzene, method = 'geom', dZ_max = 1, log= 'normal').get_energy_NN())
'''for tag_number in range(100,133885+1):
    pos = '000000'[:(6-len(str(tag_number)))] + str(tag_number)
    f = open(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz', "r")
    data = f.read()
    f.close()
    N = int(data.splitlines(False)[0])
    smiles = data.splitlines(False)[N+3].split('\t')[0].strip()
    inchi = data.splitlines(False)[N+4].split('\t')[0].strip()
    #print(smiles)
    #This case: Benzene
    if smiles == 'C1C=CC=CC1':
        print(pos)
    if inchi == 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H':
        print(pos)'''
    #Find_AEfromref(parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz'), log='verbose', dZ_max=2, method = 'geom', take_hydrogen_data_from=PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz')
    #print('\n')
    #Find_AEfromref(uncolor(parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz')), log='sparse', dZ_max=2, method = 'geom')
    #print(pos)
    #print(parse_QM9toMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz', with_hydrogen=False).fill_hydrogen_valencies(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz').energy_PySCF())
