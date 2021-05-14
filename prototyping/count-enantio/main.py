from AlEn import *
import multiprocessing as mp
import os
performance_use = 0.90

#TAGS:
#benzene:
#cyclooctatraene: dsgdb9nsd_017954

def multicore_QM9(tag_number, batch_index, dZ_max):
    pos = '000000'[:(6-len(str(tag_number)))] + str(tag_number)
    #Check for directory /logs and create one if necessary:
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    #-----------------------------Count AEs-------------------------------------
    #RAW:
    '''with open('logs/QM9_log'+batch_index+'_dZ'+str(dZ_max)+'_geom.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        Find_AEfromref(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz'), log='sparse', dZ_max=dZ_max, method = 'geom')
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')'''

    #UNCOLOR:
    '''with open('logs/QM9_uncolored_log'+batch_index+'_dZ'+str(dZ_max)+'_geom.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        Find_AEfromref(uncolor(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz')), log='sparse', dZ_max=dZ_max, method = 'geom')
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')'''

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
        Find_AEfromref(parse_XYZtoMAG(inputpath), log='verbose', dZ_max=2, method = 'geom', take_hydrogen_data_from=inputpath)
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
#mole2 = parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True)
#print(mole2.get_total_energy(), mole2.get_nuclear_energy(), mole2.get_total_energy()-mole2.get_nuclear_energy())
#mole1 = parse_XYZtoMAG('cyclooctatetraene_13halfs.xyz', with_hydrogen=True)
#print(mole1.get_total_energy(), mole1.get_nuclear_energy(), mole1.get_total_energy()-mole1.get_nuclear_energy())
#mole3 = parse_XYZtoMAG('cyclooctatetraene_14halfs.xyz', with_hydrogen=True)
#print(mole3.get_total_energy(), mole3.get_nuclear_energy(), mole3.get_total_energy()-mole3.get_nuclear_energy())
#print(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).get_total_energy())


#print(geomAE(parse_XYZtoMAG('cyclooctatetraene_13halfs.xyz'), m=[4,4], dZ=[0.5,-0.5], debug = True, get_all_energies=True, take_hydrogen_data_from='cyclooctatetraene_13halfs.xyz'))
#print(geomAE(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_010839.xyz'), m=[2,2], dZ=[1,-1], debug = True, get_all_energies=True, chem_formula = True, take_hydrogen_data_from=PathToQM9XYZ+'dsgdb9nsd_010839.xyz'))
#print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_010839.xyz').get_energy_NN(take_hydrogen_data_from=PathToQM9XYZ+'dsgdb9nsd_010839.xyz'))
#print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_010839.xyz').fill_hydrogen_valencies(PathToQM9XYZ+'dsgdb9nsd_010839.xyz').energy_PySCF())
#print(geomAE(parse_XYZtoMAG('cubane_13halfs.xyz'), m=[4,4], dZ=[0.5,-0.5], debug = True, get_all_energies=True, chem_formula = True, take_hydrogen_data_from='cubane_13halfs.xyz'))
#print(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).get_energy_NN())
#print(geomAE(parse_XYZtoMAG('benzene.xyz'), m=[2,2], dZ=[1,-1], get_all_energies=False, get_electronic_energy_difference=True, take_hydrogen_data_from='benzene.xyz', debug=False))
#Find_AEfromref(parse_XYZtoMAG('benzene.xyz'), log='normal', method='geom', dZ_max=2, take_hydrogen_data_from='benzene.xyz')
#print(total_energy_with_dZ(parse_XYZtoMAG('cyclooctatetraene_13halfs.xyz', with_hydrogen=True).geometry, dZ=[0,-0.01,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))

#print(total_energy_with_dZ(parse_XYZtoMAG('cyclooctatetraene_13halfs.xyz', with_hydrogen=True).geometry, dZ=[0,0.01,0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
#print(parse_XYZtoMAG('cyclooctatetraene_13halfs.xyz', with_hydrogen=True).get_total_energy())

'''for i in range(-100,101,10):
    #print(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).add_dZ([0.02*i,-0.02*i,-0.02*i,0.02*i,0,0]).geometry[:4])
    print(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).add_dZ([0.01*i,-0.01*i,-0.01*i,0.01*i,0,0]).get_electronic_energy())
#print(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).add_dZ([-0.02,0.02,0.02,-0.02,0,0]).get_electronic_energy())

print("Actual")
print(parse_XYZtoMAG('benzene_AE1.xyz', with_hydrogen=True).get_electronic_energy())
print(parse_XYZtoMAG('benzene_AE2.xyz', with_hydrogen=True).get_electronic_energy())'''
print("main.py started")

#mole = parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True)
#print(mole.number_atoms, mole.get_electronic_energy([]))

#print(water.get_Hessian())
"""
sum=0
print("-------------------------NNCNNCCC-------------------------")
print("Actual")
print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True).get_electronic_energy([1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0]))
for i in range(0,7):
    print("Taylor series of energy, order "+str(i)+" :")
    res = lambda_taylorseries_electronic_energy(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True), [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0], [0.5,0.5,-0.5,0.5,0.5,-0.5,-0.5,-0.5,0,0,0,0,0,0,0,0], i)
    print(res)
    sum += res
    print("Sum = "+str(sum))
sum = 0
print("-------------------------CCNCCNNN-------------------------")
print("Actual")
print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True).get_electronic_energy([0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0]))
for i in range(0,7):
    print("Taylor series of energy, order "+str(i)+" :")
    res = lambda_taylorseries_electronic_energy(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True), [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0], [-0.5,-0.5,0.5,-0.5,-0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0], i)
    print(res)
    sum += res
    print("Sum = "+str(sum))
"""
#parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).plot_delta_rho([-1,1,1,-1,0,0,0,0,0,0,0,0],[1,-1,-1,1,0,0,0,0,0,0,0,0])
#parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).plot_rho(dZ=[1,-1,-1,1,0,0,0,0,0,0,0,0])
#parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_000214.xyz', with_hydrogen=True, angle=[0,0,30]).plot_rho(dZ=[0,0,0,0,0,0,0,0,0,0,0,0], z_filter=0.001, title='Benzene')
parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_000010.xyz', with_hydrogen=True, angle=[180,90,90]).plot_rho(z_filter=0.1)
#parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_017954.xyz', with_hydrogen=True).plot_rho(dZ=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,0,0,0,0], z_filter=0.01)

#test = MoleAsGraph('H2', [[0,1]], ['H','H'], [['H', 0,0,0],['H',0,0,1.4]])

#test.get_total_energy_arbprec()
#water = parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_000003.xyz', with_hydrogen=True)
"""
H2O= MoleAsGraph(   'H2O',
                    [[0,2],[1,2]],
                    ['H','H','O'],
                    [['H', 1.809 * np.sin(104.52 / 180 * np.pi / 2), 0, 0], ['H', -1.809 * np.sin(104.52 / 180 * np.pi / 2), 0, 0], ['O', 0, 1.809 * np.sin(104.52 / 180 * np.pi / 2), 0]]
                    )
print(H2O.get_total_energy())
print(H2O.get_total_energy_arbprec())
"""

"""
sum=0
print("-------------------------NBBNCC-------------------------")
print("Actual")
print(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).get_electronic_energy([1,-1,-1,1,0,0,0,0,0,0,0,0]))
for i in range(0,7):
    print("Taylor series of energy, order "+str(i)+" :")
    res = lambda_taylorseries_electronic_energy(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True), [0,0,0,0,0,0,0,0,0,0,0,0], [1,-1,-1,1,0,0,0,0,0,0,0,0], i)
    print(res)
    sum += res
    print("Sum = "+str(sum))
sum = 0
print("-------------------------BNNBCC-------------------------")
print("Actual")
print(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True).get_electronic_energy([-1,1,1,-1,0,0,0,0,0,0,0,0]))
for i in range(0,7):
    print("Taylor series of energy, order "+str(i)+" :")
    res = lambda_taylorseries_electronic_energy(parse_XYZtoMAG('benzene.xyz', with_hydrogen=True), [0,0,0,0,0,0,0,0,0,0,0,0], [-1,1,1,-1,0,0,0,0,0,0,0,0], i)
    print(res)
    sum += res
    print("Sum = "+str(sum))
"""
"""
#print(Find_reffromtar(benzene, method = 'geom', dZ_max = 1, log= 'normal').get_energy_NN())
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
    if smiles == 'C1=CC=CC=C1':
        print(pos)
    #if inchi == 'InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H':
    #    print(pos)'''
    #Find_AEfromref(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz'), log='verbose', dZ_max=2, method = 'geom', take_hydrogen_data_from=PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz')
    #print('\n')
    #Find_AEfromref(uncolor(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz')), log='sparse', dZ_max=2, method = 'geom')
    #print(pos)
    #print(parse_XYZtoMAG(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz', with_hydrogen=False).fill_hydrogen_valencies(PathToQM9XYZ+'dsgdb9nsd_' + pos + '.xyz').energy_PySCF())
"""
"""
(base) simon@KrugAspire:~/ce$ python3 main.py
main.py started
-------------------------NBBNCC-------------------------
Actual
-435.87721373221734
Taylor series of energy, order 0 :
-433.2225790071665
Sum = -433.2225790071665
Taylor series of energy, order 1 :
2.5233423350812245e-06
Sum = -433.2225764838241
Taylor series of energy, order 2 :
-2.53075481603163
Sum = -435.75333129985574
Taylor series of energy, order 3 :
0.008558638441613642
Sum = -435.7447726614141
Taylor series of energy, order 4 :
-0.15284105071189957
Sum = -435.89761371212603
Taylor series of energy, order 5 :
0.01331066488013657
Sum = -435.8843030472459
-------------------------BNNBCC-------------------------
Actual
-435.9038223625181
Taylor series of energy, order 0 :
-433.2225790071665
Sum = -433.2225790071665
Taylor series of energy, order 1 :
-2.5233407585645296e-06
Sum = -433.2225815305072
Taylor series of energy, order 2 :
-2.5307548161007034
Sum = -435.7533363466079
Taylor series of energy, order 3 :
-0.008558638934819581
Sum = -435.7618949855427
Taylor series of energy, order 4 :
-0.15284094517550512
Sum = -435.9147359307182
Taylor series of energy, order 5 :
-0.013311751990151931
Sum = -435.9280476827084


"""
"""
(base) simon@mg:~/files/count-enantio$ python3 main.py
main.py started
-------------------------NNCNNCCC-------------------------
Actual
-739.3292463171272
Taylor series of energy, order 0 :
-738.0412452979488
Sum = -738.0412452979488
Taylor series of energy, order 1 :
2.559893156709829e-06
Sum = -738.0412427380556
Taylor series of energy, order 2 :
-1.2810620025464443
Sum = -739.3223047406021
Taylor series of energy, order 3 :
-0.00018135811782708325
Sum = -739.32248609872
Taylor series of energy, order 4 :
0.002857157678900236
Sum = -739.3196289410411
Taylor series of energy, order 5 :
-0.00044140212760958883
Sum = -739.3200703431687
Taylor series of energy, order 6 :
-0.004137434706357778
Sum = -739.324207777875
-------------------------CCNCCNNN-------------------------
Actual
-739.3284616392402
Taylor series of energy, order 0 :
-738.0412452979488
Sum = -738.0412452979488
Taylor series of energy, order 1 :
-2.559893156709829e-06
Sum = -738.0412478578419
Taylor series of energy, order 2 :
-1.2810620025642077
Sum = -739.3223098604061
Taylor series of energy, order 3 :
0.00018135741742438446
Sum = -739.3221285029887
Taylor series of energy, order 4 :
0.002857165485865188
Sum = -739.3192713375028
Taylor series of energy, order 5 :
0.0004415661400975412
Sum = -739.3188297713626
Taylor series of energy, order 6 :
-0.004138688079122776
Sum = -739.3229684594418
"""
