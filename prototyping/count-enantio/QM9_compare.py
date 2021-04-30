#Compare the same property taken from QM9 with calculation from xTB/PySCF
from AlEn import *
import os
import subprocess
import multiprocessing as mp
import matplotlib.pyplot as plt
import sys

def QM9_quantity(tag, quantity_number):
    f = open(PathToQM9XYZ + 'dsgdb9nsd_'+ tag +'.xyz', 'r')
    data = f.read()
    f.close()
    N = int(data.splitlines(False)[0]) #number of atoms including hydrogen
    quantity = data.splitlines(False)[1].split('\t')[quantity_number-1]
    #print(data)
    '''I          --------  -----------  --------------
     1  tag       -            "gdb9"; string constant to ease extraction via grep
     2  index     -            Consecutive, 1-based integer identifier of molecule
     3  A         GHz          Rotational constant A
     4  B         GHz          Rotational constant B
     5  C         GHz          Rotational constant C
     6  mu        Debye        Dipole moment
     7  alpha     Bohr^3       Isotropic polarizability
     8  homo      Hartree      Energy of Highest occupied molecular orbital (HOMO)
     9  lumo      Hartree      Energy of Lowest occupied molecular orbital (LUMO)
    10  gap       Hartree      Gap, difference between LUMO and HOMO
    11  r2        Bohr^2       Electronic spatial extent
    12  zpve      Hartree      Zero point vibrational energy
    13  U0        Hartree      Internal energy at 0 K
    14  U         Hartree      Internal energy at 298.15 K
    15  H         Hartree      Enthalpy at 298.15 K
    16  G         Hartree      Free energy at 298.15 K
    17  Cv        cal/(mol K)  Heat capacity at 298.15 K'''
    #energy_NN = parse_QM9toMAG(PathToQM9XYZ, 'dsgdb9nsd_' + tag + '.xyz').get_energy_NN()
    #print(energy_NN)
    return float(quantity)

def xTB_quantity(tag):
    command = f'xtb '+PathToQM9XYZ+'dsgdb9nsd_'+tag+'.xyz --silent'
    output = os.popen(command).read()
    #print(output)
    energy = output.split('| TOTAL ENERGY')[1].split(' Eh ')[0].strip()
    #subprocess.run('rm wbo xtbrestart charges xtbtopo.mol xtbopt.log xtbopt.xyz'.split())
    return float(energy)

def multifunc(tag_number, batch_index):
    pos = '000000'[:(6-len(str(tag_number)))] + str(tag_number)
    with open('logs/SCFvsQM9_log'+batch_index+'.txt', 'a') as f:
        sys.stdout = f # Change the standard output to the created file
        print(pos+'\t'+str(QM9_quantity(pos,14))+'\t'+str(energy_PySCF_from_QM9(PathToQM9XYZ, 'dsgdb9nsd_' + pos + '.xyz')))
        sys.stdout = original_stdout # Reset the standard output to its original value
        print(str(pos)+' -> Done')


if __name__ == "__main__":
    for count in range(1,1+1):
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
        pool.starmap(multifunc, [(i,batch_index) for i in range(start_tag,end_tag)])
        pool.close()
