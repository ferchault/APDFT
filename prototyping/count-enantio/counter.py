from pyscf import gto, scf
from pyscf.symm.geom import detect_symm #change TOLERANCE to higher values
import numpy as np

elements = {'H':1, 'He':2,
'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18,
'K':19, 'Ca':20, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,}

def Coulomb_matrix(atom):
    #returns the Coulomb matrix of a given molecule
    N = len(atom)
    result = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if (j == i):
                result[i][i] = pow(elements[atom[i][0]], 2.4)
            else:
                result[i][j] = elements[atom[i][0]]*elements[atom[j][0]]/np.linalg.norm(np.subtract(atom[i][1],atom[j][1]))
    return result

def Coulomb_neighborhood(atom):
    #returns the sum over rows/columns of the Coulomb matrix. Thus, each atom is
    #assigned its Coulombic neighborhood
    matrix = Coulomb_matrix(atom)
    return matrix.sum(axis = 0)

# The order in which the atoms are presented constitutes their ID
atom = [['C', (0,0,1)], ['C', (0,0.866025,0.5)], ['C', (0,0.866025,-0.5)],
['C', (0,0,-1)], ['C', (0,-0.866025,-0.5)], ['C', (0,-0.866025,0.5)]]
gpname, orig, axes = detect_symm(atom)

#Hard code what is returned if molecule has only one or two atoms
print(Coulomb_neighborhood(atom))
print(gpname)
