#ALL IMPORT STATEMENTS----------------------------------------------------------
import numpy as np
import math
import os
import sys
original_stdout = sys.stdout # Save a reference to the original standard output
np.set_printoptions(threshold=sys.maxsize) #show full arrays
import time
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.interpolate import griddata
import igraph
import networkx as nx
import itertools
from pyscf import gto, scf, qmmm
import pyscf
from pysmiles import read_smiles
import mpmath
mpmath.mp.dps = 30 #IMPORTANT FOR ARBITRARY PRECISION HF-COMPUTATION
import imageio
import copy

#ALL CONFIGURATIONS AND GLOBAL VARIABLES----------------------------------------
tolerance = 0.005 #the threshold to which the inertia moments of molecules are still considered close enough
looseness = 0.005 #the threshold to which the chemical environments of atoms within molecules are still considered close enough
basis = 'ccpvdz' #'def2tzvp' 'cc-pCVDZ'??? Basis set for QM calculations
representation ='yukawa' #'yukawa'# 'atomic_Coulomb' # 'exaggerated_atomic_Coulomb' #Atomic representations
standard_yukawa_range = -1 # -1 is inf <=> Coulomb potential # <10 <=> bullshit
PathToNauty27r1 = '/home/simon/nauty27r1/'
PathToQM9XYZ = '/home/simon/QM9/XYZ/'

elements = {'Ghost':0,'H':1, 'He':2,
'Li':3, 'Be':4, 'B':5, 'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15, 'S':16, 'Cl':17, 'Ar':18,
'K':19, 'Ca':20, 'Ga':31, 'Ge':32, 'As':33, 'Se':34, 'Br':35, 'Kr':36,
'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25, 'Fe':26, 'Co':27, 'Ni':28, 'Cu':29, 'Zn':30,}

inv_elements = {v: k for k, v in elements.items()}

'''Below are all the partitions of splitting m_tot = np.sum(dZ_all[i])
atoms in a pure (i.e. uncolored/isoatomic) molecule in n=len(dZ_all[i]) partitions
for dZ_max = 3 up to m_tot = 8 and n = 2 and 3'''
m_possibilities = np.array([
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
dZ_possibilities = np.array([
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

#Monkey patching PySCF's qmmm:
def add_qmmm(calc, mol, Z):
    mf = qmmm.mm_charge(calc, mol.atom_coords()*0.52917721067, Z)
    def energy_nuc(self):
        q = mol.atom_charges().copy().astype(np.float)
        q += Z
        return mol.energy_nuc(q)
    mf.energy_nuc = energy_nuc.__get__(mf, mf.__class__)
    return mf

#Dictionary which works as a dump for all previously computed energies
already_compt = {}

#ALL BASIC FUNCTIONS WITHOUT ANY DEPENDENCIES-----------------------------------
def delta(i,j):
    #Kronecker Delta
    if i == j:
        return 1
    else:
        return 0

def are_close_scalars(a,b,looseness):
    value = False
    if abs(a-b) < looseness:
        value = True
    return value

def are_close_lists(a,b):
    value = True
    for i in range(len(a)):
        if abs(a[i]-b[i]) > tolerance*np.sqrt(len(a)): #The prefactor allows less closer arrays to appear so.
            value = False
    return value

def rot(angle_x, angle_y, angle_z):
    return np.array(   [[np.cos(angle_z)*np.cos(angle_y),   np.cos(angle_z)*np.sin(angle_y)*np.sin(angle_x) - np.sin(angle_z)*np.cos(angle_x),  np.cos(angle_z)*np.sin(angle_y)*np.cos(angle_x) + np.sin(angle_z)*np.sin(angle_x)],
                        [np.sin(angle_z)*np.cos(angle_y),   np.sin(angle_z)*np.sin(angle_y)*np.sin(angle_x) + np.cos(angle_z)*np.cos(angle_x),  np.sin(angle_z)*np.sin(angle_y)*np.cos(angle_x) - np.cos(angle_z)*np.sin(angle_x)],
                        [-np.sin(angle_y),                  np.cos(angle_y)*np.sin(angle_x),                                                    np.cos(angle_y)*np.cos(angle_x)]])


def center_mole(mole, angle_aligning=True, angle=None):
    """

    """
    #Centers a molecule
    sum = [0,0,0]
    N = len(mole)
    result = mole.copy()
    for i in range(N):
        sum[0] += result[i][1]
        sum[1] += result[i][2]
        sum[2] += result[i][3]
    sum = np.multiply(sum, 1/N)
    #print(sum)
    for i in range(N):
        result[i][1:] = np.subtract(result[i][1:],sum)
    if not angle_aligning:
        return result
    else:
        #Take the first two non-linearly dep. points to rotate molecule into xy plane
        if N < 3:
            return result
        else:
            limit = 0.0001
            angle_x = 1
            angle_y = 1
            while abs(angle_x) > limit or abs(angle_y) > limit:
                for i in range(N-2):
                    if np.linalg.norm(result[i][1:]) < limit: #points too close to origin are plotted anyways
                        continue
                    else:
                        r1 = np.array(result[i][1:], copy=True)-np.array(result[i+1][1:], copy=True)
                        r2 = np.array(result[i+1][1:], copy=True) - np.array(result[i+2][1:], copy=True)
                        normal = np.cross(r1,r2)
                        normal_length = np.linalg.norm(normal)
                        if abs(normal_length) > 0.00001: #We found a suitable cadidate vector
                            break
                if abs(normal_length) <= 0.00001:
                    #print(result)
                    return result #Not one (!) suitable candidate found. The molecule is a straight line
                angle_x = np.arcsin(normal[1]/normal_length)
                angle_y = np.arcsin(normal[0]/normal_length)
                #print('aligning... [x: '+str(angle_x*180/np.pi)+', y: '+str(angle_y*180/np.pi)+']')
                R = rot(angle_x, angle_y,0)
                result_rotated = mole.copy()
                #print(R)
                #print(R, result[0])
                for i in range(N):
                    vector = np.matmul(R,result[i][1:])
                    for j in [1,2,3]:
                        result_rotated[i][j] = vector[j-1]
                result = result_rotated.copy()
    if angle == None:
        return result
    else:
        if len(angle) != 3:
            raise ValueError("angle needs to be list with 3 elements!")
        else:
            angle_x = angle[0]*np.pi/180
            angle_y = angle[1]*np.pi/180
            angle_z = angle[2]*np.pi/180
            #print(angle_x, angle_y)
            R = rot(angle_x, angle_y, angle_z)
            result_rotated = mole.copy()
            #print(R)
            #print(R, result[0])
            for i in range(N):
                vector = np.matmul(R,result[i][1:])
                for j in [1,2,3]:
                    result_rotated[i][j] = vector[j-1]
            #print(result_rotated)
            return result_rotated

def Coulomb_matrix(mole):
    #returns the Coulomb matrix of a given molecule
    N = len(mole)
    result = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if (j == i):
                charge = elements[mole[i][0]]
                summand = 0.5*pow(charge, 2.4)
                #print(summand)
                result[i][i] = summand
            else:
                summand = elements[mole[i][0]]*elements[mole[j][0]]/np.linalg.norm(np.subtract(mole[i][1:],mole[j][1:]))
                #print(summand) #Find out about the size of the summands
                result[i][j] = summand
    return result

def exaggerated_Coulomb_matrix(mole):
    #returns an exaggerated Coulomb matrix of a given molecule, sqrt of inverse distance, charges are to 4-th power now.
    #The elements are gated, such that any discussion of electronic similarity can be restricted to an atoms direct neighborhood
    N = len(mole)
    result = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if (j == i):
                charge = elements[mole[i][0]]
                summand = 0.5*pow(charge, 4.8)
                #print(summand)
                result[i][i] = summand
            else:
                summand = pow(elements[mole[i][0]]*elements[mole[j][0]],2)/pow(np.linalg.norm(np.subtract(mole[i][1:],mole[j][1:])),0.5)
                #print(summand) #Find out about the size of the summands
                result[i][j] = summand
    return result

def yukawa_matrix(mole, yukawa_range = standard_yukawa_range):
    N = len(mole)
    result = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if (j == i):
                charge = elements[mole[i][0]]
                summand = 0.5*pow(charge, 2.4)
                #print(summand)
                result[i][i] = summand
            else:
                r = np.linalg.norm(np.subtract(mole[i][1:],mole[j][1:]))
                if yukawa_range > 0:
                    summand = elements[mole[i][0]]*elements[mole[j][0]]*np.exp(-r/yukawa_range)/r
                else:
                    summand = elements[mole[i][0]]*elements[mole[j][0]]/r
                #print(summand) #Find out about the size of the summands
                result[i][j] = summand
    return result

def atomrep(mole, representation=representation, yukawa_range=standard_yukawa_range):
    N = len(mole)
    if representation == 'atomic_Coulomb':
        '''returns the sum over rows/columns of the Coulomb matrix.
        Thus, each atom is assigned its Coulombic neighborhood'''
        matrix = Coulomb_matrix(mole)
    elif representation == 'exaggerated_atomic_Coulomb':
        matrix = exaggerated_Coulomb_matrix(mole)
    elif representation == 'yukawa':
        matrix = yukawa_matrix(mole, yukawa_range = yukawa_range)
    #Calculate the norm:
    sum = 0
    result = []
    for i in range(N):
        for j in range(N):
            sum += matrix[i][j]**2
        sum = sum**0.5
        result.append(sum)
        sum = 0
    return result

def atomrep_inertia_tensor(mole, representation=representation):
    #Calculate an inertia tensor but with atomrep instead of masses
    N = len(mole)
    CN = atomrep(mole, representation=representation)
    result_tensor = np.zeros((3,3))
    sum = 0
    for i in range(3):
        for j in range(i+1):
            for k in range(N):
                sum += CN[k]*(np.dot(mole[k][1:], mole[k][1:])*delta(i,j) - mole[k][1+i]*mole[k][1+j])
            result_tensor[i][j] = sum
            result_tensor[j][i] = sum
            sum = 0
    return result_tensor

def atomrep_inertia_moment(mole, representation=representation):
    #Calculate the inertia moments of a molecule with atomrep instead of masses
    #and sort them in ascending order
    w,v = np.linalg.eig(atomrep_inertia_tensor(mole,representation=representation))
    #Only the eigen values are needed, v is discarded
    moments = np.sort(w)
    return moments

def array_compare(arr1, arr2):
    '''arr1 = [...]
    arr2 = [[...],[...],[...],...]
    Is there an approximate copy of arr1 in arr2'''
    within = False
    for i in range(len(arr2)):
        if are_close_lists(arr1, arr2[i]):
            within = True
    return within

def sum_formula(array_of_atoms):
    values, counts = np.unique(np.array(array_of_atoms), return_counts=True)
    formula = ''
    for i in range(len(values)):
        if values[i] == 'C':
            if counts[i] > 1:
                formula = str(counts[i]) + formula
            formula = 'C' + formula
        else:
            formula += str(values[i])
            if counts[i] > 1:
                formula += str(counts[i])
    return formula

def bond_count(edge_layout, elements_at_index):
    '''Count all the bonds of edge_layout with the vertices defined in
    elements_at_index and return them'''
    #First: Translate all edges into bond-strings
    collect_bonds = []
    for edge in edge_layout:
        bond_vertices = [inv_elements[i] for i in np.sort([elements[elements_at_index[edge[0]]], elements[elements_at_index[edge[1]]]])]
        bond_name = bond_vertices[0]+bond_vertices[1]
        collect_bonds = np.append(collect_bonds, bond_name)
    return collect_bonds

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


def geom_hash(input_geometry, Z):
    #Assigns a hash value to a geometry
    hash = ''
    N = len(input_geometry)
    for i in range(N):
        hash += '___'
        hash += input_geometry[i][0]+str(Z[i])
        hash += '___'
        for j in [1,2,3]:
            hash += str(round(input_geometry[i][j],3))
    return hash


#EVERYTHING BELONGING TO ARBITRARY PRECISION------------------------------------

def orbital_list(element):
    if element in ['H','He']:
        return ["1s"]
    if element in ['Li','Be']:
        return ["1s", "2s"]
    if element in ["B","C","N","O","F","Ne"]:
        return ["1s", "2s", "2p"]
    if element in ['Na','Mg']:
        return ["1s", "2s", "2p", "3s"]
    if element in ['Al', 'Si', 'P', 'S','Cl','Ar']:
        return ["1s", "2s", "2p", "3s", "3p"]
    if element in ['K', 'Ca']:
        return ["1s", "2s", "2p", "3s", "3p", "4s"]

def MAGtoMole(MAG):
    Mole = []
    sum_elec = 0
    for i in range(len(MAG.geometry)):
        Mole.append(Atom(MAG.geometry[i][0], (mpmath.mpf(MAG.geometry[i][1]), mpmath.mpf(MAG.geometry[i][2]), mpmath.mpf(MAG.geometry[i][3])), elements[MAG.geometry[i][0]], orbital_list(MAG.geometry[i][0])))
        sum_elec += elements[MAG.geometry[i][0]]
    return Mole, sum_elec


#CLASS DEFINITION OF MoleAsGraph------------------------------------------------

class MoleAsGraph:
    def __init__(self, name, edge_layout, elements_at_index, geometry, without_hydrogen=False):
        '''Caution: For all methods so far, the indexing of the geometry and the graphs (i.e. all
        remaining attributes) does not have to match! Keep this in mind: They may have different
        indices!!!!!! They do not even have to be the same molecules but then the Constructor may
        print a Warning'''
        self.name = name
        self.edge_layout = edge_layout #edge_layout = [[site_index, connected site index (singular!!!)], [...,...], [...,...]]
        self.elements_at_index = elements_at_index #Which element is at which vertex number
        self.geometry = center_mole(geometry, angle_aligning=False) #The usual xyz representation
        if len(np.unique(edge_layout)) == len(elements_at_index):
            self.number_atoms = len(elements_at_index)
        else:
            print('Number of vertices and number of elements do not match!')
        if len(edge_layout) != 0:
            self.max_index = max([max(sublist) for sublist in edge_layout])
        else:
            self.max_index = 0
        if len(elements_at_index) != self.max_index+1:
            print("Number of atoms does not match naming of vertices: enumerate the vertices with integers without omissions!")
        self.orbits = self.get_orbits_from_graph()
        self.equi_atoms =  self.get_equi_atoms_from_geom(ignore_hydrogen = without_hydrogen)

        if without_hydrogen == True:
            N_before = len(self.geometry)
            count = 0
            while count < N_before:
                if self.geometry[count][0] == 'H':
                    #Delete H in geometry, refresh number_atoms
                    np.delete(self.geometry, count).tolist()
                    N_before -= 1
                else:
                    count += 1

        #This part can only be used if you are sure that the indexing of atoms in the graph is the same as in the xyz
        #self.orbits = np.array(self.get_equi_atoms_from_geom(), dtype=object)

        #If the indexing of vertices does not match the indexing of geometry:
        #if not np.array([elements_at_index[i] == geometry[i][0] for i in range(self.number_atoms)]).all():
        #    print("Warning in MoleAsGraph: Indexing of atoms in attribute 'geometry' does not match indexing in attribute 'elements_at_index' of molecule "+name)

    def get_number_automorphisms(self):
        #Prepare the graph
        g = igraph.Graph([tuple(v) for v in self.edge_layout])
        #Get all automorphisms with colored vertices according to self.elements_at_index
        automorphisms = g.get_automorphisms_vf2(color=[elements[self.elements_at_index[i]] for i in range(self.number_atoms)])
        #Get rid of all molecules without any orbits (the identity does not count):
        return len(automorphisms)


    def get_orbits_from_graph(self):
        #return [] #The caveman approach: If orbits are not needed, why bother
        """
        The vf2 algorithm pf igraph sometimes forces the OS to kill the process;
        hence, the extra bit of code up front
        """
        #Prepare the graph
        g = igraph.Graph([tuple(v) for v in self.edge_layout])
        #Get all automorphisms with colored vertices according to self.elements_at_index
        automorphisms = g.get_automorphisms_vf2(color=[elements[self.elements_at_index[i]] for i in range(self.number_atoms)])
        #Get rid of all molecules without any orbits (the identity does not count):
        if len(automorphisms) == 1:
            similars = [[]]
            return similars
        else:
            '''Any element of the list of vertices = set of elements, that moves along
            a fixed path upon the group acting on the set is part of an orbit. Since we know
            all allowed automorphisms, we can check for all the possible vertices at positions
            that are reached through the permutations = automorphisms'''
            #print('----------------')
            #print(self.name)
            #print(automorphisms)
            #print('----------------')
            similars = np.array(np.zeros(self.number_atoms), dtype=object)
            for i in range(self.number_atoms):
                orbit = []
                for j in range(len(automorphisms)):
                    orbit = np.append(orbit, automorphisms[j][i])
                orbit = np.unique(orbit)
                #print(orbit)
                similars[i] = orbit
            #Delete all similars which include only one atom:
            num_similars = 0
            while num_similars < len(similars):
                if len(similars[num_similars])>1:
                    num_similars += 1
                else:
                    similars = np.delete(similars, num_similars, axis = 0)
            #Unique the list:
            sites = [list(map(int,x)) for x in set(tuple(x) for x in similars)]
        return sites

    def get_equi_atoms_from_geom(self, ignore_hydrogen = False):
        mole = self.geometry.copy()
        #Sort CN and group everything together that is not farther than tolerance
        CN = atomrep(center_mole(mole, angle_aligning=False))
        #In atomrep, the hydrogens have been considered. Now, get rid of them again
        indices = np.argsort(CN).tolist()
        indices2 = np.copy(indices).tolist()
        lst = []
        similars = []
        for i in range(len(indices)-1):
            if i not in indices2:
                continue
            lst.append(indices[i])
            indices2.remove(i)
            for j in range(i+1,len(indices)):
                if are_close_scalars(CN[indices[i]],CN[indices[j]],looseness):
                    lst.append(indices[j])
                    indices2.remove(j)
            similars.append(lst)
            lst = []
        #print('Flag')
        #print(similars)
        #If without hydrogen: Delete all hydrogens
        if ignore_hydrogen:
            num_similars = 0
            while num_similars < len(similars):
                num_members = 0
                while num_members < len(similars[num_similars]):
                    if self.geometry[similars[num_similars][num_members]][0] != 'H':
                        num_members += 1
                    else:
                        del similars[num_similars][num_members]
                num_similars += 1
        #Delete all similars which include only one atom or none at all:
        num_similars = 0
        while num_similars < len(similars):
            if len(similars[num_similars])>1:
                num_similars += 1
            else:
                del similars[num_similars]
        return similars

    def get_nuclear_energy(self, Z=[]):
        #Calculate the nuclear energy of the molecule
        sum = 0
        if len(Z) == 0:
            Z = np.zeros((self.number_atoms)).tolist()
        for i in range(self.number_atoms):
            for j in range(i+1,self.number_atoms):
                sum += (elements[self.geometry[i][0]]+Z[i])*(elements[self.geometry[j][0]]+Z[j])/np.linalg.norm(np.subtract(self.geometry[i][1:],self.geometry[j][1:]))
        return sum*0.529177210903 #Result needs to be in Ha, and the length has been in Angstrom

    def get_molecular_norm(self):
        return np.linalg.norm(atomrep_inertia_tensor(self.geometry.copy()))

    def print_atomic_norms(self, yukawa_range = standard_yukawa_range, tolist=False, with_hydrogen=False):
        N = self.number_atoms
        rep_norm = atomrep(self.geometry, yukawa_range=yukawa_range)
        result = []
        if not tolist:
            for i in range(N):
                if self.geometry[i][0] == 'H' and with_hydrogen == False:
                    continue
                else:
                    print(self.name+'\t'+self.geometry[i][0]+'\t'+str(i)+'\t'+str(rep_norm[i]))
                    #Name   Chemical Element    Index   SMILES  Norm
        else:
            for i in range(N):
                if self.geometry[i][0] == 'H' and with_hydrogen == False:
                    continue
                else:
                    result.append([self.name, self.geometry[i][0], i,rep_norm[i]])
            return result

    def get_total_energy(self, Z=[], basis=basis):
        if geom_hash(self.geometry, Z) in already_compt:
            return already_compt[geom_hash(self.geometry, Z)]
        else:
            #Z are additional charges; electrons are accounted for Z
            #PARSE THE HYDROGENS!!!!!
            atom_string = ''
            overall_charge = 0
            if len(Z) == 0:
                Z = np.zeros((self.number_atoms)).tolist()
            for i in range(len(self.geometry)): #get the atoms and their coordinates
                atom_string += self.geometry[i][0]
                overall_charge += elements[self.geometry[i][0]]+Z[i]
                for j in [1,2,3]:
                    atom_string += ' ' + str(self.geometry[i][j])
                atom_string += '; '
            mol = gto.Mole()
            mol.unit = 'Angstrom'
            mol.atom = atom_string[:-2]
            mol.basis = basis
            mol.verbose = 0
            mol.nelectron = int(overall_charge+0.001) #Avoid .999 when adding thirds
            mol.build()
            calc = add_qmmm(scf.RHF(mol), mol, Z)
            hfe = calc.kernel(verbose=0)
            total_energy = calc.e_tot
            already_compt.update({geom_hash(self.geometry, Z):total_energy})
            return total_energy

    def get_electronic_energy(self, Z=[], basis=basis):
        return self.get_total_energy(Z, basis=basis) - self.get_nuclear_energy(Z)

    def plot_rho_2D(self, Z=[], z_offset = 0, title = '', basis=basis):
        #PARSE THE HYDROGENS!!!!!
        atom_string = ''
        charge = 0
        if len(Z) == 0:
            Z = np.zeros((self.number_atoms)).tolist()
        for i in range(len(self.geometry)): #get the atoms and their coordinates
            atom_string += self.geometry[i][0]
            charge += elements[self.geometry[i][0]]+Z[i]
            for j in [1,2,3]:
                atom_string += ' ' + str(self.geometry[i][j])
            atom_string += '; '
        mol = gto.Mole()
        mol.unit = 'Angstrom'
        mol.atom = atom_string[:-2]
        mol.basis = basis
        mol.verbose = 0
        mol.nelectron = int(charge+0.001) #Avoid .999 when adding thirds
        mol.build()
        calc = add_qmmm(scf.RHF(mol), mol, Z)
        hfe = calc.kernel(verbose=0)
        dm1_ao = calc.make_rdm1()
        grid = pyscf.dft.gen_grid.Grids(mol)
        grid.level = 3
        grid.build()
        ao_value = pyscf.dft.numint.eval_ao(mol, grid.coords, deriv=0)
        rhos = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_ao, xctype="LDA")
        #Now we have all rhos at grid.coords; plot them:
        x = []
        y = []
        density = []
        z_filter = 0.1 #The tolerance in z when something is still considered part of the plane
        for i in range(len(grid.coords)):
            if abs(grid.coords[i][2] - z_offset/0.52917721067) < z_filter:
                x.append(grid.coords[i][0])
                y.append(grid.coords[i][1])
                density.append(rhos[i])
                #z.append(grid.coords[i][2])
        #print(x)
        #plt.scatter(x, y, c=density)
        #plt.show()
        #Source: https://earthscience.stackexchange.com/questions/12057/how-to-interpolate-scattered-data-to-a-regular-grid-in-python
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        # target grid to interpolate to
        xi = np.arange(x_min,x_max,0.1)
        yi = np.arange(y_min,y_max,0.1)
        xi,yi = np.meshgrid(xi,yi)
        # interpolate
        zi = griddata((x,y),density,(xi,yi),method='nearest')
        #zi = griddata((x,y),density,(xi,yi),method='cubic')
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.contourf(xi,yi,zi,np.arange(0,0.81,0.05))
        plt.colorbar(label=r'$\rho$ [$a_0^{-3}$]')
        #Find maximum distance of molecule to scale window
        max_value = 0
        for i in range(self.number_atoms):
            for j in [1,2]: #The z axis is not of interest here
                if abs(self.geometry[i][j]) > max_value:
                    max_value = abs(self.geometry[i][j])
        max_value *= 1.6/0.52917721067 #in Bohr radii
        plt.xlim([-max_value,max_value])
        plt.ylim([-max_value,max_value])
        #plt.plot(x,y,'k.')
        plt.title(title)
        plt.xlabel(r'x [$a_0$]')
        plt.ylabel(r'y [$a_0$]')
        plt.savefig('rho_plots/'+self.name + '_rho.png',dpi=150)
        plt.close(fig)


    def plot_rho_3D(self, Z=[], title = '', basis=basis):
        #PARSE THE HYDROGENS!!!!!
        atom_string = ''
        charge = 0
        if len(Z) == 0:
            Z = np.zeros((self.number_atoms)).tolist()
        for i in range(len(self.geometry)): #get the atoms and their coordinates
            atom_string += self.geometry[i][0]
            charge += elements[self.geometry[i][0]]+Z[i]
            for j in [1,2,3]:
                atom_string += ' ' + str(self.geometry[i][j])
            atom_string += '; '
        mol = gto.Mole()
        mol.unit = 'Angstrom'
        mol.atom = atom_string[:-2]
        mol.basis = basis
        mol.verbose = 0
        mol.nelectron = int(charge+0.001) #Avoid .999 when adding thirds
        mol.spin = 0
        mol.build()
        calc = add_qmmm(scf.RHF(mol), mol, Z)
        hfe = calc.kernel(verbose=0)
        dm1_ao = calc.make_rdm1()
        grid = pyscf.dft.gen_grid.Grids(mol)
        grid.level = 3
        grid.build()
        ao_value = pyscf.dft.numint.eval_ao(mol, grid.coords, deriv=0)
        rhos = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_ao, xctype="LDA")
        #Crate a GIF from all the files (all slices)
        filenames = []
        counter = 0
        #Find maximum distance of molecule to scale window
        max_value = 0
        for i in range(self.number_atoms):
            for j in [1,2]: #The z axis is not of interest here
                if abs(self.geometry[i][j]) > max_value:
                    max_value = abs(self.geometry[i][j])
        max_value *= 1.6/0.52917721067 #in Bohr radii
        for z_offset in np.arange(-max_value*0.25,max_value*0.25,0.05):
            #Now we have all rhos at grid.coords; plot them:
            x = []
            y = []
            density = []
            z_filter = 0.1 #The tolerance in z when something is still considered part of the plane
            for i in range(len(grid.coords)):
                if abs(grid.coords[i][2] - z_offset/0.52917721067) < z_filter:
                    x.append(grid.coords[i][0])
                    y.append(grid.coords[i][1])
                    density.append(rhos[i])
                    #z.append(grid.coords[i][2])
            #print(x)
            #plt.scatter(x, y, c=density)
            #plt.show()
            #Source: https://earthscience.stackexchange.com/questions/12057/how-to-interpolate-scattered-data-to-a-regular-grid-in-python
            x_min = min(x)
            x_max = max(x)
            y_min = min(y)
            y_max = max(y)
            # target grid to interpolate to
            xi = np.arange(x_min,x_max,0.1)
            yi = np.arange(y_min,y_max,0.1)
            xi,yi = np.meshgrid(xi,yi)
            # interpolate
            zi = griddata((x,y),density,(xi,yi),method='nearest')
            #zi = griddata((x,y),density,(xi,yi),method='cubic')
            # plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.contourf(xi,yi,zi,np.arange(0,0.81,0.05))
            plt.colorbar(label=r'$\rho$ [$a_0^{-3}$]')
            plt.xlim([-max_value,max_value])
            plt.ylim([-max_value,max_value])
            #plt.plot(x,y,'k.')
            plt.title(title)
            plt.xlabel(r'x [$a_0$]')
            plt.ylabel(r'y [$a_0$]')
            filename = f'weird_named_png'+str(counter)+'.png'
            plt.savefig(filename,dpi=150)
            plt.close(fig)
            filenames.append(filename)
            counter += 1
        # build gif
        with imageio.get_writer('rho_plots/'+self.name + '_rho.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        # Remove files
        for filename in set(filenames):
            os.remove(filename)
        #Source: https://towardsdatascience.com/basics-of-gifs-with-pythons-matplotlib-54dd544b6f30


    def plot_delta_rho_2D(self, dZ1, dZ2, Z = [], z_offset = 0, title = '', basis=basis):
        #PARSE THE HYDROGENS!!!!!
        atom_string = ''
        charge1 = 0
        charge2 = 0
        if len(Z) == 0:
            Z = np.zeros((len(dZ2))).tolist()
        if len(dZ1) != len(dZ2) or len(dZ1) != self.number_atoms or len(dZ1) != len(Z):
            raise ValueError("Z, dZ1, dZ2 and "+self.name+" must have the same number of atoms.")
        for i in range(len(self.geometry)): #get the atoms and their coordinates
            atom_string += self.geometry[i][0]
            charge1 += elements[self.geometry[i][0]]+Z[i]
            charge2 += elements[self.geometry[i][0]]+Z[i]
            for j in [1,2,3]:
                atom_string += ' ' + str(self.geometry[i][j])
            atom_string += '; '
        #Create two distinct molecules
        mol1 = gto.Mole()
        mol1.unit = 'Angstrom'
        mol1.atom = atom_string[:-2]
        mol1.basis = basis
        mol1.nelectron = int(charge1)
        mol1.verbose = 0
        mol1.build()
        mol2 = gto.Mole()
        mol2.unit = 'Angstrom'
        mol2.atom = atom_string[:-2]
        mol2.basis = basis
        mol2.nelectron = int(charge2)
        mol2.verbose = 0
        mol2.build()

        calc1 = add_qmmm(scf.RHF(mol1), mol1, [Z[j]+dZ1[j] for j in range(len(Z))])
        hfe1 = calc1.kernel(verbose=0)
        dm1_ao_1 = calc1.make_rdm1()
        grid1 = pyscf.dft.gen_grid.Grids(mol1)
        grid1.level = 3
        grid1.build()
        ao_value1 = pyscf.dft.numint.eval_ao(mol1, grid1.coords, deriv=0)
        rhos1 = pyscf.dft.numint.eval_rho(mol1, ao_value1, dm1_ao_1, xctype="LDA")
        #Now we have all rhos at grid.coords; plot them:
        x1 = []
        y1 = []
        density1 = []
        z_filter = 0.1
        for i in range(len(grid1.coords)):
            if abs(grid1.coords[i][2] - z_offset/0.52917721067) < z_filter:
                x1.append(grid1.coords[i][0])
                y1.append(grid1.coords[i][1])
                density1.append(rhos1[i])

        calc2 = add_qmmm(scf.RHF(mol2), mol2, [Z[j]+dZ2[j] for j in range(len(Z))])
        hfe2 = calc2.kernel(verbose=0)
        dm1_ao_2 = calc2.make_rdm1()
        grid2 = pyscf.dft.gen_grid.Grids(mol2)
        grid2.level = 3
        grid2.build()
        ao_value2 = pyscf.dft.numint.eval_ao(mol2, grid2.coords, deriv=0)
        rhos2 = pyscf.dft.numint.eval_rho(mol2, ao_value2, dm1_ao_2, xctype="LDA")
        #Now we have all rhos at grid.coords; plot them:
        x2 = []
        y2 = []
        density2 = []
        for i in range(len(grid2.coords)):
            if abs(grid2.coords[i][2] - z_offset/0.52917721067) < z_filter:
                x2.append(grid2.coords[i][0])
                y2.append(grid2.coords[i][1])
                density2.append(rhos2[i])
        #print(x)
        #plt.scatter(x, y, c=density)
        #plt.show()
        #Source: https://earthscience.stackexchange.com/questions/12057/how-to-interpolate-scattered-data-to-a-regular-grid-in-python
        x_min = min(x1)
        x_max = max(x1)
        y_min = min(y1)
        y_max = max(y1)
        # target grid to interpolate to
        xi = np.arange(x_min,x_max,0.1)
        yi = np.arange(y_min,y_max,0.1)
        xi,yi = np.meshgrid(xi,yi)
        # interpolate
        zi = griddata((x1,y1),density1,(xi,yi),method='nearest') - griddata((x2,y2),density2,(xi,yi),method='nearest')
        #zi = griddata((x1,y1),density1,(xi,yi),method='cubic') - griddata((x2,y2),density2,(xi,yi),method='cubic')
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.contourf(xi,yi,zi,np.arange(-0.3,0.31,0.05))
        plt.colorbar(label=r'$\rho$ [$a_0^{-3}$]')
        #Find maximum distance of molecule to scale window
        max_value = 0
        for i in range(self.number_atoms):
            for j in [1,2]:
                if abs(self.geometry[i][j]) > max_value:
                    max_value = abs(self.geometry[i][j])
        max_value *= 1.6/0.52917721067 #in Bohr radii
        plt.xlim([-max_value,max_value])
        plt.ylim([-max_value,max_value])
        #plt.plot(x,y,'k.')
        plt.title(title)
        plt.xlabel(r'x [$a_0$]')
        plt.ylabel(r'y [$a_0$]')
        plt.savefig('rho_plots/'+self.name + '_delta_rho.png',dpi=150)
        plt.close(fig)


    def plot_delta_rho_3D(self, dZ1, dZ2, Z = [], z_offset = 0, title = '', basis=basis):
        #PARSE THE HYDROGENS!!!!!
        atom_string = ''
        charge1 = 0
        charge2 = 0
        if len(Z) == 0:
            Z = np.zeros((len(dZ2))).tolist()
        if len(dZ1) != len(dZ2) or len(dZ1) != self.number_atoms or len(Z) != len(dZ1):
            raise ValueError("dZ1, dZ2 and "+self.name+" must have the same number of atoms.")
        for i in range(len(self.geometry)): #get the atoms and their coordinates
            atom_string += self.geometry[i][0]
            charge1 += elements[self.geometry[i][0]]+Z[i]
            charge2 += elements[self.geometry[i][0]]+Z[i]
            for j in [1,2,3]:
                atom_string += ' ' + str(self.geometry[i][j])
            atom_string += '; '
        #Create two distinct molecules
        mol1 = gto.Mole()
        mol1.unit = 'Angstrom'
        mol1.atom = atom_string[:-2]
        mol1.basis = basis
        mol1.nelectron = int(charge1)
        mol1.verbose = 0
        mol1.build()
        mol2 = gto.Mole()
        mol2.unit = 'Angstrom'
        mol2.atom = atom_string[:-2]
        mol2.basis = basis
        mol2.nelectron = int(charge2)
        mol2.verbose = 0
        mol2.build()
        Z1 = [int(Z[j]+dZ1[j]) for j in range(len(Z))]
        calc1 = add_qmmm(scf.RHF(mol1), mol1, dZ1)
        hfe1 = calc1.kernel(verbose=0)
        dm1_ao_1 = calc1.make_rdm1()
        grid1 = pyscf.dft.gen_grid.Grids(mol1)
        grid1.level = 3
        grid1.build()
        ao_value1 = pyscf.dft.numint.eval_ao(mol1, grid1.coords, deriv=0)
        rhos1 = pyscf.dft.numint.eval_rho(mol1, ao_value1, dm1_ao_1, xctype="LDA")
        Z2 = [Z[j]+dZ2[j] for j in range(len(Z))]
        calc2 = add_qmmm(scf.RHF(mol2), mol2, Z2)
        hfe2 = calc2.kernel(verbose=0)
        dm1_ao_2 = calc2.make_rdm1()
        grid2 = pyscf.dft.gen_grid.Grids(mol2)
        grid2.level = 3
        grid2.build()
        ao_value2 = pyscf.dft.numint.eval_ao(mol2, grid2.coords, deriv=0)
        rhos2 = pyscf.dft.numint.eval_rho(mol2, ao_value2, dm1_ao_2, xctype="LDA")
        #Now we have all rhos at grid.coords; plot them:
        #Find maximum distance of molecule to scale window
        max_value = 0
        for i in range(self.number_atoms):
            for j in [1,2]: #The z axis is not of interest here
                if abs(self.geometry[i][j]) > max_value:
                    max_value = abs(self.geometry[i][j])
        max_value *= 1.6/0.52917721067 #in Bohr radii
        counter = 0
        filenames = []
        for z_offset in np.arange(-max_value*0.25,max_value*0.25,0.05):
            #Now we have all rhos at grid.coords; plot them:
            x1 = []
            y1 = []
            density1 = []
            z_filter = 0.1
            for i in range(len(grid1.coords)):
                if abs(grid1.coords[i][2] - z_offset/0.52917721067) < z_filter:
                    x1.append(grid1.coords[i][0])
                    y1.append(grid1.coords[i][1])
                    density1.append(rhos1[i])
            #Now we have all rhos at grid.coords; plot them:
            x2 = []
            y2 = []
            density2 = []
            for i in range(len(grid2.coords)):
                if abs(grid2.coords[i][2] - z_offset/0.52917721067) < z_filter:
                    x2.append(grid2.coords[i][0])
                    y2.append(grid2.coords[i][1])
                    density2.append(rhos2[i])
            #print(x)
            #plt.scatter(x, y, c=density)
            #plt.show()
            #Source: https://earthscience.stackexchange.com/questions/12057/how-to-interpolate-scattered-data-to-a-regular-grid-in-python
            x_min = min(x1)
            x_max = max(x1)
            y_min = min(y1)
            y_max = max(y1)
            # target grid to interpolate to
            xi = np.arange(x_min,x_max,0.1)
            yi = np.arange(y_min,y_max,0.1)
            xi,yi = np.meshgrid(xi,yi)
            # interpolate
            zi = griddata((x1,y1),density1,(xi,yi),method='nearest') - griddata((x2,y2),density2,(xi,yi),method='nearest')
            #zi = griddata((x1,y1),density1,(xi,yi),method='cubic') - griddata((x2,y2),density2,(xi,yi),method='cubic')
            # plot
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.contourf(xi,yi,zi,np.arange(-0.3,0.31,0.05))
            plt.colorbar(label=r'$\rho$ [$a_0^{-3}$]')
            #Find maximum distance of molecule to scale window
            max_value = 0
            for i in range(self.number_atoms):
                for j in [1,2]:
                    if abs(self.geometry[i][j]) > max_value:
                        max_value = abs(self.geometry[i][j])
            max_value *= 1.6/0.52917721067 #in Bohr radii
            plt.xlim([-max_value,max_value])
            plt.ylim([-max_value,max_value])
            #plt.plot(x,y,'k.')
            plt.title(title)
            plt.xlabel(r'x [$a_0$]')
            plt.ylabel(r'y [$a_0$]')
            filename = f'weird_named_png'+str(counter)+'.png'
            plt.savefig(filename,dpi=150)
            plt.close(fig)
            filenames.append(filename)
            counter += 1

        # build gif
        with imageio.get_writer('rho_plots/'+self.name + '_delta_rho.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        # Remove files
        for filename in set(filenames):
            os.remove(filename)
        #Source: https://towardsdatascience.com/basics-of-gifs-with-pythons-matplotlib-54dd544b6f30


    def fill_hydrogen_valencies(self, input_PathToFile):
        '''If the xyz file from which this molecule originates is known,
        the valencies can be filled with the hydrogens as given in the file.
        Add the geometric information line by line with an offset subtracted!!!!
        and for each line, make one vertex and one edge to the closest heavy atom'''
        #check if file is present
        if os.path.isfile(input_PathToFile):
            #open text file in read mode
            f = open(input_PathToFile, "r")
            data = f.read()
            f.close()
        else:
            print('File', input_PathToFile, 'not found.')
        #Initalize
        name = self.name
        new_geometry = self.geometry.copy()
        new_elements_at_index = self.elements_at_index.copy()
        new_edge_layout = self.edge_layout.copy()
        N_heavy = self.number_atoms #number of previously give atoms, all heavy
        N = int(data.splitlines(False)[0]) #number of atoms including hydrogen
        #Compare self.geometry to the first entry of the file (= line 2) -> obtain offset
        line2 = data.splitlines(False)[2].split('\t')
        coord_orig = [float(line2[1]),float(line2[2]),float(line2[3])]
        offset = [0,0,0]
        for i in [0,1,2]:
            offset[i] = coord_orig[i] - self.geometry[0][i+1]
        for i in range(2,N+2): #get only the hydrogens and their coordinates
            line = data.splitlines(False)[i]
            #Check for hydrogen specifically
            x = line.split('\t')
            if x[0] != 'H':
                continue
            else:
                new_geometry.append(['H', float(x[1])-offset[0],float(x[2])-offset[1],float(x[3])-offset[2]])
                new_elements_at_index.append('H')
            #Now: find the index of the atom with the shortest distance
            shortest_distance = 100000
            for j in range(N_heavy):
                distance = np.linalg.norm(np.subtract(self.geometry[j][1:],new_geometry[-1][1:]))
                if distance < shortest_distance:
                    shortest_distance = distance
                    index_of_shortest = j
            new_edge_layout.append([int(index_of_shortest),len(new_geometry)-1])
        return MoleAsGraph(name, new_edge_layout, new_elements_at_index, new_geometry)


    def get_total_energy_arbprec(self, digits=30):
        mpmath.mp.dps = digits
        molecule, N = MAGtoMole(self)
        return energy_tot_arbprec(molecule, N)

#MoleAsGraph EXAMPLES-----------------------------------------------------------
"""
anthracene = MoleAsGraph('Anthracene',
                        [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[6,7],[7,8],[8,9],[9,5],[7,10],[10,11],[11,12],[12,13],[13,8]],
                        ['C','C','C','C','C','C','C','C','C','C','C','C','C','C'],
                        [['C', 0,0,0.5], ['C', 0,0.8660254037844386467637231707,1], ['C', 0,2*0.8660254037844386467637231707,0.5],
                        ['C', 0,2*0.8660254037844386467637231707,-0.5], ['C', 0,0.8660254037844386467637231707,-1], ['C', 0,0,-0.5],
                        ['C', 0,-0.8660254037844386467637231707,1], ['C', 0,-2*0.8660254037844386467637231707,0.5], ['C', 0,-2*0.8660254037844386467637231707, -0.5], ['C', 0,-0.8660254037844386467637231707,-1],
                        ['C', 0,-3*0.8660254037844386467637231707,1], ['C', 0,-4*0.8660254037844386467637231707,0.5], ['C', 0,-4*0.8660254037844386467637231707, -0.5], ['C', 0,-3*0.8660254037844386467637231707,-1]])
benzene = MoleAsGraph(  'Benzene',
                        [[0,1],[1,2],[2,3],[3,4],[4,5],[0,5]],
                        ['C','C','C','C','C','C'],
                        [['C', 0,0,1], ['C', 0,0.8660254037844386467637231707,0.5], ['C', 0,0.8660254037844386467637231707,-0.5],
                        ['C', 0,0,-1], ['C', 0,-0.8660254037844386467637231707,-0.5], ['C', 0,-0.8660254037844386467637231707,0.5]])

cube = MoleAsGraph(     'Cube',
                        [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]],
                        ['C','C','C','C','C','C','C','C'],
                        [['C',0,0,0],['C',0,1,0],['C',0,1,1],['C',0,0,1],['C',1,0,0],['C',1,1,0],['C',1,1,1],['C',1,0,1]])

naphthalene = MoleAsGraph(  'Naphthalene',
                            [[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,6],[6,7],[7,8],[8,9],[9,5]],
                            ['C','C','C','C','C','C','C','C','C','C'],
                            [['C', 0,0,0.5], ['C', 0,0.8660254037844386467637231707,1], ['C', 0,2*0.8660254037844386467637231707,0.5],
                            ['C', 0,2*0.8660254037844386467637231707,-0.5], ['C', 0,0.8660254037844386467637231707,-1], ['C', 0,0,-0.5],
                            ['C', 0,-0.8660254037844386467637231707,1], ['C', 0,-2*0.8660254037844386467637231707,0.5], ['C', 0,-2*0.8660254037844386467637231707, -0.5], ['C', 0,-0.8660254037844386467637231707,-1]])

water = MoleAsGraph(        'Water',
                            [[0,1],[1,2]],
                            ['H', 'O', 'H'],
                            [['H', 0, +1.43233673, -0.96104039], ['O', 0, 0, 0.24026010], ['H', 0, -1.43233673, -0.96104039]])
"""

#PARSER FUNCTION FOR QM9--------------------------------------------------------
def parse_XYZtoMAG(input_PathToFile, with_hydrogen = False, angle_aligning=True, angle=None):
    '''MoleAsGraph instance returned'''
    #check if file is present
    if os.path.isfile(input_PathToFile):
        #open text file in read mode
        f = open(input_PathToFile, "r")
        data = f.read()
        f.close()
    else:
        print('File', input_PathToFile, 'not found.')
    #Get the name of the molecule
    MAG_name = input_PathToFile.split('/')[-1].split('.')[0]
    N = int(data.splitlines(False)[0]) #number of atoms including hydrogen
    #Get the geometry of the molecule
    mole = []
    N_heavyatoms = copy.deepcopy(N)
    elements_at_index = []
    for i in range(2,N+2): #get the atoms and their coordinates
        line = data.splitlines(False)[i]
        if line.split('\t')[0] == 'H':
            N_heavyatoms -= 1
        symbol = line.split('\t')[0]
        x = float(line.split('\t')[1].strip())
        y = float(line.split('\t')[2].strip())
        z = float(line.split('\t')[3].strip())
        mole.append([symbol,x,y,z])
        if not with_hydrogen and symbol == 'H':
            continue
        else:
            elements_at_index.append(symbol)
    #Find edge_layout:
    try:
        network = read_smiles(data.splitlines(False)[N+3].split('\t')[0], explicit_hydrogen=with_hydrogen)
    except:
        network = read_smiles(data.splitlines(False)[N+3].split(' ')[0], explicit_hydrogen=with_hydrogen)
    edge_layout = [list(v) for v in network.edges()]
    #print(mole)
    #print(edge_layout)
    """
    #All of this is only necessary if one explicitly introduces elements outside of the PSE
    N_hydrogen = N - N_heavyatoms
    Hydrogen_counter = 0
        for i in range(N):
            #get only the hydrogens and their coordinates
            if mole[i][0] == 'H':
                Hydrogen_counter += 1
                #find the index of the heavy atom with the shortest distance
                shortest_distance = 100000
                for j in range(N):
                    if mole[j][0] == 'H':
                        continue
                    else:
                        distance = np.linalg.norm(np.subtract(mole[j][1:],mole[i][1:]))
                        if distance < shortest_distance:
                            shortest_distance = distance
                            index_of_shortest = j
                edge_layout.append([int(index_of_shortest),int(N_heavyatoms+Hydrogen_counter-1)])
    print(edge_layout)
    """
    mole = center_mole(mole, angle_aligning=angle_aligning, angle=angle)
    if with_hydrogen:
        return MoleAsGraph(MAG_name, edge_layout, elements_at_index, mole)
    else:
        #Careful here: This MAG object has the graph properties without hydrogen, but geometry still with!
        return MoleAsGraph(MAG_name, edge_layout, elements_at_index, mole, without_hydrogen = True)


#ALL HIGHER LEVEL FUNCTIONS WITH VARIOUS DEPENDENCIES---------------------------
def dlambda_electronic_energy(mole, Z, dlambda, order):
    #Z is the deviation of the molecule from integer nuclear charges
    #dlambda is needed as the basis vector for the parameter lambda and is the change of nuclear charges at lamda=1
    step = 0.02
    if order < 1:
        #print(mole.geometry)
        if geom_hash(mole.geometry, Z) in already_compt:
            return already_compt[geom_hash(mole.geometry, Z)]
        else:
            result = mole.get_electronic_energy(Z)
            already_compt.update({geom_hash(mole.geometry, Z):result})
            return result
    else:
        def f(b):
            return dlambda_electronic_energy(mole, [x+b*step*y for x,y in zip(Z,dlambda)], dlambda, order-1)
        #return (-f(4)/280 + 4*f(3)/105 - f(2)/5 + 4*f(1)/5 - 4*f(-1)/5 + f(-2)/5 - 4*f(-3)/105 + f(-4)/280)/step
        return (-f(2)/12 + 2*f(1)/3 - 2*f(-1)/3 + f(-2)/12)/step

def lambda_taylorseries_electronic_energy(mole, Z, dlambda, order):
    """
    dlambda is a list with the desired difference in nuclear charge of the endpoints
    compared to the current state of the molecule (so the difference transmuted for
    lambda = 1
    """
    return dlambda_electronic_energy(mole, Z, dlambda, order)/math.factorial(order)


def geomAE(graph, m=[2,2], dZ=[1,-1], debug = False, with_all_energies = False, with_electronic_energy_difference = False, with_Taylor_expansion = False, take_hydrogen_data_from=''):
    '''Returns the number of alchemical enantiomers of mole that can be reached by
    varying m[i] atoms in mole with identical Coulombic neighborhood by dZ[i].
    In case of method = 'geom', log = 'verbose', the path for the xyz data of the hydrogens is
    needed to fill the valencies.'''
    N_m = len(m)
    N_dZ = len(dZ)
    start_time = time.time()
    mole = np.copy(np.array(graph.geometry, dtype=object))
    N = copy.copy(graph.number_atoms)
    equi_atoms = np.copy(np.array(graph.equi_atoms, dtype=object))

    if N < np.sum(m):
        raise ValueError("Too less atoms in molecule for sum of to be transmuted atoms.")
    if (np.sum(np.multiply(m,dZ)) != 0):
        raise ValueError("Netto change in charge must be 0")
    if 0 in dZ:
        raise ValueError("0 not allowed in array dZ")
    if N_m != N_dZ:
        raise ValueError("Number of changes and number of charge values do not match!")
    if len(equi_atoms) == 0:
        return 0

    '''All of these sites in each set need to be treated simultaneously. Hence, we
    flatten the array. However, we later need to make sure that only each set fulfills
    netto charge conservation. This is why similars is initalized'''
    similars = list(itertools.chain(*equi_atoms))
    '''This is the list of all atoms which can be transmuted simultaneously.
    Now, count all molecules which are possible excluding mirrored or rotated versions'''
    count = 0
    #Initalize empty array temp_mole for all configurations.
    temp_mole = []
    #Get necessary atoms listed in equi_atoms
    for i in range(len(similars)):
        temp_mole.append(mole[similars[i]])
    #Make sure, that m1+m2 does no exceed length of similars
    if np.sum(m) > len(similars):
        return 0

    '''Now: go through all combinations of transmuting m atoms of set similars
    by the values stored in dZ. Then: compare their atomrep_inertia_moments
    and only count the unique ones'''
    atomwise_config = np.zeros((len(similars), N_dZ+1), dtype='int') #N_dZ+1 possible states: 0, dZ1, dZ2, ...
    standard_config = np.zeros((len(similars)))
    #All allowed charges for ONE atom at a time
    for i in range(len(similars)):
        #no change:
        atomwise_config[i][0] = elements[temp_mole[i][0]]
        #Initalize standard_config:
        standard_config[i] = elements[temp_mole[i][0]]
        #just changes:
        for j in range(N_dZ):
            atomwise_config[i][j+1] = elements[temp_mole[i][0]]+dZ[j]
    #All possible combinations of those atoms with meshgrid; the * passes the arrays element-wise
    mole_config_unfiltered = np.array(np.meshgrid(*atomwise_config.tolist(), copy=False)).T.reshape(-1,len(similars))
    #Initalize a molecule configuration:
    mole_config = np.zeros((1,len(similars)),dtype='int')
    mole_config = np.delete(mole_config, 0, axis = 0)

    for k in range(len(mole_config_unfiltered)):
        # m1 sites need to be changed by dZ1, m2 sites need to be changed by dZ2, etc...
        if np.array([(m[v] == (np.subtract(mole_config_unfiltered[k],standard_config) == dZ[v]).sum()) for v in range(N_dZ)]).all():
            '''Check that the netto charge change in every set of equivalent atoms is 0'''
            pos = 0
            for i in range(len(equi_atoms)):
                sum = 0
                #This loop has to start where the last one ended
                for j in range(pos,pos+len(equi_atoms[i])):
                    sum += mole_config_unfiltered[k][j] - standard_config[j]
                pos += len(equi_atoms[i])
                if sum != 0:
                    break
                if (sum == 0) and (i == len(equi_atoms)-1):
                    #print(mole_config_unfiltered[k])
                    mole_config = np.append(mole_config, [mole_config_unfiltered[k]], axis = 0)
    if len(mole_config) == 0:
        return 0
    if np.min(mole_config) < 0:
        #Check that atoms have not been transmuted to negative charges
        raise ValueError("Values in dZ lead to negative nuclear charges in electronically equivalent atoms.")
    #Fourth: All remaining configs, their Coulomb inertia moments and their Delta_Coulomb inertia moments are saved and uniqued
    CIM = np.zeros((len(mole_config), 3))

    Total_CIM = np.zeros((len(mole_config),2), dtype=object) #Entry 0: Config; entry 1: atomrep_inertia_moments
    for i in range(len(mole_config)):
        for j in range(len(similars)):
            temp_mole[j][0] = inv_elements[mole_config[i][j]]
        CIM[i] = atomrep_inertia_moment(temp_mole)
        Total_CIM[i][0] = np.copy(temp_mole)
        Total_CIM[i][1] = np.copy(CIM[i])

    '''Now, all possible combinations are obtained; with the following loops,
    we can get rid of all the spacial enantiomers: Delete all SPACIALLY
    equivalent configurations, i.e. all second, third, etc. occurences of a
    spacial configuration'''

    #Initalize array of already seen CIMs. No better way to do this?
    seen = np.array([[1.,2.,3.]])
    seen = np.delete(seen, 0, axis= 0)
    config_num = 0
    while config_num <len(Total_CIM):
        if not array_compare(Total_CIM[config_num][1], seen):
            seen = np.append(seen, [Total_CIM[config_num][1]], axis = 0)
            config_num += 1
        else:
            Total_CIM = np.delete(Total_CIM, config_num, axis = 0)
    #print(len(Total_CIM))

    '''Alchemical enantiomers are those molecules which do not transmute
    into themselves (or its spatial enantiomer) under the mirroring in charge,
    i.e. if one adds the inverted configuration of transmutations to twice the molecule,
    its CIM has changed.'''

    config_num = 0
    while config_num <len(Total_CIM):
        current_config = np.zeros(len(similars),dtype=object)
        for i in range(len(similars)):
            current_config[i] = elements[Total_CIM[config_num][0][i][0]]
        mirror_config = 2*standard_config - current_config
        #print(current_config)
        #print(mirror_config)
        #print(Total_CIM[config_num][1]) #The CIM of current_config
        for i in range(len(similars)):
            temp_mole[i][0] = inv_elements[mirror_config[i]]
        #print(atomrep_inertia_moment(temp_mole))
        #print('----------')
        if are_close_lists(Total_CIM[config_num][1], atomrep_inertia_moment(temp_mole)):
            Total_CIM = np.delete(Total_CIM, config_num, axis = 0)
        else:
            config_num += 1

    '''All is done. Now, print the remaining configurations and their contribution
    to count.'''
    count += len(Total_CIM)

    if with_electronic_energy_difference and len(Total_CIM) > 0:
        '''Explicitly calculate the energies of all the configurations in Total_CIM[0]
        and their mirrors, then print their energy difference'''
        if take_hydrogen_data_from != '':
            full_mole = parse_XYZtoMAG(take_hydrogen_data_from, with_hydrogen=True)
            for i in range(len(Total_CIM)):
                #print('--------------------------------')
                Z_new = np.zeros((full_mole.number_atoms)).tolist()
                num = 0
                #Initalize the two strings to discriminate the AEs
                AE_string1 = ''
                AE_string2 = ''
                for j in similars:
                    Z_new[j] = elements[Total_CIM[i][0][num][0]]-elements[full_mole.geometry[j][0]]
                    num += 1
                #Fill both strings
                for j in range(full_mole.number_atoms):
                    AE_string1 += inv_elements[elements[full_mole.geometry[j][0]]+Z_new[j]]
                    AE_string2 += inv_elements[elements[full_mole.geometry[j][0]]-Z_new[j]]
                #print(Z)
                diff = full_mole.get_electronic_energy(Z = Z_new) - full_mole.get_electronic_energy(Z = [-x for x in Z_new])
                print("------------------------"+AE_string1+" minus "+AE_string2+"-------------------------")
                print(str(diff)+'\t'+str(full_mole.name))
        else:
            print("'take_hydrogen_data_from' needs an argument")

    if with_Taylor_expansion and len(Total_CIM) > 0:
        max_order = 2
        '''Explicitly calculate the Taylor expansion of all the configurations in Total_CIM[0]
        and their mirrors, then print their energy difference'''
        if take_hydrogen_data_from != '':
            full_mole = parse_XYZtoMAG(take_hydrogen_data_from, with_hydrogen=True)
            for i in range(len(Total_CIM)):
                #print('--------------------------------')
                Z = np.zeros((full_mole.number_atoms)).tolist()
                num = 0
                #Initalize the only one string to discriminate the AEs
                AE_string1 = ''
                AE_string2 = ''
                for j in similars:
                    Z[j] = elements[Total_CIM[i][0][num][0]]-elements[full_mole.geometry[j][0]]
                    num += 1
                #Fill both strings
                for j in range(full_mole.number_atoms):
                    AE_string1 += inv_elements[elements[full_mole.geometry[j][0]]+Z[j]]
                    AE_string2 += inv_elements[elements[full_mole.geometry[j][0]]-Z[j]]
                #print(Z)
                print("------------------------"+AE_string1+"-------------------------")
                sum=0
                print("Actual energy [Ha]")
                print(full_mole.get_electronic_energy(Z = Z))
                for i in range(0,max_order+1):
                    print("Taylor series of energy, order "+str(i)+" :")
                    res = lambda_taylorseries_electronic_energy(full_mole, np.zeros((full_mole.number_atoms)).tolist(), Z, i)
                    print(res)
                    sum += res
                    print("Sum = "+str(sum))
                print("------------------------"+AE_string2+"-------------------------")
                sum=0
                print("Actual energy [Ha]")
                print(full_mole.get_electronic_energy(Z = [-x for x in Z]))
                for i in range(0,max_order+1):
                    print("Taylor series of energy, order "+str(i)+" :")
                    res = lambda_taylorseries_electronic_energy(full_mole, np.zeros((full_mole.number_atoms)).tolist(), [-x for x in Z], i)
                    print(res)
                    sum += res
                    print("Sum = "+str(sum))
        else:
            print("'take_hydrogen_data_from' needs an argument")

    if with_all_energies and len(Total_CIM) > 0:
        '''Explicitly calculate the energies of all the configurations in Total_CIM[0],
        but do so by returning a list of MoleAsGraph objects'''
        if take_hydrogen_data_from != '':
            print('----------------------------')
            full_mole = parse_XYZtoMAG(take_hydrogen_data_from, with_hydrogen=True)
            for i in range(len(Total_CIM)):
                #print('--------------------------------')
                Z = np.zeros((full_mole.number_atoms)).tolist()
                num = 0
                for j in similars:
                    Z[j] = elements[Total_CIM[i][0][num][0]]-elements[full_mole.geometry[j][0]]
                    num += 1
                #print(Z)
                energy_total = full_mole.get_total_energy(Z = Z)
                energy_nuclear = full_mole.get_nuclear_energy(Z = Z)
                print("Total Energy [Ha]: "+str(energy_total)+"\tElectronic Energy [Ha]: "+str(energy_total-energy_nuclear)+"\tNuclear Energy [Ha]: "+str(energy_nuclear))
        else:
            print("'take_hydrogen_data_from' needs an argument")

    if debug == True:
        num_sites = len(similars)
        print('---------------')
        print("Time:", (time.time() - start_time),"s")
        print('---------------')
        for i in range(len(Total_CIM)):
            #This prints the current configuration
            print(Total_CIM[i][0])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #Fill the points into xs,ys,zs
            xs = np.zeros((num_sites))
            ys = np.zeros((num_sites))
            zs = np.zeros((num_sites))
            n = np.zeros((num_sites),dtype=object)
            for j in range(num_sites):
                xs[j] = Total_CIM[i][0][j][1]
                ys[j] = Total_CIM[i][0][j][2]
                zs[j] = Total_CIM[i][0][j][3]
                n[j] = Total_CIM[i][0][j][0]
            ax.scatter(xs, ys, zs, marker='o', facecolor='black')
            #print(Total_CIM[i][0][1][0])
            for j, txt in enumerate(n):
                ax.text(xs[j], ys[j], zs[j], n[j])
            #ax.set_xlabel('X')
            #ax.set_ylabel('Y')
            #ax.set_zlabel('Z')
        plt.show()
    return count

def nautyAE(graph, m = [2,2], dZ=[+1,-1], debug = False, bond_energy_rules = False):
    #graph is an instance of the MoleAsGraph class
    #dZ and m are each arrays that include the transmutations and their number
    start_time = time.time()
    m = np.array(m)
    dZ = np.array(dZ)
    N_m = len(m)
    N_dZ = len(dZ) #number of transmutations = number of colors-1
    max_node_number = np.amax(graph.edge_layout)+1
    N = len(np.unique(graph.edge_layout))
    if len(graph.orbits) == 0:
        #There are no orbits, so there are no AE
        return 0
    if 0 in dZ:
        raise ValueError("0 not allowed in array dZ")
    if N_dZ < 2:
        raisorbitsor("Number of transmutations must be at least 2")
    if N_m != N_dZ:
        raise ValueError("Number of transmutations does not match their number!")
    #Check for overall charge conservation
    if (np.sum(np.multiply(m,dZ)) != 0):
        raise ValueError("Netto change in charge must be 0")
    if N == 1:
        raise ValueError("Graph needs to have at least 2 vertices.")
    if N != max_node_number:
        raise ValueError("Enumerate the vertices with integers without omissions")
    if N_dZ != len(np.unique(dZ)):
        raise ValueError("Equal values in multiple entries")
    if N < np.sum(m):
        raise ValueError("Too less vertices in graph for total number of to be transmuted atoms.")

    #Use graph-based algorithm nauty27r1; build the string command to be passed to the bash
    #Command for the standard case looks like:
    #echo 'n=10;0:1;1:2;2:3;3:4;4:5;5:0;0:6;6:7;7:8;8:9;9:5;' | /home/simon/Desktop/nauty27r1/dretog -q | /home/simon/Desktop/nauty27r1/vcolg -q -T -m3 |
    #awk '{count1 = 0; count2 = 0; for (i=3; i<13; i++){if ($i == 1) count1++; else if ($i == 2) count2++;} if ((count1 == 2) && (count2 == 2)) print}'
    #This immediatly checks wether charge conservation and the correct number of colors are given.
    command = "echo 'n=" + str(N) + ";"
    for i in range(len(graph.edge_layout)):
        command += str(graph.edge_layout[i][0]) +":" + str(graph.edge_layout[i][1]) + ";"
    command += "' | "+PathToNauty27r1+"dretog -q | "+PathToNauty27r1+"vcolg -q -T -m"
    command += str(N_dZ+1)
    command += " | awk '{"
    for i in range(N_dZ):
        command += "count"+str(i+1)+"=0; "
    command += "for (i=3; i<" + str(N+3) + "; i++){if ($i == 1) count1++; else if ($i == 2) count2++;"
    for color in range(2,N_dZ):
        command += " else if ($i == " + str(color+1) + ") count" + str(color+1) + "++;"
    command += "} if ((count1 == " + str(m[0]) + ") && (count2 == " + str(m[1]) + ")"
    for color in range(2,N_dZ):
        command += " && (count" + str(color+1) + " == " + str(m[color]) + ")"
    command += ") print}'"
    output = os.popen(command).read()
    #print(command,"\n")
    #Color 0 is the standard, colors 1,2,3,etc. are the transmutations
    #Parse output to an array
    num_lines = output.count('\n')
    graph_config = np.zeros((num_lines),dtype=object)
    for i in range(num_lines):
        line = output.splitlines(False)[i]
        #Get rid of everything after '  ':
        line = line.split('  ')[0]
        #Parse numbers to integer array:
        numbers = [int(j) for j in line.split(' ')]
        #Delete first two elements
        numbers = np.delete(numbers, (0,1), axis = 0)
        graph_config[i] = numbers

    '''The parsed array needs to fulfill two things:
    1) Is the netto charge within all orbits conserved? An orbit is the set of all
    equivalent atoms in the geometry-based method.
    2) Is the alchemically mirrored graph not itself?
    This is the same as asking if graph and mirror are isomorphic which can only happen
    if and only if for each pair of m[i],dZ[i] there exists a pair m[j],-dZ[j]
    that is equal.'''

    #print(graph_config)
    #Answering question one:
    config_num = 0
    while config_num < len(graph_config):
        for i in range(len(graph.orbits)):
            sum = 0
            for j in graph.orbits[i]:
                #Avoid getting the last element
                if graph_config[config_num][j] != 0:
                    sum += dZ[graph_config[config_num][j]-1]
            if sum != 0:
                graph_config = np.delete(graph_config, config_num, axis = 0)
                break
            if i == len(graph.orbits)-1:
                config_num += 1
    #print(graph_config)
    #Prepare some dicts
    color2dZ = {0:0}
    for i in range(N_dZ):
        color2dZ[i+1] = dZ[i]
    dZ2color = {v: k for k, v in color2dZ.items()}
    #Answering the second question:
    '''Find out if all those graphs are able to self mirror'''
    self_mirrorable = np.array([-i in dZ for i in dZ]).all()
    if self_mirrorable:
        '''Use igraph's isomorphic-function to delete graphs which are isomorphic
        after transmutation'''
        #Prepare the graph
        g1 = igraph.Graph([tuple(v) for v in graph.edge_layout])
        config_num = 0
        while config_num < len(graph_config):
            if g1.isomorphic_vf2(color1=graph_config[config_num], color2=[dZ2color[-color2dZ[graph_config[config_num][i]]] for i in range(N)]):
                graph_config = np.delete(graph_config, config_num, axis = 0)
            else:
                config_num += 1
    count = len(graph_config)


    #-----------------------------Very optional, very prelimenary-------------------------
    #Find all the rules for bond energies between the AEs in one pair:
    if bond_energy_rules == True:
        print('----------------------------')
        #Initalize vector to hold all coefficients of all possible bonds
        #Find all possible nuclear charges:
        chem_elem = np.copy(graph.elements_at_index)
        transmutations = dZ
        transmutations = np.append(transmutations,0)
        available_charges = np.unique([elements[chem_elem[i]] for i in range(len(chem_elem))])
        possible_charges = np.unique([[i+j for i in available_charges] for j in transmutations])
        #Sort the vector
        possible_charges = np.sort(possible_charges)
        #Create the vector of all possible combinations
        possible_bonds = []
        for i in range(len(possible_charges)):
            for j in range(i, len(possible_charges)):
                possible_bonds = np.append(possible_bonds, inv_elements[possible_charges[i]]+inv_elements[possible_charges[j]])
        #print(possible_bonds)
        ref_bonds = bond_count(graph.edge_layout, graph.elements_at_index)
        multip = np.zeros((len(possible_bonds)), dtype='int')
        for j in range(len(ref_bonds)):
            for k in range(len(possible_bonds)):
                if ref_bonds[j] == possible_bonds[k]:
                    multip[k] += 1

        rule = []
        for i in range(count):
            #We need the bonds of both molecules of the pair of AE!!!
            bonds1 = bond_count(graph.edge_layout, [inv_elements[elements[graph.elements_at_index[j]]+color2dZ[graph_config[i][j]]] for j in range(N)])
            bonds2 = bond_count(graph.edge_layout, [inv_elements[elements[graph.elements_at_index[j]]-color2dZ[graph_config[i][j]]] for j in range(N)])
            #Count the multiplicities of each bond and store them in a possible_bonds-like vector:
            multip1 = np.zeros((len(possible_bonds)), dtype='int')
            multip2 = np.zeros((len(possible_bonds)), dtype='int')
            for j in range(len(bonds1)):
                for k in range(len(possible_bonds)):
                    if bonds1[j] == possible_bonds[k]:
                        multip1[k] += 1
                    if bonds2[j] == possible_bonds[k]:
                        multip2[k] += 1
            #Prelimenary printout:
            out =''
            sign = 1
            first_nonzero = False
            for i in range(len(multip1)):
                if multip1[i]-multip2[i] != 0:
                    if first_nonzero == False:
                        first_nonzero = True
                        if multip1[i]-multip2[i] < 0:
                            sign = -1
                    out += '  '+str(sign*(multip1[i]-multip2[i]))+' E_'+str(possible_bonds[i])
            if len(out.strip()) != 0:
                rule = np.append(rule, out+' = 0')
        final_set = np.unique(rule)
        print('Bond energy rules:')
        for r in final_set:
            print(r)

    if debug == True:
        print('---------------')
        print("Time:", (time.time() - start_time),"s")
        print('---------------')
        print('Alchemical Enantiomers:')
        for i in range(count):
            x = [inv_elements[elements[graph.elements_at_index[j]]+color2dZ[graph_config[i][j]]] for j in range(N)]
            print(str(sum_formula(x)) + ": " + str(x))
    return count

def uncolor(graph):
    '''Find the most symmetric reference molecule by setting all atoms to the
    rounded average integer charge.'''
    tot_nuc_charge = 0
    for i in range(graph.number_atoms):
        tot_nuc_charge += elements[graph.elements_at_index[i]]
    average_element = inv_elements[int(tot_nuc_charge/graph.number_atoms)]
    new_elements = np.empty((graph.number_atoms), dtype='str')
    new_geometry = graph.geometry
    #Explicitly set the average element:
    #average_element = 'C'
    for i in range(graph.number_atoms):
        new_elements[i] = average_element
        new_geometry[i][0] = average_element
    return MoleAsGraph('isoatomic'+graph.name, graph.edge_layout,new_elements,new_geometry)

def Find_theoAEfromgraph(N, dZ_max):
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
    print('Number of atoms: '+str(N)+'\tdZ_max: '+str(dZ_max)+'\tPossibles / % : '+str(count*batching*100/(num_lines)))

def Find_AEfromref(graph, dZ_max = 3, log = 'normal', method = 'geom', take_hydrogen_data_from = '', with_all_energies = False, with_bond_energy_rules = False, with_electronic_energy_difference = False, with_Taylor_expansion = False):
    """
    graph = MoleAsGraph
    dZ_max = 1,2,3
    log = 'normal', 'sparse', 'quiet'
    energy_log-options:
        with_all_energies (geom only)
        with_bond_energy_rules (graph only)
        with_electronic_energy_difference (geom only)
        with_Taylor_expansion (geom only)
    """
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
        m_all = [[]]
        dZ_all = [[]]
    #For plotting: save number of transmuted atoms num_trans and time
    #print(m_all)
    #print(dZ_all)
    num_trans = []
    times = []
    total_number = 0
    if log == 'normal':
        print('\n'+ graph.name + '; method = ' + method + '\n----------------------------')

    for i in range(len(m_all)):
        random_config = np.zeros((graph.number_atoms))
        pos = 0
        for j in range(len(m_all[i])):
            for k in range(m_all[i][j]):
                random_config[pos] = dZ_all[i][j]
                pos += 1
        chem_form = str(sum_formula([inv_elements[elements[graph.elements_at_index[v]]+random_config[v]] for v in range(graph.number_atoms)]))
        if log == 'normal' or log == 'verbose':
            print(chem_form)
        m_time = time.time()
        if method == 'graph':
            x = nautyAE(graph, m_all[i], dZ_all[i], debug= False, bond_energy_rules = with_bond_energy_rules)
        if method == 'geom':
            x = geomAE(graph, m_all[i], dZ_all[i], debug= False, with_all_energies = with_all_energies, with_electronic_energy_difference = with_electronic_energy_difference, with_Taylor_expansion = with_Taylor_expansion, take_hydrogen_data_from= take_hydrogen_data_from)
        if log == 'normal' or log == 'verbose':
            print('Time:', time.time()-m_time)
            print('Number of AEs:', x,'\n')
        num_trans.append(np.sum(m_all[i]))
        times.append(time.time()-m_time)
        total_number += x
    if log == 'normal':
        print('----------------------------')
        print(graph.name)
        print('Total time:', time.time()-start_time)
        print('Total number of AEs:', total_number)
        print('Number of transmuted atoms:', list(num_trans))
        print('Time:', list(times))
        print('----------------------------')
    if log == 'sparse':
        print(graph.name + '\t' + str(time.time()-start_time) + '\t' + str(graph.number_atoms) + '\t' + str(total_number))
    if log == 'quiet':
        return total_number

def Find_reffromtar(graph, dZ_max = 3, method = 'graph', log = 'normal'):
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
        CN = atomrep(Geom)
        indices = np.argsort(CN).tolist()
        indices2 = np.copy(indices).tolist()
        lst = []
        sites = []
        for i in range(len(indices)-1):
            if i not in indices2:
                continue
            lst.append(indices[i])
            indices2.remove(i)
            for j in range(i+1,len(indices)):
                if are_close_scalars(CN[indices[i]],CN[indices[j]],looseness):
                    lst.append(indices[j])
                    indices2.remove(j)
            sites.append(lst)
            lst = []
        #Delete all similars which include only one atom:
        num_similars = 0
        while num_similars < len(sites):
            if len(sites[num_similars])>1:
                num_similars += 1
            else:
                del sites[num_similars]
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
        if are_close_lists(atomrep_inertia_moment(target_config), atomrep_inertia_moment(mirror_config)) or (netto_charge != 0):
            #If yes, wipe chem_config such that the original molecule is returned
            chem_config = np.array([graph.geometry[i][0] for i in range(graph.number_atoms)], copy=True)
    #Return a MoleAsGraph object
    for i in range(len(chem_config)):
        Geom[i][0] = chem_config[i]
    if log == 'normal':
        print('Name: reffrom'+str(graph.name))
        print('Edge layout: '+str(graph.edge_layout))
        print('Elements at index: '+str(chem_config))
        print('GEOMETRY:\n' +str(Geom))
    return MoleAsGraph('reffrom'+graph.name, graph.edge_layout,chem_config.tolist(), Geom.tolist())
