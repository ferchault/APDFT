from pysmiles import read_smiles
import networkx as nx
import matplotlib.pyplot as plt
from config import *

#The tag of the QM9 molecule
tag = '022079'


#Get the SMILES from the correct file
f = open(PathToQM9XYZ + 'dsgdb9nsd_'+ tag +'.xyz', 'r')
data = f.read()
f.close()
N = int(data.splitlines(False)[0]) #number of atoms including hydrogen
#Get SMILES:
mol = read_smiles(data.splitlines(False)[N+3].split('\t')[0])
elements = nx.get_node_attributes(mol, name = "element")
nx.draw(mol, with_labels=True, labels = elements, pos=nx.spring_layout(mol))
plt.gca().set_aspect('equal')
plt.show()
