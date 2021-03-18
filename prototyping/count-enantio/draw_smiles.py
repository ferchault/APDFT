from pysmiles import read_smiles
import networkx as nx
import matplotlib.pyplot as plt
from config import *
import sys

#The tag of the QM9 molecule
#tag = '022079'

def draw(tag):
    #Get the SMILES from the correct file
    f = open(PathToQM9XYZ + 'dsgdb9nsd_'+ tag +'.xyz', 'r')
    data = f.read()
    f.close()
    N = int(data.splitlines(False)[0]) #number of atoms including hydrogen
    #Get SMILES:
    smiles = data.splitlines(False)[N+3].split('\t')[0]
    mol = read_smiles(smiles)
    elements = nx.get_node_attributes(mol, name = "element")
    print(smiles)
    nx.draw(mol, with_labels=True, labels = elements, pos=nx.spring_layout(mol))
    plt.gca().set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    draw(sys.argv[1])
