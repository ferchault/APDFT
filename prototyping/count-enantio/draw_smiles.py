from pysmiles import read_smiles
import networkx as nx
import matplotlib.pyplot as plt
from config import *
import sys
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

#The tag of the QM9 molecule
#tag = '022079'

def draw(tag, out='bash'):
    #Get the SMILES from the correct file
    f = open(PathToQM9XYZ + 'dsgdb9nsd_'+ tag +'.xyz', 'r')
    data = f.read()
    f.close()
    N = int(data.splitlines(False)[0]) #number of atoms including hydrogen
    #Get SMILES:
    smiles = data.splitlines(False)[N+3].split('\t')[0]
    if out == 'bash':
        mol = read_smiles(smiles)
        elements = nx.get_node_attributes(mol, name = "element")
        print(smiles)
        nx.draw(mol, with_labels=True, labels = elements, pos=nx.spring_layout(mol))
        plt.gca().set_aspect('equal')
        plt.show()
    if out == 'file':
        #img=Draw.MolsToGridImage(Chem.MolFromSmiles(smiles),molsPerRow=4,subImgSize=(200,200),legends=smiles)
        #img.save('chem_fig')
        mol = Chem.MolFromSmiles(smiles)
        #Draw.MolToFile(mol, 'figures/structures/'+tag+'.png')
        Draw.MolToFile(mol, 'figures/outliers/'+tag+'.png')

if __name__ == "__main__":
    draw(sys.argv[2], sys.argv[1])
