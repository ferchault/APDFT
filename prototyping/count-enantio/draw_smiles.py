#USE IN BASH: conda activate my-rdkit-env

from pysmiles import read_smiles
import networkx as nx
import matplotlib.pyplot as plt
from config import *
import sys
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

def draw(tag, out='tmp'):
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
        Draw.MolToFile(mol, 'figures/structures/'+tag+'.png')
    if out == 'tmp':
        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(mol, 'figures/tmp_structures/'+tag+'.png')

if __name__ == "__main__":
    if len(sys.argv) == 3:
        draw(sys.argv[1], out=sys.argv[2])
    if len(sys.argv) == 2:
        draw(sys.argv[1])

    #for i in range(4,133885+1):
    #    tag = '000000'[:(6-len(str(i)))] + str(i)
    #    draw(tag, 'file')

'''Original data:
Outstanding candidates (QM9_log_dZ2):
['dsgdb9nsd_017966', '0.3742401599884033', '8', '408\n']
['dsgdb9nsd_017972', '0.3638904094696045', '8', '408\n']
['dsgdb9nsd_017954', '0.2801809310913086', '8', '408\n']
['dsgdb9nsd_017955', '0.24740839004516602', '8', '408\n']
['dsgdb9nsd_017956', '0.24203753471374512', '8', '408\n']
['dsgdb9nsd_017957', '0.22999238967895508', '8', '408\n']
['dsgdb9nsd_017960', '0.2216951847076416', '8', '408\n']
['dsgdb9nsd_017963', '0.23308849334716797', '8', '408\n']

['dsgdb9nsd_037318', '1.8507859706878662', '9', '711\n']
['dsgdb9nsd_037327', '1.9686915874481201', '9', '711\n']
['dsgdb9nsd_037330', '1.704589605331421', '9', '711\n']

['dsgdb9nsd_037858', '1.345862865447998', '9', '362\n']
['dsgdb9nsd_037861', '1.3442637920379639', '9', '362\n']
['dsgdb9nsd_037864', '1.4254262447357178', '9', '362\n']

['dsgdb9nsd_039605', '0.4264981746673584', '9', '1044\n']
['dsgdb9nsd_039606', '0.45586633682250977', '9', '1044\n']
['dsgdb9nsd_039607', '0.4344182014465332', '9', '1044\n']
['dsgdb9nsd_039608', '0.4561190605163574', '9', '1044\n']
['dsgdb9nsd_039613', '0.42791080474853516', '9', '1044\n']
['dsgdb9nsd_039614', '0.42717909812927246', '9', '1044\n']
['dsgdb9nsd_039616', '0.43955516815185547', '9', '1044\n']

['dsgdb9nsd_042066', '0.9349937438964844', '9', '1044\n']
['dsgdb9nsd_042068', '0.8418529033660889', '9', '1044\n']
['dsgdb9nsd_042129', '0.8613934516906738', '9', '1044\n']
['dsgdb9nsd_042075', '0.8984086513519287', '9', '1044\n']
['dsgdb9nsd_042139', '0.9172379970550537', '9', '1044\n']
['dsgdb9nsd_042107', '0.9382381439208984', '9', '1044\n']
['dsgdb9nsd_042112', '1.0779104232788086', '9', '1044\n']
['dsgdb9nsd_042116', '0.9662747383117676', '9', '1044\n']
['dsgdb9nsd_042121', '1.0150351524353027', '9', '1044\n']
['dsgdb9nsd_042054', '0.918269157409668', '9', '1044\n']
['dsgdb9nsd_042056', '0.9463493824005127', '9', '1044\n']
['dsgdb9nsd_042058', '0.9986448287963867', '9', '1044\n']
['dsgdb9nsd_042060', '0.8922762870788574', '9', '1044\n']
['dsgdb9nsd_042062', '0.8955185413360596', '9', '1044\n']
'''
