import sys
sys.path.insert(0, '/home/misa/APDFT/prototyping/atomic_energies/')
import qml_interface as qmi

def split_elementwise(data_group, molecule_sizes_group, prop_name):
    """
    split property specified in propname elementwise for data defined in data_group
    data_group: list, the elements of the lists are also lists that contain the atoms and their atomic properties for a molecule
    molecule_sizes_group is the length of the molecules in data_group
    """
    charges_group = qmi.generate_label_vector(data_group, molecule_sizes_group.sum(), value='charge')
    prop = qmi.generate_label_vector(data_group, molecule_sizes_group.sum(), value=prop_name)
    idc_by_charge = qmi.partition_idx_by_charge(charges_group)
    prop_dict = {k: [] for k in idc_by_charge.keys()}
    for k in prop_dict.keys():
        prop_dict[k] = prop[idc_by_charge[k]]
    return(prop_dict)

def get_charge_neighbours(mol):
    """
    returns the nuclear charge of the bonding partners for all atoms in mol
    if mol has hydrogens added, also their neighbours will be considered
    """
    charge_neighbours = []
    chN = []
    chH = []
    for atom in mol.GetAtoms():
        degree = atom.GetTotalDegree() # number of binding partners
        neighbors = atom.GetNeighbors()
        charges_neighbors = 0
        for n in neighbors:
            charges_neighbors += n.GetAtomicNum()
        chN.append(charges_neighbors)
        charge_H = degree - len(neighbors) # number of hydrogens bonded that are not explicit; is equal to charge of implicit hydrogens
        chH.append(charge_H)
        charge = (charges_neighbors + charge_H)
        charge_neighbours.append(charge)
    return(charge_neighbours)