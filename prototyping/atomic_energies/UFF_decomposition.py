import numpy as np

def get_UFF_decomposition(ob_output):
    """
    wrapper returns dictionary of decomposed energies
    """
    with open(ob_output) as f:
        file = f.readlines()
    
    elements = parse_element(file)
    positions = parse_positions(file)
    e_bond = parse_e_bond(file)
    e_angle = parse_e_angle(file)
    e_torsional = parse_e_torsional(file)
    e_vdw = parse_e_vdw(file)
    UFF_decomposition = set_up_UFF_dict(elements, positions, e_bond, e_angle, e_torsional, e_vdw)
    return(UFF_decomposition)

def set_up_UFF_dict(elements, positions, e_bond, e_angle, e_torsion, e_vdw):
    """
    change format to dict of numpy arrays
    calculate atomic energies from contributions
    """
    atom_id = []
    for k in e_bond.keys():
        atom_id.append(k)
        
        positions_array = np.array(list(positions.values()))
        e_bond_array = np.array(list(e_bond.values()))
        e_angle_array = np.array(list(e_angle.values()))
        e_torsion_array = np.array(list(e_torsion.values()))
        e_vdw_array = np.array(list(e_vdw.values()))
        
    e_atomic_array = e_bond_array+e_angle_array+e_torsion_array+e_vdw_array
    
    atomic_energies_UFF = {'atom_id_babel':atom_id, 'element':elements, 'position':positions_array, 'e_bond':e_bond_array, 'e_angle':e_angle_array, 'e_torsion':e_torsion_array, 'e_vdw':e_vdw_array, 'e_atomic':e_atomic_array}
    return(atomic_energies_UFF)
    
def sort_dict(dict_unsorted):
    # sort by numerical key
    keys_sorted = list(dict_unsorted.keys())
    keys_sorted.sort()
    dict_sorted = dict()
    for ks in keys_sorted:
        dict_sorted[ks] = dict_unsorted[ks]
    return(dict_sorted)

def parse_element(file):
    """
    get elements from output file
    """
    # find section for atom types
    linenumber = 'Nan'
    for ln, line in enumerate(file):
        if line == 'A T O M   T Y P E S\n':
            linenumber = ln
            break

    # parse section for elements
    parse = file[linenumber+3:]
    i = 0

    elements = []
    while parse[i] != '\n':
        elements.append(parse[i].split()[1].split('_')[0])

        i += 1
    return(elements)

def parse_positions(file):
    """
    get positions from output file
    """
    # find section for bond stretching because here I also write positions
    linenumber = 'Nan'
    for ln, line in enumerate(file):
        if line == 'B O N D   S T R E T C H I N G\n':
            linenumber = ln
            break
    # parse section for positions
    parse = file[linenumber+5:]
    i = 0

    position_dict = dict()
    while '     TOTAL BOND STRETCHING ENERGY' not in parse[i]:
        # get index and position of atom i
        idx_i = int(parse[i].strip('\n').split()[-8])
        position_i = [float(pos) for pos in parse[i].strip('\n').split()[-6:-3]]
        # sanity check that same index gives alsways same position
        if idx_i in position_dict.keys():
            assert position_dict[idx_i] == position_i, "index assigned to different positions"
        else:
            position_dict[idx_i] = position_i

        # get index and position of atom j
        idx_j = int(parse[i].strip('\n').split()[-7])
        position_j = [float(pos) for pos in parse[i].strip('\n').split()[-3:]]
        # sanity check that same index gives alsways same position
        if idx_j in position_dict.keys():
            assert position_dict[idx_j] == position_j, "index assigned to different positions"
        else:
            position_dict[idx_j] = position_j

        i += 1
    
    # sort by babel atom indices
    position_dict_sorted = sort_dict(position_dict)
    return(position_dict_sorted)


def parse_e_bond(file):
    """
    get bond energies from output file
    """
    # find section for bond stretching because here I also write positions
    linenumber = 'Nan'
    for ln, line in enumerate(file):
        if line == 'B O N D   S T R E T C H I N G\n':
            linenumber = ln
            break

    # parse section for positions
    parse = file[linenumber+5:]
    i = 0

    prop_dict = dict()
    while 'TOTAL BOND STRETCHING ENERGY' not in parse[i]:
        # get index of atom i
        idx_i = int(parse[i].strip('\n').split()[-8])
        # get index of atom j
        idx_j = int(parse[i].strip('\n').split()[-7])
        # get bond energy
        e_bond = float(parse[i].strip('\n').split()[7])
        
        # add half energy if index already exists otherwise initialize key and assign energy
        if idx_i in prop_dict.keys():
            prop_dict[idx_i] += e_bond/2
        else:
            prop_dict[idx_i] = e_bond/2
            
        if idx_j in prop_dict.keys():
            prop_dict[idx_j] += e_bond/2
        else:
            prop_dict[idx_j] = e_bond/2

        i += 1
    return(sort_dict(prop_dict))


def parse_e_angle(file):
    """
    get angle energies from output file
    """
    # find section for bond stretching because here I also write positions
    linenumber = 'Nan'
    for ln, line in enumerate(file):
        if line == 'A N G L E   B E N D I N G\n':
            linenumber = ln
            break

    # parse section for positions
    parse = file[linenumber+5:]
    i = 0

    prop_dict = dict()
    while 'TOTAL ANGLE BENDING ENERGY' not in parse[i]:
        # get atom indices
        idx_i = int(parse[i].strip('\n').split()[-3])
        idx_j = int(parse[i].strip('\n').split()[-2])
        idx_k = int(parse[i].strip('\n').split()[-1])
        
        # get angle energy
        e_angle = float(parse[i].strip('\n').split()[-4])
        
        # add 1/3 of energy if index already exists otherwise initialize key and assign energy
        if idx_i in prop_dict.keys():
            prop_dict[idx_i] += e_angle/3
        else:
            prop_dict[idx_i] = e_angle/3
            
        if idx_j in prop_dict.keys():
            prop_dict[idx_j] += e_angle/3
        else:
            prop_dict[idx_j] = e_angle/3
            
        if idx_k in prop_dict.keys():
            prop_dict[idx_k] += e_angle/3
        else:
            prop_dict[idx_k] = e_angle/3

        i += 1
    return(sort_dict(prop_dict))

def parse_e_torsional(file):
    """
    get torsional energies from output file
    """
    # find section for bond stretching because here I also write positions
    linenumber = 'Nan'
    for ln, line in enumerate(file):
        if line == 'T O R S I O N A L\n':
            linenumber = ln
            break

    # parse section for positions
    parse = file[linenumber+5:]
    i = 0

    prop_dict = dict()
    while 'TOTAL TORSIONAL ENERGY' not in parse[i]:
        # get atom indices
        idx_i = int(parse[i].strip('\n').split()[-4])
        idx_j = int(parse[i].strip('\n').split()[-3])
        idx_k = int(parse[i].strip('\n').split()[-2])
        idx_l = int(parse[i].strip('\n').split()[-1])
        
        # get angle energy
        e_torsional = float(parse[i].strip('\n').split()[-5])/4
        
        # add 1/4 of energy if index already exists otherwise initialize key and assign energy
        if idx_i in prop_dict.keys():
            prop_dict[idx_i] += e_torsional
        else:
            prop_dict[idx_i] = e_torsional
            
        if idx_j in prop_dict.keys():
            prop_dict[idx_j] += e_torsional
        else:
            prop_dict[idx_j] = e_torsional
            
        if idx_k in prop_dict.keys():
            prop_dict[idx_k] += e_torsional
        else:
            prop_dict[idx_k] = e_torsional
            
        if idx_l in prop_dict.keys():
            prop_dict[idx_l] += e_torsional
        else:
            prop_dict[idx_l] = e_torsional

        i += 1
    return(sort_dict(prop_dict))

def parse_e_vdw(file):
    """
    get bond energies from output file
    """
    # find section for bond stretching because here I also write positions
    linenumber = 'Nan'
    for ln, line in enumerate(file):
        if line == 'V A N   D E R   W A A L S\n':
            linenumber = ln
            break

    # parse section for positions
    parse = file[linenumber+5:]
    i = 0

    prop_dict = dict()
    while 'TOTAL VAN DER WAALS ENERGY' not in parse[i]:
        # get index of atom i
        idx_i = int(parse[i].strip('\n').split()[-2])
        # get index of atom j
        idx_j = int(parse[i].strip('\n').split()[-1])
        # get vdw energy
        e_vdw = float(parse[i].strip('\n').split()[-3])/2
        
        # add half energy if index already exists otherwise initialize key and assign energy
        if idx_i in prop_dict.keys():
            prop_dict[idx_i] += e_vdw
        else:
            prop_dict[idx_i] = e_vdw
            
        if idx_j in prop_dict.keys():
            prop_dict[idx_j] += e_vdw
        else:
            prop_dict[idx_j] = e_vdw

        i += 1
    return(sort_dict(prop_dict))