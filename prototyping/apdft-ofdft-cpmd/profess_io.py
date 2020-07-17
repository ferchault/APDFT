def parse_out_file(file, value_name):
    value = []
    with open(file, 'r') as f:
        for line in f:
            if not value_name in line:
                pass
            else:
                if value_name == 'TOTAL ENERGY':
                    value = float(line.split()[-3])
                    return(value)
                elif value_name == 'NVE Kinetic energy' or value_name == 'NVE Potential energy':
                    value.append(float(line.split()[-2]))
    return(value)

def parse_ion_file(file):
    positions = []
    with open(file, 'r') as f:
        for line in f:
            if not 'BLOCK POSITIONS' in line:
                pass
            else:
                positions = parse_position_section(f)
                break
    return(positions)
                
def parse_position_section(fs):
    pos = []
    for line in fs:
        if not 'END BLOCK POSITIONS' in line:
            coords = line.split()[1:4]
            coords = [float(c) for c in coords]
            pos.append(coords)
        else:
            break
    return(pos)

def parse_velocity_file(file):
    velocities = None
    with open(file, 'r') as f:
        for line in f:
            if not 'ION VELOCITIES (a.u.)' in line:
                pass
            else:
                velocities = parse_velocity_section(f)
                break
    return(velocities)
                
def parse_velocity_section(fs):
    velocities = []
    for line in fs:
        velocities_atom = line.split()
        velocities_atom = [float(v) for v in velocities_atom]
        velocities.append(velocities_atom)
    return(velocities)

def parse_force_file(file):
    forces = []
    with open(file, 'r') as f:
        for line in f:
            if not 'TOTAL-FORCE' in line:
                pass
            else:
                f.readline()
                f.readline()
                forces = parse_force_section(f)
                break
    return(forces)

def parse_force_section(f):
    forces = []
    for line in f:
        if not '........' in line:
            forces_atom = line.split()[2:5]
            forces_atom = [float(for_a) for for_a in forces_atom]
            forces.append(forces_atom)
        else:
            break
    return(forces)

def generate_cell_section(cell_par):
    cell_section = []
    cell_section.append('%BLOCK LATTICE_CART\n')
    for cell_vec in cell_par:
        cell_vec_string = '\t' + '\t'.join(map(str, cell_vec))
        cell_section.append(cell_vec_string+'\n')
    cell_section.append('%END BLOCK LATTICE_CART\n')
    return(cell_section)

def generate_position_section(atom_types, positions, pos_type):
    positions_section = []
    positions_section.append(f'%BLOCK POSITIONS_{pos_type}\n')
    
    for at, pos in zip(atom_types, positions):
        atom_string = f'\t{at}\t' + '\t'.join(map(str, pos))
        positions_section.append(atom_string+'\n')
    positions_section.append(f'%END BLOCK POSITIONS_{pos_type}\n')
    return(positions_section)

def generate_pp_section(elements, pp_names):
    pp_section = []
    pp_section.append('%BLOCK SPECIES_POT\n')
    for el, pp in zip(elements, pp_names):
        pp_section.append(f'\t{el}\t{pp}\n')
    pp_section.append('%END BLOCK SPECIES_POT\n')
    return(pp_section)
    
def generate_ion_file(cell_par, atom_types, positions, pos_type, pp):
    cell_section = generate_cell_section(cell_par)
    position_section = generate_position_section(atom_types, positions, pos_type)
    elements = set(atom_types)
    pp_section= generate_pp_section(elements, pp)
    ion_file = cell_section + position_section + pp_section
    return(ion_file)

def write_file(fname, fcontent):
    with open(fname, 'w') as f:
        f.writelines(fcontent)