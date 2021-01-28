import numpy as np

def parse_out_file(file, value_name):
    # change alias to correct name in file
    if value_name == 'ekin':
        value_name = 'Particle   Kinetic   Energy'
    elif value_name == 'epot':
        value_name = 'Particle   Potential Energy'
    elif value_name == 'lattice_vectors':
        value_name = 'LATTICE INFORMATION'
    elif value_name == 'temperature':
        value_name = 'Temperature'
    elif value_name == 'nve temperature':
        value_name = 'NVE Temperature'
    elif value_name == 'pressure':
        value_name = 'Ext Pressure * Vol'
        
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
                elif value_name == 'Final total energy':
                    value = float(line.split()[3])
                elif value_name == 'Particle   Kinetic   Energy':
                    value.append(float(line.split(':')[-1]))
                elif value_name == 'Particle   Potential Energy':
                    value.append(float(line.split(':')[-1]))
                elif value_name == 'Temperature':
                    value.append(float(line.split(':')[-1]))
                elif value_name == 'NVE Temperature':
                    value.append(float(line.split(' ')[-2]))    
                elif value_name == 'Ext Pressure * Vol':
                    value.append(float(line.split(':')[-1]))
                elif value_name == 'LATTICE INFORMATION':
                    lvectors = []
                    f.readline()
                    for i in range(3):
                        a = f.readline()
                        vector_components = a.split()[2:]
                        lvectors.append(([float(v) for v in vector_components]))
                    lvectors = np.array(lvectors)
                    value.append(lvectors)
    return(value)

def parse_out_file_MD(file, value_name):
    value = []
    with open(file, 'r') as f:
        for line in f:
            if not 'ENERGY REPORT' in line:
                pass
            else:
                if value_name == 'temperature':
                    f.readline()
                    line = f.readline()
                    value.append(float(line.split(':')[-1]))
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

def parse_ion_file_complete(file):
    lattice = []
    positions = []
    atom_types = []
    with open(file, 'r') as f:
        for line in f:
            if '%BLOCK LATTICE' in line:
                lattice = parse_lattice_section(f)
            elif '%BLOCK POSITIONS' in line:
                positions, elements = parse_position_section_complete(f)
            
    return(lattice, positions, elements)

def parse_lattice_section(fs):
    lattice = []
    for line in fs:
        if not 'END BLOCK LATTICE' in line:
            coords = line.split()
            coords = [float(c) for c in coords]
            lattice.append(coords)
        else:
            break
    return(lattice)

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

def parse_position_section_complete(fs):
    pos = []
    elements = ''
    for line in fs:
        if not 'END BLOCK POSITIONS' in line:
            coords = line.split()[1:4]
            coords = [float(c) for c in coords]
            pos.append(coords)
            elements += line.split()[0]
        else:
            break
    return(pos, elements)

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

def parse_force_file(file, file_format = '.force.out'):
    forces = []
    with open(file, 'r') as f:
        for line in f:
            if not 'TOTAL-FORCE' in line:
                pass
            else:
                f.readline()
                f.readline()
                forces = parse_force_section(f, file_format)
                break
    return(forces)

def parse_force_section(f, file_format):
    forces = []
    for line in f:
        if not '........' in line:
            if file_format == '.force.out':
                forces_atom = line.split()[2:5]
            elif file_format == '.out':
                forces_atom = line.split()[3:6]
            forces_atom = [float(for_a) for for_a in forces_atom]
            forces.append(forces_atom)
        else:
            break
    return(forces)

def generate_cell_section(cell_par):
    """
    cell_par: array of lattice vecors [ax, ay, az], np.array([[x,y,z], [x,y,z,], [x,y,z]])
    """
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
    """
    cell_par: array of lattice vecors [ax, ay, az], np.array([[x,y,z], [x,y,z,], [x,y,z]])
    atom_types: list of element for every atom Len(atom_types)  = number of atoms
    positions: list of positions
    pos_type: 'CART' or 'FRAC'
    pp: pseudo potential file names (one for every element)
    """
    cell_section = generate_cell_section(cell_par)
    position_section = generate_position_section(atom_types, positions, pos_type)
    elements = set(atom_types)
    pp_section= generate_pp_section(elements, pp)
    ion_file = cell_section + position_section + pp_section
    return(ion_file)

def write_file(fname, fcontent):
    with open(fname, 'w') as f:
        f.writelines(fcontent)