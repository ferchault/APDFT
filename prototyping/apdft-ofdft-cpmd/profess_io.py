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