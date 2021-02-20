def write_atom(atomsym, coordinates):
    """
    prepare the input for one atom:
    the name of the pp is 'element_name' + idx of atom in Compound object + '_SG_LDA'
    the coordinates are read from Compund as well (must be shifted to center before)
    """
    line1 = f'*{atomsym}_SG_LDA FRAC\n'
    line2 = ' LMAX=S\n'
    line3 = ' 1\n'
    line4 = ' ' + str(coordinates[0]) + ' ' + str(coordinates[1]) + ' ' + str(coordinates[2]) + '\n'
    return( [line1, line2, line3, line4] )
    
def write_atom_section(atomsymbols, coordinates):
    """
    atomsymbols: list of element names
    coordinates: list of coordinates
    concantenates inputs for individual atoms to one list where each element is one line of the input file
    """
    atom_section = ['&ATOMS']
    for atsym, c in zip(atomsymbols, coordinates):
        atom = write_atom(atsym, c)
        atom_section.extend(atom)
    atom_section.append('&END')
    return(atom_section)

def write_input(atomsymbols, charge, coordinates, gpts, L, write_path, template_path='/home/misa/projects/Atomic-Energies/data/cpmd_params_template.inp'):
    """
    writes input file for molecule with specified parameters boxisze L, charge, number of gpts for mesh
    """
    param_section = write_params(L, charge, gpts, template_path='/home/misa/projects/Atomic-Energies/data/cpmd_params_template.inp')
    atom_section = write_atom_section(atomsymbols, coordinates)
    with open(write_path, 'w') as f:
        f.writelines(param_section+['\n']+atom_section)
        
def write_params(L, charge, gpts, template_path='/home/misa/projects/Atomic-Energies/data/cpmd_params_template.inp'):
    """
    add correct parameters for boxsize L, charge and gpts to template
    """
    with open(template_path, 'r') as f:
        template_params = f.readlines()

    for i, line in enumerate(template_params):
        if 'CELL ABSOLUTE' in line:
            template_params[i+1] = f'        {L} {L} {L} 0.0 0.0 0.0\n'
        elif 'CHARGE' in line:
            template_params[i+1] = f'        {charge}\n'
        elif 'MESH' in line:
            template_params[i+1] = f'    {gpts} {gpts} {gpts}\n'
    return(template_params)
