import ase.io as aio

def generate_input(xyz_filepath, mop_filepath):
    """
    reads molecule geometry from xyz_filepath (in xyz-format) and returns .mop MOPAC input-file
    at location given by mop_filepath
    PM7 hamiltonian is used and all coordinates except for the ones of the first atom are optimized
    """
    # read xyz file
    #xyz_filepath = '/data/sahre/projects/atomic-energies/data/bonding_trends/pbe0_data/row_2/fragments_single/NN_opt.xyz'
    atoms = aio.read(xyz_filepath)

    # write input file
    mop_file = ['PM7\n', 'Molecule\n', 'All coordinates are cartesian\n']

    # optimize all atoms except for first one
    opt = len(atoms)*[1]
    opt[0] = 0

    for el, R, optI in zip(atoms.get_chemical_symbols(), atoms.get_positions(), opt):
        line = f'{el} {R[0]} {optI} {R[1]} {optI} {R[2]} {optI}\n'
        mop_file.append(line)

    # write code to .mop file
    #mop_filepath = '/data/sahre/projects/atomic-energies/data/bonding_trends/pm7/test.mop'
    with open(mop_filepath, 'w') as f:
        #for line in mop_file:
        f.writelines(mop_file)