import os

compounds = []
with open('/home/misa/projects/Atomic-Energies/data/vacuum_reference/qm9_data/dirs', 'r') as f:
    for line in f:
        compounds.append(line.strip('\n'))

path = '/home/misa/projects/Atomic-Energies/data/vacuum_reference/qm9_data'
for c in compounds:
    file = ['basis def2-qzvp\n', 'intg_method simps\n']
    file.append(f'structure_file {c}.xyz')
    save_dir = os.path.join(path, c)
    filepath = os.path.join(save_dir, 'input_parameters')
    with open(filepath, 'w') as f:
        f.writelines(file)