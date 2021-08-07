import sys

# paths contains directories in which cpmd shall be executed

path_info = sys.argv[1]
cores = sys.argv[2]
cpmd_bin = sys.argv[3]
paths = []
with open(path_info, 'r') as f:
    for line in f:
        paths.append(line.strip('\n'))

lines = []
for p in paths:
    lines.append(f'cd {p}; mpirun --bind-to none -n {cores} {cpmd_bin} run.inp > run.log\n')
assert 'cpmd.x' not in sys.argv[4], 'Dont overwrite cpmd.x again!'
with open(sys.argv[4], 'w') as f:
    f.writelines(lines)
