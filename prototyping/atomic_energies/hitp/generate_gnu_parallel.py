import sys

path_info = sys.argv[1]
cores = sys.argv[2]
paths = []
with open(path_info, 'r') as f:
    for line in f:
        paths.append(line.strip('\n'))

lines = []
for p in paths:
    lines.append(f'cd {p}; mpirun -n {cores} /home/misa/opt/CPMD_fftw_serial/bin/cpmd.x run.inp > run.log\n')
with open(sys.argv[3], 'w') as f:
    f.writelines(lines)
