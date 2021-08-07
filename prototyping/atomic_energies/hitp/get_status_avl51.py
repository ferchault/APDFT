import os

import sys
sys.path.insert(0, '/home/misa/git_repositories/APDFT/prototyping/atomic_energies/hitp')
sys.path.insert(0, '/home/sahre/git_repositories/APDFT/prototyping/atomic_energies/hitp')

from get_status_report import analyse_logfile

paths_file = sys.argv[1]
paths=[]
with open(paths_file, 'r') as f:
    for line in f:
        paths.append(line.strip('\n'))
results = []
for workdir in paths:
    if os.path.isfile(os.path.join(workdir, 'run.log')):
        with open(os.path.join(workdir, 'run.log'), 'r') as f:
            logfile = f.readlines()
        status = analyse_logfile(logfile)
        results.append(f'{workdir}: {status}\n')
        if status != 'converged':
            print(f'{workdir}: {status}')
    else:
        results.append(f'{workdir}: not started\n')

cwd = os.getcwd()
print(cwd)
with open(os.path.join(cwd, 'results'), 'w') as f:
    f.writelines(results)