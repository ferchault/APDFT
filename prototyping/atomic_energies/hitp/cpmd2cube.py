import subprocess
import os
import shutil
import sys
import pickle

def run_cpmp2cube(cpmd2cube_exe, work_dir):
    os.chdir(work_dir)
    cpmd2cube_exe = '/home/misa/opt/cpmd2cube/cpmd2cube.x'
    process = subprocess.run([cpmd2cube_exe, 'DENSITY'], capture_output = True,text=True)
    return(process)

def check_conversion(stdout, stderr):
    if stderr == 'STOP read_density_file, c0 incorrectly sized\n':
        return('c0 error')
    elif len(stdout) > 2 and len(stderr)==0:
        lines = stdout.split('\n')
        sum_squared = float(lines[2].split()[-1][:-1])
        if sum_squared < 1:
            return('reasonable density')
        else:
            return('sum^2 to large')
    else:
        return('unexpected event occured')

def save_obj(obj, fname ):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    

density_dirs = sys.argv[1]
workdirs = []
with open(density_dirs, 'r') as f:
    for line in f:
        workdirs.append(line.strip('\n'))
        

cpmd2cube_exe = sys.argv[2]
run_info = dict()
for wd in workdirs:
    # make cube file
    i = 0
    success = 'c0 error'
    while success == 'c0 error':
        print(f'Try {i}')
        process = run_cpmp2cube(cpmd2cube_exe, wd)
        success = check_conversion(process.stdout, process.stderr)
        run_info[wd] = success
        i += 1
    # move cube file
    if success == 'reasonable density':
        src = os.path.join(wd, 'DENSITY.cube') # cube path source

        # make cube dir if not existing
        cube_dir = os.path.join('/'.join(wd.split('/')[:-1]), 'cube-files')
        os.makedirs(cube_dir, exist_ok=True)

        lam_val = wd.split('/')[-1]
        cube_name = f'{lam_val}.cube' # new cube name
        dest = os.path.join(cube_dir, cube_name) # cube path destination
        
        shutil.move(src, dest)
        
        # delete DENSITY.pdb file
        density_pdb = os.path.join(wd, 'DENSITY.pdb')
        os.remove(density_pdb)
    
    
    # save run_info
    slurm_id = sys.argv[3]
    slurm_submitdir = sys.argv[4]
    info_filepath = os.path.join(slurm_submitdir, f'run_info_{slurm_id}')
    save_obj(run_info, info_filepath)