import pathlib
import glob
import os
import sys
import pandas as pd
import numpy as np

def analyse_logfile(logfile):
    status = 'broken'
    # determine status of calculation
    for i, line in enumerate(logfile):
        if ' JOB LIMIT TIME EXCEEDED FOR A NEW LOOP' in line:
            status = 'not converged'
            break
        # FINAL RESULTS also appears in unconverged calculations but only after JOB LIMIT ..., so this should never be seen for unconverged calcs
        elif 'FINAL RESULTS' in line and not status == 'not converged': 
            iteration, conv, success = check_convergence(logfile, i)
            if not success:
                status = 'broken'
            elif conv < 1e-6:
                status = 'converged'
            else:
                status = 'not converged'
                #print('CPMD did not run out of time and the calculation terminated normally, but it did apparently not converge.')
#         elif 'CPU TIME' in line:
#             timeline = line.split()
#             time = float(timeline[3])*3600 + float(timeline[5])*60 + float(timeline[7])

#     if status == 'broken':
#         time = None
    return(status)

def check_convergence(file, linenumber):
    """
    if converged 8 or 9 lines above FINAL RESULTS should be last iteration
    try to parse this one
    """
    success = True
    try:
        last_iteration = file[linenumber-8]
        last_iteration = last_iteration.strip('\n')
        iteration, conv = int(last_iteration.split()[0]), float(last_iteration.split()[1])
    except:
        try:
            last_iteration = file[linenumber-9]
            last_iteration = last_iteration.strip('\n')
            iteration, conv = int(last_iteration.split()[0]), float(last_iteration.split()[1])
        except:
            print("Could not parse log-file")
            success = False
    return(iteration, conv, success)

def check_errorfile(workdir):
    """
    parses latest errorfile for errors
    """
    
    ### find last error file
    error_files = glob.glob(os.path.join(workdir, 'error.*'))
    error_files.extend(glob.glob(os.path.join(workdir, 'slurm*')))
    error_files_ct = []
    for e in error_files:
        fname = pathlib.Path(e)
        error_files_ct.append((fname.stat().st_mtime, fname.stat().st_size, e))
    error_files_ct.sort()
    last_errorfile_size = error_files_ct[-1][1]
    last_errorfile = error_files_ct[-1][2]
    
    if last_errorfile_size == 0: # error file empty
        status = 'no error'
    else:
        # read error file
        status = 'no error'
        allowed_warning1 = 'Note: The following floating-point exceptions are signalling: IEEE_DENORMAL'
        allowed_warning2 = 'Note: The following floating-point exceptions are signalling: IEEE_UNDERFLOW_FLAG IEEE_DENORMAL'
        allowed_warning3 = '*_SG_LDA’: No such file or directory'
        with open(last_errorfile, 'r') as f:
            for line in f:
                if not allowed_warning1 in line and not allowed_warning2 in line and not allowed_warning3 in line:
                    status = 'broken'
                    break
    return(status)

def enable_restart(database):
    # enable restart option
    workdirs_restart = database.loc[(database['status'] == 'not converged') & (database['restart'] == False), 'workdir']
    restart_option = []
    for wd in workdirs_restart:
        restart_option.append(fix_input_file(wd))
    database.loc[(database['status'] == 'not converged') & (database['restart'] == False), 'restart'] = restart_option

def error_empty(workdir):
    """
    checks if latest error file is empty
    """
    error_files = glob.glob(os.path.join(workdir, 'error.*'))
    error_files.extend(glob.glob(os.path.join(workdir, 'slurm*')))
    error_files_ct = []
    for e in error_files:
        fname = pathlib.Path(e)
        error_files_ct.append((fname.stat().st_mtime, fname.stat().st_size))
    error_files_ct.sort()
    is_empty = bool(error_files_ct[-1][1] == 0)
    return(is_empty)

def fix_input_file(p, restart_corrupted=False):
# read input file
    inp_file = []
    with open(os.path.join(p, 'run.inp'), 'r') as f:
        for line in f:
            inp_file.append(line)
    # set restart option correctly
    restart_file = os.path.join(p, 'RESTART.1')
    restart_exists = os.path.isfile(restart_file)
    found_restart_option = False
    restart_enabled = False
    for i, line in enumerate(inp_file):
        if 'RESTART WAVEFUNCTION LATEST' in line and restart_exists and not restart_corrupted:
            found_restart_option = True # everything is ok
            print(f'Everything ok in {p}')
            restart_enabled = True
        # disable restart if there is no restart file
        if 'RESTART WAVEFUNCTION LATEST' in line and not restart_exists:
            inp_file[i] = '  rESTART WAVEFUNCTION LATEST\n'
            found_restart_option = True
            restart_enabled = False
            print(f'Disabled restart in {p} because couldnt find a restart file. I hope you dont mind.')
        # disable restart if there is restart file that is corrupted
        if 'RESTART WAVEFUNCTION LATEST' in line and restart_corrupted:
            inp_file[i] = '  rESTART WAVEFUNCTION LATEST\n'
            found_restart_option = True
            restart_enabled = False
            print(f'Disabled restart in {p} because restart file is corrupted. Is that ok with you?')
        # enable restart if there is a restart file that is not corrupted
        elif 'rESTART WAVEFUNCTION LATEST' in line and restart_exists and not restart_corrupted:
            inp_file[i] = '  RESTART WAVEFUNCTION LATEST\n'
            found_restart_option = True
            restart_enabled = True
            print(f'Enabled restart (from rESTART) in {p} because there is an intact restart file. Isn\'t that great, sunshine?')
    # enable restart if there is a restart file that is not corrupted and restart not added yet
    if not found_restart_option and restart_exists and not restart_corrupted:
        for i, line in enumerate(inp_file):
            if 'OPTIMIZE WAVEFUNCTION' in line:
                print(f'Added the restart option in {p} because there is an intact restart file. This should help you to finish your PhD in time.')
                inp_file.insert(i+1, '  RESTART WAVEFUNCTION LATEST\n')
                restart_enabled = True
                break
    
    # check that PCG MINIMIZE is used
    # change TIMESTEP to 5 if necessary
    timestep = False
    for i, line in enumerate(inp_file):
        if 'TIMESTEP' in line:
            timestep = True
            inp_file[i+1] = '    5\n'
            if not 'PCG MINIMIZE' in inp_file[i-1]:
                inp_file.insert(i, '  PCG MINIMIZE\n')
                print(f'I think you forgot to use the PCG minimizer instead of DIIS in {p}. I changed that for you. Aint I a nice program?')
    assert timestep

    with open(os.path.join(p, 'run.inp'), 'w') as f:
        for line in inp_file:
            f.write(line)
    return(restart_enabled)
            
# check status
def get_status(workdir):
    # no log file
    time = None
    if os.path.isfile(os.path.join(workdir, 'finished')):
        # either converged, not converged, broken
        status = get_status_finished(workdir)
    elif os.path.isfile(os.path.join(workdir, 'submitted')):
        status = 'submitted'
    elif os.path.isfile(os.path.join(workdir, 'running')):
        status = 'running'
    elif os.path.isfile(os.path.join(workdir, 'run.log')):
        status = get_status_finished(workdir)
    else:
        status = 'undefined'

    return(status)

def get_status_finished(workdir):
    # read last error file
    status = check_errorfile(workdir)
    if status == 'no error':
        logfile = []
        with open(os.path.join(workdir, 'run.log'), 'r') as f:
            logfile = f.readlines()
        status = analyse_logfile(logfile)
    return(status)

def status_report(db, save_path):
    not_converged = db.loc[db['status']=='not converged', 'workdir']
    broken = db.loc[db['status']=='broken', 'workdir']
    undefined = db.loc[db['status']=='undefined', 'workdir']
    
    with open(os.path.join(save_path, 'not_converged'), 'w') as f:
        for line in not_converged:
            f.write(line+'\n')
    
    with open(os.path.join(save_path, 'broken'), 'w') as f:
        for line in broken:
            f.write(line+'\n')
        
    with open(os.path.join(save_path, 'undefined'), 'w') as f:
        for line in undefined:
            f.write(line+'\n')
    
    summary = dict()
    for s in ['not converged', 'converged', 'broken', 'submitted', 'running', 'undefined']:
        summary[s] = len(db.loc[db['status']==s])
    with open(os.path.join(save_path, 'status_report'), 'w') as f:
        for k in summary.keys():
            f.write(f"{k}: {summary[k]}\n")

def update(database):
    """
    update of status of database entries
    """
    workdirs = list(database.loc[database['status'] != 'converged' ,'workdir'])
    
    for wd in workdirs:
        #print(workdir)
        status = get_status(wd)
        database.loc[database['workdir'] == wd ,'status'] = status
#         if time:
#             if np.isnan(database.loc[database['workdir'] == wd, 'time'].item()):
#                 database.loc[database['workdir'] == wd , 'time'] = time
#             else:
#                 database.loc[database['workdir'] == wd ,'time'] += time
                              
if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = '/scicore/home/lilienfeld/sahre0000/projects/atomic_energies/results/amons/amons_32'
    # load database
    print('Loading Database...', flush = True)
    db = pd.read_pickle(os.path.join(path, 'calculation_manager.pd'))
    # update
    print('Updating database...', flush = True)
    update(db)
    print('Enabling restart option...', flush = True)
    enable_restart(db)
    # save results
    print('Writing status report...', flush = True)
    status_report(db, path) 
    print('Saving Database...', flush = True)
    db.to_pickle(os.path.join(path, 'calculation_manager.pd'))
    print('All done! Maybe the restart option should be enabled in some input files.', flush= True)
