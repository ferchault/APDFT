import pathlib
import glob
import os
import pandas as pd
import numpy as np

# check status
def get_status(workdir):
    # no log file
    time = None
    if os.path.isfile(os.path.join(workdir, 'finished')):
        # either converged, not converged, broken
        status, time = get_status_finished(workdir)
    elif os.path.isfile(os.path.join(workdir, 'submitted')):
        status = 'submitted'
    elif os.path.isfile(os.path.join(workdir, 'running')):
        status = 'running'
    elif os.path.isfile(os.path.join(workdir, 'run.log')):
        status, time = get_status_finished(workdir)
    else:
        status = 'undefined'

    return(status, time)

def get_status_finished(workdir):
    # is the latest error file not empty? -> if yes, broken
    if not error_empty:
        status = 'broken'
        time = None
    else: # read logfile
        logfile = []
        with open(os.path.join(workdir, 'run.log'), 'r') as f:
            logfile = f.readlines()
        status, time = analyse_logfile(logfile)
    return(status, time)
    

    return(last_error_file)

def error_empty(workdir):
    error_files = glob.glob(os.path.join(workdir, 'error.*'))
    error_files_ct = []
    for e in error_files:
        fname = pathlib.Path(e)
        error_files_ct.append((fname.stat().st_mtime, fname.stat().st_size))
    error_files_ct.sort()
    is_empty = bool(error_files_ct[-1][1] == 0)
    return(is_empty)


def check_convergence(file, linenumber):
    """
    if converged9 lines above FINAL RESULTS should be last iteration
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

def analyse_logfile(logfile):
    status = 'broken'
    # determine status of calculation
    for i, line in enumerate(logfile):
        if ' JOB LIMIT TIME EXCEEDED FOR A NEW LOOP' in line:
            status = 'not converged'
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
        elif 'CPU TIME' in line:
            timeline = line.split()
            time = float(timeline[3])*3600 + float(timeline[5])*60 + float(timeline[7])

    if status == 'broken':
        time = None
    return(status, time)

def update(database):
    """
    update of status of database entries
    """
    for i in database.index:
        workdir = database.loc[i]['workdir']
        #print(workdir)
        status, time = get_status(workdir)
        database.loc[i]['status'] = status
        if time:
            if np.isnan(database.loc[i]['time']):
                database.loc[i]['time'] = time
            else:
                database.loc[i]['time'] += time
                
def status_report(db, save_path):
    not_converged = db.loc[db['status']=='not converged', 'workdir']
    broken = db.loc[db['status']=='broken', 'workdir']
    
    with open(os.path.join(save_path, 'not_converged'), 'w') as f:
        for line in not_converged:
            f.write(line+'\n')
    
    with open(os.path.join(save_path, 'broken'), 'w') as f:
        for line in broken:
            f.write(line+'\n')
    
    summary = dict()
    for s in ['not converged', 'converged', 'broken', 'submitted', 'running', 'undefined']:
        summary[s] = len(db.loc[db['status']==s])
    with open(os.path.join(save_path, 'status_report'), 'w') as f:
        for k in summary.keys():
            f.write(f"{k}: {summary[k]}\n")
            
# load database
db = pd.read_pickle("/scicore/home/lilienfeld/sahre0000/projects/atomic_energies/results/slice_ve38/calculations_databse")

# update
update(db)
# save results
status_report(db, '/scicore/home/lilienfeld/sahre0000/projects/atomic_energies/results/slice_ve38')