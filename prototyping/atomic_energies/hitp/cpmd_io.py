import os
import glob
import pathlib

##########################################################
#                  Modify input file                     #
##########################################################

def disable_initialize_random(file_list):
    # disable option if enabled
    for i, line in enumerate(file_list):
        if 'INITIALIZE WAVEFUNCTION RANDOM' in line:
            file_list[i] = '  iNITIALIZE WAVEFUNCTION RANDOM\n'
    return(file_list)

def enable_initialize_random(file_list):
    enabled = False
    # check if already enabled
    for i, line in enumerate(file_list):
        if 'INITIALIZE WAVEFUNCTION RANDOM' in line:
            enabled = True
            break
        elif 'iNITIALIZE WAVEFUNCTION RANDOM' in line:
            file_list[i] = '  INITIALIZE WAVEFUNCTION RANDOM\n'
            enabled = True
            break
    if not enabled:
        file_list.insert(2, '  INITIALIZE WAVEFUNCTION RANDOM\n')
    return(file_list)
            
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

##########################################################
#                    Parse log-file                       #
##########################################################

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

def get_energy_contribution(logfile, name):
    energy = 0
    for line in logfile:
        if name in line:
            energy = float(line.split()[-2]) # return last value (should be after calc is converged)
    return(energy)

##########################################################
#                    Error handling                      #
##########################################################

error_message_lookup = {'e000': 'Unknwon error','e001':'Program received signal SIGSEGV: Segmentation fault - invalid memory reference.', 'e002':'Force Terminated job', 
                        'e003':'module: command not found', 'e004':'Exited with exit code 231', 'e005':'forcing job termination', 'e006':'Disk quota exceeded',
                       'e007':'Local-Error-file exists'}

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

def error_empty(workdir):
    error_files = glob.glob(os.path.join(workdir, 'error.*'))
    error_files.extend(glob.glob(os.path.join(workdir, 'slurm*')))
    error_files_ct = []
    for e in error_files:
        fname = pathlib.Path(e)
        error_files_ct.append((fname.stat().st_mtime, fname.stat().st_size))
    error_files_ct.sort()
    is_empty = bool(error_files_ct[-1][1] == 0)
    return(is_empty)

def get_last_errorfile(workdir):
    error_files = glob.glob(os.path.join(workdir, 'error.*'))
    error_files.extend(glob.glob(os.path.join(workdir, 'slurm*')))
    error_files_ct = []
    for e in error_files:
        fname = pathlib.Path(e)
        error_files_ct.append((fname.stat().st_mtime, fname.name))
    error_files_ct.sort()
    lastfile = os.path.join(workdir, error_files_ct[-1][1])
    return(lastfile)

def parse_error_file(file_content):
    for line in file_content:
        for k in error_message_lookup.keys():
            if error_message_lookup[k] in line:
                return(k)
    return('e000')

##########################################################
#                     Parse pp-files                     #
##########################################################

def parse_pp_file(file):
    pp_param = {'Z_ion':0, 'r_loc':0, 'c1':0, 'c2':0, 'c3':0,'c4':0}
    with open(file, 'r') as f:
        for line in f:
            if 'ZV' in line:
                pp_param['Z_ion'] = float(line.strip('\n').split()[-1])
            elif 'RC' in line:
                pp_param['r_loc'] = float(line.strip('\n').split()[0])
            elif '#C' in line:
                items = line.strip('\n').split()
                num_coeff = int(items[0])
                for i, c in enumerate(items[1:num_coeff+1]):
                    pp_param[f'c{i}'] = float(c)
    return(pp_param)