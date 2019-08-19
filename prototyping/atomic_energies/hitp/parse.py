#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:58:13 2019

@author: misa
"""

import os
import itertools
import re
import pandas as pd
import numpy as np

def read_filestream_from_line(fs, start_line):
    """
    file: path to file
    start_line: line from which on file will be read
    lines: lines of file from start_line to end as itertools object
    return: lines of file from start_line to end as elements of a list
    """
    not_start_line = lambda line: not(line == start_line)
#    with open(file, 'r') as f:
    lines = itertools.dropwhile(not_start_line, [line for line in fs])
    return(list(lines))
    
def read_filestream_till_line(fs, end_line):
    """
    file: path to file
    start_line: line from which on file will be read
    lines: lines of file from start_line to end as itertools object
    return: lines of file from start_line to end as elements of a list
    """
    not_end_line = lambda line: not(line in end_line)
#    with open(file, 'r') as f:
    lines = itertools.takewhile(not_end_line, [line for line in fs])
    return(list(lines))

def remove_strings_from_list(strings_to_remove, list_):
    """
    removes all elements from list_ that contain the patterns specified in strings_to_remove
    strings_to_remove: list of patterns (not necessarily a string)
    list_: list from which elements will be removed
    return: list without elements containing the patterns specified in strings_to_remove
    """
    remove_strings = lambda line : pattern_in_string(strings_to_remove, line)
    lines = itertools.filterfalse(remove_strings, [line for line in list_])
    return(list(lines))

def pattern_in_string(pattern_list, string):
    found = []
    for el in pattern_list:
        found.append(bool(re.search(el, string)))
    return(any(found))


def parse_log_file(path_log):
    # does log-file exist, if not calculation still running
    if not os.path.isfile(path_log):
        status = "Running or not started"
        return(status, None, None)
    else:
        # parse log-file
        # get gradient_max of last iteration
        with open(path_log) as fs:
            start = start = ' NFI      GEMAX       CNORM           ETOT        DETOT      TCPU\n'
            fs_cropped = read_filestream_from_line(fs, start)
            end_line = [' *            JOB LIMIT TIME EXCEEDED FOR A NEW LOOP            *\n', ' DENSITY WRITTEN TO FILE DENSITY\n']
            fs_cropped = read_filestream_till_line(fs_cropped, end_line)
            
        
        strings_to_remove = ['LINE', 'RESTART', '\*', 'ODIIS']
        fs_cropped = remove_strings_from_list(strings_to_remove, fs_cropped)
        
        # remove \n, split lines in separate values, remove empty lists (linebreaks)
        fs_cropped = [line.strip('\n)') for line in fs_cropped]
        fs_cropped=[line.split() for line in fs_cropped]
        fs_cropped = list(filter(None, fs_cropped))

        # test if fs_cropped empty
        if len(fs_cropped) == 0:
            status = 'broken'
            return(status, 'fs_cropped is empty', None)
        
        # use first line as header
        header = fs_cropped[0]
        # convert values to floats
        data=[ [float(el) for el in line] for line in fs_cropped[1:] ]
        
        # write scf information to panads dataframe
        df = pd.DataFrame(data, columns=header)
        gemax = df['GEMAX'][len(df['GEMAX'])-1]
        
        if np.dtype(gemax)!='float64':
            print('Problem with gemax in %s' % path_log)
        assert np.dtype(gemax)=='float64', "gemax = %s" % gemax
        
        num_it = int(df['NFI'][len(df['NFI'])-1])
        if type(num_it)!=int:
            print('Problem with num_it in %s' % path_log)
        assert type(num_it)==int, "num_it = %s" % num_it
        
        save_file = 'parsed_' + os.path.basename(path_log)
        dirname = os.path.dirname(path_log)
        #df.to_csv(os.path.join(dirname, save_file), sep='\t', header=True, index=False)
        status = "finished"
        return(status, gemax, num_it)
        
