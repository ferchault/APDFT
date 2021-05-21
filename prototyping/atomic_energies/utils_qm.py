#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:23:17 2020

@author: misa
"""

import pickle

def load_obj(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
    
def save_obj(obj, fname ):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def read_list(fname):
    data = []
    with open(fname, 'r') as f:
        for line in f:
            data.append(line.strip('\n'))
    return(data)

def write_list(fname, ls):
    with open(fname, 'w') as f:
        for line in ls:
            f.write(line+'\n')