#!/usr/bin/env python
import numpy as np
import tqdm
import gzip
import numba

nlines = 1620580590

@numba.njit
def convert_line(line):
	data = ""
	skip = False
	nbn = 0
	for element in line[1:]:
		if skip:
			skip = False
			continue
		skip = True
		if element == "0": 
			data += "6 "
			continue
		if element == "1": 
			data += "7 "
			nbn += 1
			continue
		if element == "2": 
			data += "5 "
			continue
	return data[:-1], nbn
	
	
fhs = {}
for nbn in range(11):
	fhs[nbn] = open('out-%d' % nbn, 'w', buffering=1024*1024*200)
	
for line in tqdm.tqdm(gzip.open("out.list"), total=nlines):
	line, nbn = convert_line(line.decode('utf-8'))
	fhs[nbn].write(line + "\n")
for nbn, fh in fhs.items():
	fh.close()
