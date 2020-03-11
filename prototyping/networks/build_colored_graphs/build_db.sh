#!/bin/bash
PATH="/mnt/c/Users/guido/opt/nauty/nauty27rc3:$PATH"

python ../mol2_to_graph6.py inp.mol2 out.g6

vcolg -m3 -T out.g6 | gzip > out.list.gz

python ../list2binary.py
