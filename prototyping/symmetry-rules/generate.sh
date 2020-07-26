#!/bin/bash
NAUTYPATH="/mnt/c/Users/guido/opt/nauty/nauty27rc3"
NATOMS=$1
DELTAZMAX=$2

# generate all connected graphs with NATOMS many nodes
$NAUTYPATH/geng -c -t -f -D3 $NATOMS > graphlist
# remove all but graphs that can be on a hexgrid
python check-hex-representable.py graphlist > graphlist-hex
# build all colored graphs
$NAUTYPATH/vcolg -m$(($DELTAZMAX*2+1)) -T graphlist-hex graphlist-colored
# filter all non-alchemical symmetry graphs
echo -e "\n\nEQUATIONS:\n\n"
# convert graphs into unique equations
cat graphlist-colored | python filter-alchemy-graphs.py  | python graph-equation.py  | sort -u
