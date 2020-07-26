#!/bin/bash
NAUTYPATH="/home/grudorff/opt/nauty/nauty27r1"
NATOMS=$1
DELTAZMAX=$2


LABEL="${NATOMS}-${DELTAZMAX}"
# generate all connected graphs with NATOMS many nodes
$NAUTYPATH/geng -c -t -f -D3 $NATOMS > graphlist-$LABEL
# remove all but graphs that can be on a hexgrid
python check-hex-representable.py graphlist-$LABEL > graphlist-hex-$LABEL
# build all colored graphs
$NAUTYPATH/vcolg -m$(($DELTAZMAX*2+1)) -T graphlist-hex-$LABEL graphlist-colored-$LABEL
# filter all non-alchemical symmetry graphs
echo -e "\n\nEQUATIONS:\n\n"
# convert graphs into unique equations
cat graphlist-colored-$LABEL | python filter-alchemy-graphs.py  | python graph-equation.py | sort -u
