#!/bin/bash
NAUTYPATH="/home/grudorff/opt/nauty/nauty27r1"
#NAUTYPATH="/mnt/c/Users/guido/opt/nauty/nauty27rc3/"

NATOMS=$1	# number of changed sites (should not be too large...)
DELTAZMAX=$2	# 1, 2, ..., n
GRID=$3		# HEXAGONAL or DIAMOND
TUPLES=$4	# 2 for bonds, 3 for three-body

LABEL="${NATOMS}-${DELTAZMAX}-${GRID}-${TUPLES}"
SCRATCHDIR="/tmp"

# generate all connected graphs with NATOMS many nodes
$NAUTYPATH/geng -c -t -f -D3 $NATOMS > $SCRATCHDIR/graphlist-$LABEL

# remove all but graphs that can be on the grid
if [ "$GRID" == "HEXAGONAL" ]
then
	python check-hex-representable.py $SCRATCHDIR/graphlist-$LABEL > $SCRATCHDIR/graphlist-grid-$LABEL
elif [ "$GRID" == "DIAMOND" ]
then
	echo "NOTIMPL"
else
	echo "Unsupported grid."
	exit 1
fi

# build all colored graphs
$NAUTYPATH/vcolg -m$(($DELTAZMAX*2+1)) -T $SCRATCHDIR/graphlist-grid-$LABEL $SCRATCHDIR/graphlist-colored-$LABEL

# convert graphs into unique equations
cat $SCRATCHDIR/graphlist-colored-$LABEL | python filter-alchemy-graphs.py | python graph-equation.py $TUPLES | sort -u
