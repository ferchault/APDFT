#!/bin/bash
source /home/guido/opt/conda/bin/activate
conda activate regular
NAUTYDIR="/home/guido/opt/nauty/nauty27r1"
/home/guido/wrk/polya/python/polya.py $1 $2 -generators <(cat - | $NAUTYDIR/dreadnaut | grep -v level | grep -v orbit | grep -v cpu | sed 's/$/-/' | tr -d '\n' | sed 's/-  [ ]*/ /g;s/-[ ]*/\n/g')
