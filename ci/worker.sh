#!/bin/bash
BASEDIR=$(pwd)
CIDIR=~/_CI
HEAD=$(git rev-parse HEAD)

source ~/.bashrc
cd $CIDIR
rm -rf *
conda activate ci-env

PATH="$BASEDIR/src:$PATH" python $BASEDIR/ci/ci.py $HEAD $BASEDIR/src
