#!/bin/bash
BASEDIR=$(pwd)
CIDIR=~/_CI
HEAD=$(git rev-parse HEAD)

source ~/.bashrc
conda activate ci-env
pip install -r requirements.txt
cd dep
./install.sh

cd $CIDIR
rm -rf *

if [ "$1" == "ONLINE" ]
then
	MODE="ONLINE"
else
	MODE="OFFLINE"
fi
PATH="$BASEDIR/src:$PATH" python $BASEDIR/ci/ci.py $HEAD $BASEDIR/src $MODE
