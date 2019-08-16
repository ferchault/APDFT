#!/bin/bash

for directory in $(cat r4); 
do
    mkdir -m 777 -p $(sed 's/run4/run5/' <<< $directory)
    mv $directory*_SG_LDA $(sed 's/run4/run5/' <<< $directory)
    mv $directory"run.inp" $(sed 's/run4/run5/' <<< $directory)
    mv $directory"RESTART.1" $directory"LATEST" $(sed 's/run4/run5/' <<< $directory) 
done
