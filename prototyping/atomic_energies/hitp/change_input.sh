#!/bin/bash

for directory in $(cat r4); do
#    sed -i '/\s\sOPTIMIZE WAVEFUNCTION/a \s\sRESTART WAVEFUNCTION LATEST'  $directory"run.inp"
#    sed -i '/\s\sRESTART WAVEFUNCTION LATEST/a \  PCG MINIMIZE'  $directory"run.inp"
    sed -i '/\s\sTIMESTEP/!b;n;c \    0\.005' $directory"run.inp"
done
