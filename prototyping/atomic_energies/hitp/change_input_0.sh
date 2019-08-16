#!/bin/bash

for directory in $(cat restart0); do
    sed -i '/\s\sOPTIMIZE WAVEFUNCTION/a \  RESTART WAVEFUNCTION LATEST'  $directory"run.inp"
    sed -i '/\s\sTIMESTEP/!b;n;c \    0\.05' $directory"run.inp"
done
