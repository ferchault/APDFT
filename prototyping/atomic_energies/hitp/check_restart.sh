#!/bin/bash

# find directories with Restart and log-file (finished calculation and latest run for compound)
# finished: the paths of log-files in directories containing log and RESTART-files
if [ -f rexist ]; then
	rm rexist
fi
find . -name "RESTART.1" > rexist
sed -i 's/RESTART\.1//' rexist
if [ -f finished ]; then
	rm finished
fi
for direc in $(cat rexist); do find $direc -name "run.log"; done > finished

# find log-files in finished that did not converge and add their paths to restart_needed

if [ -f restart_needed ]; then
        rm restart_needed
fi

for file in $(cat finished)
do
grep -l "JOB LIMIT TIME" $file >> restart_needed
done

sed -i 's/run\.log//' restart_needed

rm rexist finished
