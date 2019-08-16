#!/bin/bash

rm tmp_special
find . -name "run.log" | grep $1 > tmp_special

for file in $(cat tmp_special)
do
grep -l "JOB LIMIT TIME" $file
done
		
