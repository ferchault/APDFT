#!/bin/bash

find . -name "run.log" | grep $1 > tmp

for file in $(cat tmp)
do
grep -L "JOB LIMIT TIME" $file
done
rm tmp		
