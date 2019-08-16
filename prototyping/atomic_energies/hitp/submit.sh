#!/bin/bash

for molfolder in $(cat new_submission); 
do 
	for runfolder in $molfolder/run0/*; 
	do 
		(cd $runfolder; /scicore/home/lilienfeld/rudorff/share/wishlist-local ../../../../../job_scripts/run_6hours_wl.job); 
	done; 
done

