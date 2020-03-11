#!/bin/bash

for runfolder in $(cat r4); 
do
	(cd $runfolder; /scicore/home/lilienfeld/rudorff/share/wishlist-local ../../../../../job_scripts/run_6hours_wl.job);
done

