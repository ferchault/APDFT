#!/bin/bash

find . -name RESTART.1 | grep "ve_38" | cut -d/ -f1,2 > tmp_submit
sort tmp_submit > submitted_sorted
rm tmp_submit
diff all_sorted submitted_sorted | grep "^<" | sed 's/<\s//' > not_submitted
rm submitted_sorted
shuf not_submitted | head -n 250 > new_submission
rm not_submitted
