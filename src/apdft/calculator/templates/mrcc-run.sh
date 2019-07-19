#!/bin/bash
mv run.inp MINP
dmrcc > run.log
rm fort.*
mv MINP run.inp
