#!/bin/bash
# call as run.sh DELTALAMBDA

CP2K="/home/guido/opt/cp2k/cp2k-8.2-Linux-x86_64.ssmp"

function do_run() {
	RUNDIR="RUN_$1"
	# generate input
	mkdir $RUNDIR
	cp tpl/BASIS_SET tpl/POTENTIAL $RUNDIR
	BWD=$(echo "-$1" | sed 's/--//')
	cat tpl/run.inp | grep -v MAX_SCF | sed "s/UUU/$1/;s/DDD/$BWD/;s/RESTART/ATOMIC/" > $RUNDIR/scf.inp

	echo "Run reference scf"
	(cd $RUNDIR; $CP2K scf.inp > scf.log)
	cat tpl/run.inp | sed "s/UUU/1/;s/DDD/-1/" > $RUNDIR/target.inp
	cat tpl/run.inp | sed "s/UUU/0/;s/DDD/0/" > $RUNDIR/reference.inp

	echo "Run singlepoints for Hellmann-Feynman derivatives"
	cp $RUNDIR/actest-RESTART.wfn $RUNDIR/scf.wfn
	(cd $RUNDIR; $CP2K target.inp > target.log)
	cp $RUNDIR/scf.wfn $RUNDIR/actest-RESTART.wfn
	(cd $RUNDIR; $CP2K reference.inp > reference.log)
}

for dlambda in $(seq -1 0.1 1)
do
	echo $dlambda
	#do_run $dlambda
done	
do_run 0.03
do_run 0.01
do_run 0.02
do_run -0.01
do_run -0.02
do_run -0.03
