Takes a single binary list of BN-dopings and applies SQM calculations on all these. Return the energies for a specified range in order.

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
g++ -I /mnt/c/Users/guido/opt/xtb/6.2.2/include/ -L /mnt/c/Users/guido/opt/xtb/6.2.2/lib/  sqm.cpp -lxtb_shared
