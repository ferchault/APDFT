#!/bin/bash
profile() {
	python -m cProfile -o program.prof rankBN.py ../../test/benzene.xyz benzene.mol2
	ROOT="$(grep -n "def work" rankBN.py  | sed 's/:.*/:work/;s/^/rankBN:/')"
	gprof2dot -f pstats program.prof --root="$ROOT" | dot -Tpng -o profile.png
}

"$@"
