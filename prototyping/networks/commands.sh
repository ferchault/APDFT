#!/bin/bash
profile() {
	python -m cProfile -o program.prof rankBN.py ../../test/benzene.xyz benzene.mol2
	ROOT="$(grep -n "def do_main" rankBN.py  | sed 's/:.*/:do_main/;s/^/rankBN:/')"
	gprof2dot -f pstats program.prof --colour-nodes-by-selftime --root="$ROOT" | dot -Tsvg -o profile.svg
}

timeit() {
	python -m timeit -n 1 -r 2 -s 'import rankBN' 'rankBN.do_main("../../test/benzene.xyz", "benzene.mol2")'
}

"$@"
