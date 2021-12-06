clear

import delimited "affect.csv"

gen t = _n
tsset t

global EDM_VERBOSITY = 1
global EDM_NTHREADS = 4

timer on 2

set seed 1
timer clear 1
timer on 1
edm explore pa, dt k(-1) e(10) alg(simplex)
timer off 1
timer list

set seed 1
timer clear 1
timer on 1
edm xmap na pa, dt k(-1) alg(smap) oneway
timer off 1
timer list


set seed 1
timer clear 1
timer on 1
edm xmap na pa, dt k(-1) e(5) alg(smap) oneway
timer off 1
timer list

timer off 2
timer list
