clear

import delimited "affect.csv"

gen t = _n
tsset t

global EDM_VERBOSITY = 1
global EDM_NTHREADS = 4
global EDM_SAVE_INPUTS = "speed-test"
cap rm speed-test.json

set seed 1
timer clear
timer on 1
edm explore pa, dt k(-1) e(10) force alg(simplex)
timer off 1
timer list

set seed 1
timer clear
timer on 1
edm xmap na pa, dt k(-1) force alg(smap) oneway
timer off 1
timer list


set seed 1
timer clear
timer on 1
edm xmap na pa, dt k(-1) e(5) force alg(smap) oneway
timer off 1
timer list
