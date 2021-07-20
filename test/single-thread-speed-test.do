clear

import delimited "affect.csv"

gen t = _n
tsset t

global EDM_VERBOSITY = 1
global EDM_NTHREADS = 1
global EDM_SAVE_INPUTS = "single-thread-speed-test"
cap rm single-thread-speed-test.json

set seed 1
timer clear
timer on 1
edm explore pa, dt k(-1) e(10) force alg(simplex)
timer off 1
timer list

set seed 1
timer clear
timer on 1
edm xmap na pa, dt k(-1) e(3) force alg(smap) oneway
timer off 1
timer list
