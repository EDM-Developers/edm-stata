clear
use "FEEL_S1_MERGED_V3_DATE_tc.DTA"

global EDM_VERBOSITY = 1
global EDM_NTHREADS = 40

global EDM_SAVE_INPUTS = "arrayfire-tests"
cap rm arrayfire-tests.json

set seed 1
timer clear
timer on 1
edm explore PA, dt k(-1) e(10) force alg(simplex) predict(exploreE10)
timer off 1
timer list

set seed 1
timer clear
timer on 1
edm xmap NA PA, dt k(-1) force alg(smap) oneway predict(xmapE2)
timer off 1
timer list


set seed 1
timer clear
timer on 1
edm xmap NA PA, dt k(-1) e(5) force alg(smap) oneway predict(xmapE5)
timer off 1
timer list

set seed 1
timer clear
timer on 1
edm xmap NA PA, dt k(-1) e(10) force alg(smap) oneway predict(xmapE10)
timer off 1
timer list
