clear
use "FEEL_S1_MERGED_V3_DATE_tc.DTA"

global EDM_VERBOSITY=1
global EDM_NTHREADS=40

set seed 1
timer clear
timer on 1
edm explore PA, dt k(-1) e(10) force alg(simplex) predict(exploreE10) saveinputs(feel_explore_dt_all_neighbours_E_10.json)
timer off 1
timer list

set seed 1
timer clear
timer on 1
edm xmap NA PA, dt k(-1) force alg(smap) oneway predict(xmapE2) saveinputs(feel_xmap_dt_all_neighbours_E_2.json)
timer off 1
timer list


set seed 1
timer clear
timer on 1
edm xmap NA PA, dt k(-1) e(5) force alg(smap) oneway predict(xmapE5) saveinputs(feel_xmap_dt_all_neighbours_E_5.json)
timer off 1
timer list

set seed 1
timer clear
timer on 1
edm xmap NA PA, dt k(-1) e(10) force alg(smap) oneway predict(xmapE10) saveinputs(feel_xmap_dt_all_neighbours_E_10.json)
timer off 1
timer list
