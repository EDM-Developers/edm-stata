* cut-down test file
clear

if c(MP) {
	qui set processor 1
}

set obs 50000
set seed 12345678

gen t = _n
tsset t

gen double x = 0.2 if _n==1
gen double y = 0.4 if _n==1

* Create a dynamic system
local r_x 3.625
local r_y 3.77
local beta_xy = 0.05
local beta_yx=0.4
local tau = 1
drawnorm double u1 u2
qui {
    forvalues i=2/`=_N' {
        replace x=l.x *(`r_x' *(1-l.x)-`beta_xy'*l.y) in `i'
        replace y=l.y *(`r_y' *(1-l.y)-`beta_yx'*l`tau'.x) in `i'
    }
}

keep in 300/50000

cap rm logmaplarge.json

timer clear
timer on 1

edm xmap x y, theta(0.2) algorithm(smap) saveinputs(logmaplarge) verbosity(1)

timer off 1
timer list

keep in 1/5000

cap rm logmapsmall.json

timer clear
timer on 1

edm explore x, e(10) saveinputs(logmapsmall)

timer off 1
timer list

clear

use "FEEL_S1_MERGED_V3_DATE_tc.DTA"

keep in 1/5000

cap rm affectsmall.json

timer clear 1
timer on 1
edm xmap PA NA, dt e(10) k(-1) force alg(smap) saveinputs(affectsmall) 
timer off 1
timer list 1

keep in 1/5000

cap rm affectbige.json

timer clear 1
timer on 1
edm xmap PA NA, dt e(150) k(20) force alg(smap) saveinputs(affectbige)
timer off 1
timer list 1
