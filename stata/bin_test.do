clear

cap log close _all
log using "bin_test.log", replace nomsg

set linesize 255
set obs 500
set seed 12345678
if c(MP) {
    qui set processor 1
}

global EDM_VERBOSITY=1
global EDM_NTHREADS=4

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

* Create binary variables
replace x=1 if abs(x) > 0.5
replace x=0 if abs(x) <= 0.5

replace y=1 if abs(y) > 0.65
replace y=0 if abs(y) <= 0.65


* burn the first 300 observations
keep in 300/500

* Determining the complexity of the system

edm xmap x y

cap log close _all
