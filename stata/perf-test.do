* cut-down test file
clear

if c(MP) {
	set processor 1
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

* burn the first 300 observations
keep in 300/50000

timer clear
timer on 1

edm xmap x y, theta(0.2) algorithm(smap) saveinputs(perfinput.h5) verbosity(1)

timer off 1
timer list