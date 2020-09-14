/* edm paper replication do file
Jinjing Li, University of Canberra
Michael Zyphur, University of Melbourne
*/

discard

set more off

set scheme sj

/* Example with a Synthetic Dataset */
/* Create a dynamic system */

set obs 1000

gen t = _n
tsset t

gen x = 0.2 if _n==1
gen y = 0.3 if _n==1
gen z = 0.1 if _n==1

local r_x 3.79
local r_y 3.79
local r_z 3.77
local beta_xy = 0.0
local beta_yx=0.2
local beta_zy = 0.5
local tau = 1
drawnorm u1 u2
qui {
    forvalues i=2/`=_N' {
        replace x=l.x *(`r_x' *(1-l.x)-`beta_xy'*l.y) in `i'
        replace y=l.y *(`r_y' *(1-l.y)-`beta_yx'*l`tau'.x) in `i'
        replace z=l.z *(`r_z' *(1-l.z)-`beta_zy'*l`tau'.y) in `i'
    }
}

//keep in 300/450
set seed 12345678

timer clear 1
timer on 1

/* Typical Output */
edm explore x

edm explore x, theta(0.2(0.1)2.0) algorithm(smap)

//edm explore x, algorithm(smap)
//edm explore x, copredict(cop) copredictvar(z) algorithm(smap)
//edm xmap x y, algorithm(smap) savesmap(beta)
//gen x_copy = z
//gen y_copy = y
//edm xmap x y, copredict(xmap_y_p) copredictvar(x_copy y_copy) algorithm(smap)

timer off 1
/* Run time in seconds */
timer list 1

/* Plot the system */
//twoway (connected x t)(connected y t) in 1/40, ytitle(value) xtitle(t) legend(position(7) ring(0)) 
//graph export plot.png, replace
