# The logistic map (synthetic dataset)

TODO: Add comments / explanations to the following code snippets.

``` stata
/* Create a dynamic system */

set obs 500

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

keep in 300/450
```

``` stata
set seed 12345678

/* Typical Output */
edm explore x
edm xmap x y

/* Plot the system */
twoway (connected x t)(connected y t) in 1/40, ytitle(value) xtitle(t) legend(position(7) ring(0)) 
graph export plot.pdf, replace

/* rho-E plot */
edm explore y, e(2/10) rep(50)
mat r= e(explore_result)
svmat r, names(col)
twoway (scatter c3 c1)(lpoly c3 c1),xtitle("E") ytitle("{it:{&rho}}") legend(order(1 "{it:{&rho}}" 2 "local polynomial smoothing") col(1) position(8) ring(0))
graph export rho-e.pdf, replace
drop c*

/* rho-theta plot */
edm explore y, e(2) algorithm(smap) theta(0(0.01)5) k(-1) 
mat r= e(explore_result)
svmat r, names(col)
twoway (line c3 c2) , legend(order(1 "{it:{&rho}}") position(5) ring(0)) xtitle("{it:{&theta}}") ytitle("{it:{&rho}}") title("{it:{&rho}}-{it:{&theta}} of variable y")
graph export rho-theta.pdf, replace
drop c*

/* ccm */
edm xmap x y, e(2)

/* ccm-convergent plot */
edm xmap x y, e(2) rep(10) library(5/150)
mat c1= e(xmap_1)
mat c2= e(xmap_2)
svmat c1, names(xy)
svmat c2, names(yx)
label variable xy3 "y|M(x)"
label variable yx3 "x|M(y)"
twoway (scatter xy3 xy2, mfcolor(%30) mlcolor(%30)) ///
    (scatter yx3 yx2, mfcolor(%30) mlcolor(%30)) ///
    (lpoly xy3 xy2)(lpoly yx3 yx2), xtitle(L) ytitle("{it:{&rho}}") legend(col(2))

graph export rho-L.pdf, replace
```

``` stata
/* ccm placebo permutation test */
keep if t !=.
gen u = runiform()
sort u
gen t_u = _n
tsset t_u
drop xy* yx*
edm xmap x y, e(2) rep(10) library(5/150)
mat cxy= e(xmap_1)
mat cyx= e(xmap_2)
svmat cxy, names(placebo_xy)
svmat cyx, names(placebo_yx)
label variable placebo_xy3 "y|M(x)"
label variable placebo_yx3 "x|M(y)"
twoway (scatter placebo_xy3 placebo_xy2, mfcolor(%30) mlcolor(%30)) ///
    (scatter placebo_yx3 placebo_yx2, mfcolor(%30) mlcolor(%30)) ///
    (lpoly placebo_xy3 placebo_xy2)(lpoly placebo_yx3 placebo_yx2), xtitle(L) ytitle("{it:{&rho}}") legend(col(2))
tsset t
graph export rho-L-placebo.pdf, replace


/* jackknife */
qui jackknife: edm xmap x y, e(2) 
ereturn display

/* jackknife result testing */

drop if t==.
foreach e of numlist 2 20 {
    qui jackknife: edm xmap x y, e(`e') 
    ereturn display
    mat b =e(b)
    mat v =e(V)
    local xy_rho`e' = b[1,2]
    local xy_se`e' =  sqrt(v[2,2])
}

ztesti 1 `xy_rho2' `xy_se2' 1 `xy_rho20' `xy_se20'


drop if t==.
foreach l of numlist 10 140 {
    edm xmap x y, library(`l') rep(100)
    mat cyx= e(xmap_2)
    svmat cyx, names(lib`l'_yx)
}

ttest lib10_yx3 == lib140_yx3, unpaired unequal
```

``` stata
/* An example of the CI option */
edm xmap y x, library(10) rep(1000) ci(90) direction(oneway)
edm xmap y x, direction(oneway)

/* Time-delayed causality */
local pair1x "x"
local pair1y "y"
local pair2x "x"
local pair2y "z"
local pair3x "y"
local pair3y "z"

gen l =.

forvalues p = 1/3{
    local m = 1
    gen result`p' = .
    forvalues laglength=-3/3{
        if `laglength'<0 {
            gen m`m' = l`=abs(`laglength')'.`pair`p'x'
        }
        else {
            gen m`m' = f`=abs(`laglength')'.`pair`p'x'
        }
        
        edm xmap  `pair`p'y' m`m', e(2)
        mat a =e(b)
        replace l = `laglength' in `m'
        replace result`p' = a[1,1] in `m'
        local ++m
    }

    

    drop m* 
}
twoway (connected result1 l) (connected result2 l) (connected result3 l), xline(-2 -1) legend(on order(1 "x~x|M(y)" 2 "x~x|M(z)" 3 "y~y|M(z)") row(1)) xtitle("Time") ytitle("{it:{&rho}}")
graph export time-delayed.pdf, replace
```
