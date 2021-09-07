/* Example using panel data */
// Note, need 'dt' everywhere or the 'tsset' will try to
// expand the dataset way beyond the capacity of RAM
import delimited "affect.csv", clear
format delivnum %tc

xtset id delivnum

global EDM_SAVE_INPUTS = "affect"
cap rm affect.json


set seed 1

timer clear
timer on 100

/* Explore the system dimension */
timer on 1
edm explore z.pa, e(2/30) seed(1234567) dt idw(0)
timer off 1
timer list

timer on 2
edm explore z.pa, e(2/30) seed(1234567) dt idw(-1)
timer off 2
timer list


timer on 3
edm explore z.pa, e(2/30) seed(1234567) dt idw(0) reldt
timer off 3
timer list

timer on 4
edm explore z.pa, e(2/30) seed(1234567) dt idw(-1) reldt
timer off 4
timer list


/* Explore convergance property (multispatial) */
timer on 5
edm xmap z.pa z.na, e(10) library(20(10)200 200(250)13000) rep(10) dt idw(0)
timer off 5
timer list

mat cyx= e(xmap_2)
mat cxy= e(xmap_1)
svmat cyx, names(affect_yx)
svmat cxy, names(affect_xy)
label variable affect_xy3 "NA|M(PA)"
label variable affect_yx3 "PA|M(NA)"
twoway (scatter affect_xy3 affect_xy2, mfcolor(%30) mlcolor(%30)) ///
    (scatter affect_yx3 affect_yx2, mfcolor(%30) mlcolor(%30)) ///
    (lpoly affect_xy3 affect_xy2)(lpoly affect_yx3 affect_yx2), xtitle(L) ytitle("{it:{&rho}}") legend(col(1))
    
graph export affect-rho-L-euclidean.pdf, replace

drop affect*

/* Explore convergance property (no panel mixing) */
timer on 6
edm xmap z.pa z.na, e(10) library(20(10)200 200(250)13000) rep(10) dt idw(-1)
timer off 6
timer list

mat cyx= e(xmap_2)
mat cxy= e(xmap_1)
svmat cyx, names(affect_yx)
svmat cxy, names(affect_xy)
label variable affect_xy3 "NA|M(PA)"
label variable affect_yx3 "PA|M(NA)"
twoway (scatter affect_xy3 affect_xy2, mfcolor(%30) mlcolor(%30)) ///
    (scatter affect_yx3 affect_yx2, mfcolor(%30) mlcolor(%30)) ///
    (lpoly affect_xy3 affect_xy2)(lpoly affect_yx3 affect_yx2), xtitle(L) ytitle("{it:{&rho}}") legend(col(1))
    
graph export affect-rho-L-indep-panels-euclidean.pdf, replace

/* save the betas using smap algorithms */
timer on 7
edm xmap z.pa z.na, e(10) alg(smap) dt k(-1) savesmap(beta)
timer off 7
timer list

/* Plot the effects of temperature on crime */
twoway (kdensity beta1_b1_rep1), xtitle("Contemporaneous effect of PA on NA") ytitle("Density")
graph export affect-smap1-euclidean.pdf, replace
twoway (scatter beta1_b1_rep1 pa, xtitle("PA") ytitle("Contemporaneous effect of PA on NA") msize(small))(lpoly beta1_b1_rep1 pa), legend(on order( 1 "Local coefficient" 2 "Local polynomial smoothing"))
graph export affect-smap2-euclidean.pdf, replace



/* Explore the nonlinearity */
timer on 8
edm explore z.pa, e(10) alg(smap) dt k(-1) theta(0 1)
edm explore z.na, e(10) alg(smap) dt k(-1) theta(0 1)
timer off 8
timer list
