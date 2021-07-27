/* Example with a Real-World Dataset */
/* Visualising coupling strength using the Chicago crime dataset*/
import delimited "chicago.csv", clear
tsset t

global EDM_SAVE_INPUTS = "chicago"
cap rm chicago.json

// Start off running the commands with the Euclidean distance

set seed 1

timer clear
timer on 100

/* Explore the system dimension */
timer on 1
edm explore z.temp, e(2/30) crossfold(5) seed(1234567)
timer off 1
timer list

/* Explore convergance property */
timer on 2
edm xmap z.temp z.crime, e(7) library(10(5)200 210(10)1000 1020(20)2000 2050(50)4350 4365) rep(10)
timer off 2
timer list 

mat cyx= e(xmap_2)
mat cxy= e(xmap_1)
svmat cyx, names(chicago_yx)
svmat cxy, names(chicago_xy)
label variable chicago_xy3 "Crime|M(Temperature)"
label variable chicago_yx3 "Temperature|M(Crime)"
twoway (scatter chicago_xy3 chicago_xy2, mfcolor(%30) mlcolor(%30)) ///
    (scatter chicago_yx3 chicago_yx2, mfcolor(%30) mlcolor(%30)) ///
    (lpoly chicago_xy3 chicago_xy2)(lpoly chicago_yx3 chicago_yx2), xtitle(L) ytitle("{it:{&rho}}") legend(col(1))
    
graph export chicago-rho-L-euclidean.pdf, replace


/* save the betas using smap algorithms */
timer on 3
edm xmap z.temp z.crime, e(7) alg(smap) k(-1) savesmap(beta)
timer off 3
timer list

/* Plot the effects of temperature on crime */
twoway (kdensity beta1_b1_rep1), xtitle("Contemporaneous effect of temperature on crime") ytitle("Density")
graph export chicago-crime1-euclidean.pdf, replace
twoway (scatter beta1_b1_rep1 temp, xtitle("Temperature (Fahrenheit)") ytitle("Contemporaneous effect of temperature on crime") msize(small))(lpoly beta1_b1_rep1 temp), legend(on order( 1 "Local coefficient" 2 "Local polynomial smoothing"))
graph export chicago-crime2-euclidean.pdf, replace

timer off 100
timer list


// Run again with Wasserstein distance
global EDM_SAVE_INPUTS = "chicago-wasserstein"
cap rm chicago-wasserstein.json

clear
import delimited "chicago.csv", clear
tsset t

global EDM_DISTANCE="Wasserstein"

timer clear
timer on 100

timer on 1
edm explore z.temp, e(2/30) crossfold(5) seed(1234567)
timer off 1
timer list

/* Explore convergance property */
timer on 2
edm xmap z.temp z.crime, e(7) library(10(5)200 210(10)1000 1020(20)2000 2050(50)4350 4365) rep(2)
timer off 2
timer list 

mat cyx= e(xmap_2)
mat cxy= e(xmap_1)
svmat cyx, names(chicago_yx)
svmat cxy, names(chicago_xy)
label variable chicago_xy3 "Crime|M(Temperature)"
label variable chicago_yx3 "Temperature|M(Crime)"
twoway (scatter chicago_xy3 chicago_xy2, mfcolor(%30) mlcolor(%30)) ///
    (scatter chicago_yx3 chicago_yx2, mfcolor(%30) mlcolor(%30)) ///
    (lpoly chicago_xy3 chicago_xy2)(lpoly chicago_yx3 chicago_yx2), xtitle(L) ytitle("{it:{&rho}}") legend(col(1))
    
graph export chicago-rho-L-wasserstein.pdf, replace


/* save the betas using smap algorithms */
timer on 3
edm xmap z.temp z.crime, e(7) alg(smap) k(-1) savesmap(beta)
timer off 3
timer list

/* Plot the effects of temperature on crime */
twoway (kdensity beta1_b1_rep1), xtitle("Contemporaneous effect of temperature on crime") ytitle("Density")
graph export chicago-crime1-wasserstein.pdf, replace
twoway (scatter beta1_b1_rep1 temp, xtitle("Temperature (Fahrenheit)") ytitle("Contemporaneous effect of temperature on crime") msize(small))(lpoly beta1_b1_rep1 temp), legend(on order( 1 "Local coefficient" 2 "Local polynomial smoothing"))
graph export chicago-crime2-wasserstein.pdf, replace

timer off 100
timer list
