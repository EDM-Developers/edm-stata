* cut-down test file
discard

set obs 500
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
keep in 300/500



* Determining the complexity of the system

* identify optimal E

edm explore x, e(2/10)

edm xmap x y, k(5)

edm explore x, k(5) crossfold(10)

edm explore x, theta(0.2(0.1)2.0) algorithm(smap)

edm xmap x y, theta(0.2) algorithm(smap) savesmap(beta)
assert beta1_b2_rep1 !=. if _n>1



edm xmap y x, predict(x2) direction(oneway)
assert x2 !=. if _n>1

edm explore x, copredict(teste) copredictvar(y)
assert teste!=. if _n>1

edm explore z.x, tp(10)

edm xmap y x, tp(10) direction(oneway)

edm xmap y x, tp(10) copredict(testx) copredictvar(x2 y) direction(oneway)
assert testx!=. if _n>2


edm xmap y x, tp(10) copredict(testx2) copredictvar(z.x2 z.y) direction(oneway)
assert testx2 !=. if _n>2

edm xmap y x, extra(u1) tp(10) copredict(testx3) copredictvar(z.x2 z.y) direction(oneway)
assert testx3 !=. if _n>2

* check explore / xmap consistency

edm xmap l.x x, direction(oneway)
mat xmap_r=e(b)
edm explore x, full
mat explore_r =e(b)
assert xmap_r[1,1] == explore_r[1,1]


* check xmap reverse consistency

edm xmap x y
mat xmap1 = e(b)
edm xmap y x
assert e(b)[1,1] == xmap1[1,2]
assert e(b)[1,2] == xmap1[1,1]

tempfile basedata
qui compress
save `basedata', replace
* test missing data
set seed 12345678
gen double u = runiform() 
drop if u<0.1
replace x = . if u<0.2
replace t=. if mod(t,19) ==1


edm explore x
edm explore x, dt

edm explore x, allowmissing
edm explore x, missingdistance(2)
edm xmap x l.x, allowmissing
edm xmap x l.x, missingdistance(2)


edm xmap x l.x, extraembed(u) allowmissing dt alg(smap) savesmap(newb) e(5)

edm xmap x l3.x, extraembed(u) allowmissing dt alg(smap) savesmap(newc) e(5) oneway dtsave(testdt)

edm explore x, extraembed(u) allowmissing dt crossfold(5)

edm explore d.x, dt


edm explore x, rep(20) ci(95)

edm xmap x y, lib(50) rep(20) ci(95)


cap drop x_copy 
cap drop x_p
cap drop xc_p
clonevar x_copy = x
edm explore x, predict(x_p) copredict(xc_p) copredictvar(x_copy) full
assert x_p!=. if _n!=1 & _n!=_N
assert xc_p!=. if _n!=1 & _n!=_N
assert x_p ==xc_p if x_p!=. 
cor x_p f.x
cor xc_p f.x

discard
cap drop x_p xc_p
edm explore x, predict(x_p) copredict(xc_p) copredictvar(x_copy) full tp(2)
sum x_p xc_p
assert xc_p!=. if x_p!=.


gen y_copy = y
edm xmap x y, tp(10) copredict(xmap_y_p) copredictvar(x_copy y_copy) direction(oneway) predict(xmap_y)
assert xmap_y_p !=. if _n>1
assert xmap_y_p == xmap_y if xmap_y !=.

preserve
drop if mod(t,17)==1
//copredict with dt
edm xmap x y, dt copredict(xmap_y_p_dt) copredictvar(x_copy y_copy) direction(oneway) predict(xmap_y_dt)
assert xmap_y_p_dt !=. if _n>1
assert xmap_y_p_dt == xmap_y_dt if xmap_y_dt !=.

edm explore x, predict(predicted_x_dt) copredict(predicted_x_copy_dt) copredictvar(x_copy) full

assert predicted_x_dt == predicted_x_copy_dt if predicted_x_dt!=.

restore


edm explore x, copredict(dx2) copredictvar(d.x)

edm explore x, predict(predicted_x) copredict(predicted_y_from_mx) copredictvar(y) full
edm explore y, predict(predicted_y) full
cor f.x f.y predicted_x predicted_y predicted_y_from_mx
cor f.x predicted_x 
assert r(rho)>0.99
cor f.y predicted_y 
assert r(rho)>0.99
cor f.y predicted_y_from_mx
assert r(rho)>0.7

* Obtaining CI of the xmap
jackknife: edm xmap x y, e(2)
ereturn display 

jackknife: edm explore x, e(2)
ereturn display 

