clear

set linesize 255

global EDM_VERBOSITY = 1
global EDM_NTHREADS = 4
global EDM_GPU = 0
global EDM_MATA = 0

if c(MP) {
    qui set processor 1
}

local obs 1755293
set obs `obs'

gen t = _n
tsset t

gen x = .
gen y = .

capture mata mata drop logistic_map()
mata:
mata set matastrict off
real matrix logistic_map(real scalar obs)
{
  r_x = 3.625
  r_y = 3.77
  beta_xy = 0.05
  beta_yx = 0.4
  tau = 1
  
  st_view(x, ., "x")
  st_view(y, ., "y")
  
  x[1] = 0.2
  y[1] = 0.4
  
  for (i = 2; i <= obs; i++) {
    x[i] = x[i-1] * (r_x * (1 - x[i-1]) - beta_xy * y[i-1])
    y[i] = y[i-1] * (r_y * (1 - y[i-1]) - beta_yx * x[i-tau])
  }
}
end

mata: logistic_map(`obs')

timer clear
timer on 1
edm explore x if _n < 200000
timer off 1
timer list

