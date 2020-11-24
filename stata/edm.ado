*! version 1.4.1, 29Oct2020, Jinjing Li, Michael Zyphur, George Sugihara, Edoardo Tescari, Patrick Laub
*! conact: <jinjing.li@canberra.edu.au>

global EDM_VERSION = "1.4.1"
/* Empirical dynamic modelling

Version history:
* 29/10/2020: suppress some debug information and fix a bug on copredict with dt
* 13/10/2020: speed improvement on matrix copying
* 15/9/2020: add plugin support
* 15/8/2020: fix a bug in the default number of k in smap and labelling
* 29/5/2020: robustness improvement
* 26/5/2020: update ci() return format and add pc tile values
* 26/5/2020: return e(ub_rho1) e(lb_rho1) etc for the ci() estimates
* 19/5/2020: Update s-map / llr algorithm
* 13/5/2020: bug fix in coprediction
* 22/4/2020: allow copredict() together with dt
* 26/3/2020: remove omitted coefficients instead of reporting as 0
* 17/2/2020: robustness improvement when dt and allowmissing are specified at the same time
* 15/2/2020: improve robustness for estimations using data with missing identifier
* 4/2/2020: update smap coefficient label
* 27/1/2020: report actual E used, add "reportrawe" option
* 23/1/2020: bug fix to disallow constant in extras
* 16/1/2020: update default distance
* 16/1/2020: allow error recovery
* 9/1/2020: allow missing data to be included in the calculation
* 9/1/2020: allow automatic time difference in the embedding via dt option
* 26/11/2019: extra embedding option for both explore and xmap
* 17/11/2019: fix a bug where the missing values of the second variables are not detected in the data cleaning process
* 2/11/2019: add -force- option, improve error message for xmap
* 24/10/2019: speed improvement for duplicate values
* 24/10/2019: bug fix on develop update
* 28/9/2019: bug fixes in explore mode
* 10/7/2019: coprediction and many others including an improvement of nearest non-overlap neighbour search
* 15/6/2019: Many new features
* 12/3/2019: labelling of the savesmap

 */

/* global EDM_DEBUG = 0 */

program define edm, eclass
	version 14

	if replay() {
		if (`"`e(cmd)'"' != "edm") {
			noi di as error "results for edm not found"
			exit 301
		}
		edmDisplay `0'
		exit `rc'
	}
	else edmParser `0'
end

program define edmDisplayCI, rclass
	syntax ,  mat(name) ci(integer) [maxr(integer 2)]
	quietly {
		noi di as result %18s "Est. mean `ci'% CI" _c

		if `maxr' == 1 {
			noi di as result %17s " " _c
		}
		local datasize = r(N)
		tempname varbuffer
		/* noi mat list `mat' */
		svmat `mat',names(`varbuffer'_ci`ci'_)

		local type1 "rho"
		local type2 "mae"

		forvalues j=1/`maxr' {
			cap ci `varbuffer'_ci`ci'_`j', level(`ci')
			if _rc !=0 {
				cap ci means `varbuffer'_ci`ci'_`j', level(`ci')
			}
			return scalar lb_mean_`type`j'' = `=r(lb)'
			return scalar ub_mean_`type`j'' = `=r(ub)'

			noi di as result "   [" _c

			/* noi di as result  %9.5g  `=`matmean'[1,`j']-invnormal(1-(100-`ci')/200)*`matsd'[1,`j']/sqrt(`rep')' _c */
			noi di as result  %9.5g  `=r(lb)' _c
			noi di as result ", " _c

			/* noi di as result  %9.5g  `=`matmean'[1,`j']+invnormal(1-(100-`ci')/200)*`matsd'[1,`j']/sqrt(`rep')' _c */
			noi di as result  %9.5g  `=r(ub)' _c

			noi di as result " ]" _c


		}
		noi qui count
		local datasize = r(N)
		/* mat list `buffer' */
		/* set trace on */
		noi di ""
		noi di as result %18s "`=(100-`ci')/2'/`=100 - (100-`ci')/2' Pc (Est.)" _c
		forvalues j=1/`maxr' {

			if `maxr' == 1 {
				noi di as result %17s " " _c
			}
			qui sum `varbuffer'_ci`ci'_`j'
			noi di as result "   [" _c
			noi di as result %9.5g `=r(mean)-invnormal(1-(100-`ci')/200)*r(sd)' _c
			return scalar lb_pce_`type`j'' = `=r(mean)-invnormal(1-(100-`ci')/200)*r(sd)'
			noi di as result ", " _c
			noi di as result %9.5g `=r(mean)+invnormal(1-(100-`ci')/200)*r(sd)' _c
			return scalar ub_pce_`type`j'' = `=r(mean)+invnormal(1-(100-`ci')/200)*r(sd)'
			noi di as result " ]" _c
		}

		noi di ""
		noi di as result %18s "`=(100-`ci')/2'/`=100 - (100-`ci')/2' Pc (Obs.)" _c
		forvalues j=1/`maxr' {

			if `maxr' == 1 {
				noi di as result %17s " " _c
			}
			_pctile `varbuffer'_ci`ci'_`j', percentile(`=(100-`ci')/2' `=100 - (100-`ci')/2' )
			noi di as result "   [" _c
			noi di as result %9.5g `=r(r1)' _c
			return scalar lb_pco_`type`j'' = `=r(r1)'
			noi di as result ", " _c
			noi di as result %9.5g `=r(r2)' _c
			return scalar ub_pco_`type`j'' = `=r(r2)'
			noi di as result " ]" _c
			drop `varbuffer'_ci`ci'_`j'
		}

		cap drop `varbuffer'_ci`ci'_*
		qui keep if _n<=`datasize'
		noi di ""
	}

end

program define edmParser, eclass

	/* syntax anything(id="subcommand or variables" min=2 max=3)  [if], [e(numlist ascending)] [theta(numlist ascending)] [manifold(string)] [converge] */

	* identify subcommand
	local subcommand "`1'"
	if strpos("`subcommand'",",") !=0 {
		local subcommand =substr("`1'",1,strpos("`subcommand'",",")-1)
	}
	local subargs = substr("`0'", strlen("`subcommand'")+1,.)

	if "`subcommand'" == "update" {
		edmUpdate `subargs'
	}
	else if "`subcommand'" == "version" {
		edmVersion `subargs'
	}
	else {
		qui xtset
		local original_t = r(timevar)
		if "`=r(panelvar)'" == "." {
			local original_id = ""
		}
		else {
			local original_id = r(panelvar)
		}

		ereturn clear
		if inlist("`subcommand'","explore","xmap") {
			cap noi {
				if "`subcommand'" == "explore" {
					edmExplore `subargs'
				}
				else if "`subcommand'" == "xmap" {
					edmXmap `subargs'
				}
			}
			if _rc !=0 {
				local error_code = _rc
				cap xtset
				if _rc !=0 {
					if "`original_id'" != ""{
						qui xtset `original_id' `original_t'
					}
					else {
						qui tsset `original_t'
					}
				}
				cap error `error_code'
				/* di as text "r(`error_code');" */
				exit(`error_code')
			}

		}
		else {
			di as error `"Invalid subcommand. Use "explore" or "xmap""'
			error 1
		}
		ereturn local cmd "edm"
		ereturn local cmdline `"edm `0'"'
	}
end

program define edmUpdate
	syntax , [DEVELOPment] [replace]
	if "`development'" == "development" {
		di "Updating edm from the development channel"
		net install edm, from("https://jinjingli.github.io/edm/") `replace'
	}
	else {
		di "Updating edm from SSC"
		ssc install edm, `replace'
	}
	discard

end

program define edmVersion
	syntax , [test]
	dis "${EDM_VERSION}"
end


program define edmPluginCheck, rclass
	syntax , [mata]
	if "${EDM_MATA}" == "1" | "`mata'" =="mata" {
		return scalar mata_mode = 1

	}
	else {
		cap smap_block_mdap
		if _rc == 199 {
			di as text "Warning: Using slow (mata) edm implementation (failed to load the compiled plugin)"
		}
		return scalar mata_mode = _rc==199
	}
end

/*
program define edmCoremap, eclass
	syntax anything  [if], [e(integer 2)] [theta(real 1)] [k(integer 0)] [library(integer 0)] [seed(integer 0)] [ALGorithm(string)] [tau(integer 1)] [DETails] [Predict(name)] [tp(integer 1)] [COPredict(name)] [copredictvar(string)] [full] [force] [EXTRAembed(string)] [ALLOWMISSing] [MISSINGdistance(real 0)] trainset(string) predictset(string)
 */


program define edmExplore, eclass sortpreserve
	syntax anything  [if], [e(numlist ascending)] [theta(numlist ascending)] [k(integer 0)] [REPlicate(integer 1)] [seed(integer 0)] [ALGorithm(string)] [tau(integer 1)] [DETails] [Predict(name)] [CROSSfold(integer 0)] [CI(integer 0)] [tp(integer 1)] [COPredict(name)] [copredictvar(string)] [full] [force] [EXTRAembed(string)] [ALLOWMISSing] [MISSINGdistance(real 0)] [dt] [DTWeight(real 0)] [DTSave(name)] [reportrawe] [CODTWeight(real 0)] [dot(integer 1)] [mata] [nthreads(integer 0)] [saveinputs(string)] [verbosity(integer 0)]

	* set seed
	if `seed' != 0 {
		set seed `seed'

	}
	if `tp' <1 {
		di as error "tp must be greater than or equal to 1"
		error 9
	}
	* check predict
	if "`predict'" !="" {
		confirm new variable `predict'
	}

	if `crossfold' >0 {
		if `replicate' >1 {
			di as error "Replication must be not set if crossfold validation is used."
			error 119
		}
		if "`full'" == "full" {
			di as error "option full cannot be specified in combination with crossfold."
			error 119
		}
	}
	* default values

	if "`theta'" == ""{
		local theta = 1
	}

	* identify data structure
	qui xtset
	if "`=r(panelvar)'" != "." {
		local ispanel =1
		local panel_id = r(panelvar)
	}
	else {
		local ispanel =0
	}
	if !inlist("`algorithm'","smap","simplex","llr","") {
		dis as error "Not valid algorithm specification"
		error 121
	}
	if "`algorithm'" == "" {
		local algorithm "simplex"
	}
	qui xtset
	local tmax = r(tmax)
	local tmin = r(tmin)
	local tdelta = r(tdelta)
	local timevar "`=r(timevar)'"
	local total_t = int((r(tmax)-r(tmin))/r(tdelta)) + 1


	edmPluginCheck, `mata'
	/* return list */
	local mata_mode = r(mata_mode)
	if "${EDM_VERBOSITY}"!="" {
		local verbosity=${EDM_VERBOSITY}
	}
	if "${EDM_NTHREADS}"!="" {
		local nthreads=${EDM_NTHREADS}
	}

	* create manifold as variables
	tempvar x y
	tokenize "`anything'"

	local ori_x "`1'"
	local ori_y "`2'"
	if "`3'" != "" {
		error 103
	}
	if "`2'" == "" {
		local univariate 1
		local y ""
	}
	else {
		local univariate 0
	}
	if "`e'" =="" {
		local e = 3 - `univariate'
	}
	local report_actuale = "`reportrawe'" ==""
	marksample touse
	markout `touse' `timevar' `panel_id'

	forvalues i=1/2 {
		if `i' ==1 {
			local currentv = "x"
		}
		else {
			local currentv = "y"
		}
		if substr("``i''",1,2) == "z." {
			/* local `i' "`=substr("``i''",3,.)'" */
			/* tab `touse' */
			qui sum `=substr("``i''",3,.)' if `touse' ==1
			qui gen double ``currentv'' = (`=substr("``i''",3,.)' - r(mean))/r(sd)
			/* qui replace ``currentv'' = 0 if `touse' !=1 */
			/* count if `n`i'' !=.
			di "normalisation triggered." */
		}
		else {
			if "``i''" != "" {
				qui gen double ``currentv'' = ``i''
				/* qui gen double ``currentv'' = ``i'' if `touse' == 1 */
				/* qui replace ``currentv'' = 0 if `touse' !=1 */
			}

		}

	}
	local zcount = 0
	local zlist ""
	tempvar zusable
	edmExtractExtra `extraembed'
	/* noi return list */
	local parsed_extravars = strtrim(r(extravars))
	local parsed_dt = "`dt'" =="dt"
	local parsed_dtw = "`dtweight'"
	if "`dtsave'" != ""{
		confirm new variable `dtsave'
	}
	local parsed_dtsave = "`dtsave'"

	/* di "`parsed_dtsave'"
	di "parsed extra:`parsed_extravars':" */
	if `parsed_dt' == 1 {
		/* general algorithm for generating t patterns
		1. keep only touse
		2. drop all missings
		3. regenerate t
		4. generate oldt_pattern
		 */
		qui {
			preserve
			keep if `touse' ==1
			if !(`missingdistance' !=0 | "`allowmissing'"=="allowmissing") {
				keep if `x' != .
			}
			xtset
			local original_t = r(timevar)
			/* keep if `original_t' !=. */
			if "`=r(panelvar)'" == "." {
				local original_id = ""
				local byori =""
			}
			else {
				local original_id = r(panelvar)
				local byori ="by `original_id': "
				/* keep if `original_id' !=. */
			}
			tempvar newt
			sort `original_id' `original_t'
			`byori' gen `newt' = _n
			if "`original_id'" != ""{
				xtset `original_id' `newt'
			}
			else {
				tsset `newt'
			}

			tempvar dt_value
			qui gen double `dt_value' = d.`original_t'
			/* sum `dt_value' */
			keep `original_id' `original_t' `newt' `dt_value'
			/* assert `original_id' !=. & `original_t' !=. */
			tempfile updatedt_main
			save `updatedt_main'
			restore

			// update copredict mainfold
			if "`copredictvar'" != "" {

				preserve
				keep if `touse' ==1
				// this part is for filtering only
				if !(`missingdistance' !=0 | "`allowmissing'"=="allowmissing") {
					tokenize "`copredictvar'"
					local co_x "`1'"
					local co_y "`2'"
					foreach v in "x" "y" {
						tempvar co_`v'_new
						if substr("`co_`v''",1,2) =="z." {
							gen `co_`v'_new' = `=substr("`co_`v''",3,.)' if `touse' ==1
						}
						else {
							if "`co_`v''" !="" {
								gen `co_`v'_new' = `co_`v'' if `touse' ==1
							}
							else {
								continue
							}

						}
						/* noi di "`co_x'"
						noi sum `co_x' `co_x_new' */

						keep if `co_`v'_new' !=.
					}
				}

				tempvar newt_co
				sort `original_id' `original_t'
				`byori' gen `newt_co' = _n
				if "`original_id'" != ""{
					qui xtset `original_id' `newt_co'
				}
				else {
					qui tsset `newt_co'
				}

				tempvar dt_value_co
				gen double `dt_value_co' = d.`original_t'
				/* sum `dt_value_co' */

				keep `original_id' `original_t' `newt_co' `dt_value_co'
				tempfile updatedt_co
				save `updatedt_co'
				restore
			}

			merge m:1 `original_id' `original_t' using `updatedt_main', assert(master match) nogen
			if "`copredictvar'" != "" {
				merge m:1 `original_id' `original_t' using `updatedt_co', assert(master match) nogen
			}

			/* di "merged" */
			/* sum `original_t' `newt' */
			sort `original_id' `newt'
			if "`original_id'" != ""{
				qui xtset `original_id' `newt'
			}
			else {
				qui tsset `newt'
			}
			if !inlist("`parsed_dtsave'","",".") {
				clonevar `parsed_dtsave' = `dt_value'
				qui label variable `parsed_dtsave' "Time delta (`original_t')"
			}
		}

	}
	qui {
		gen byte `zusable' = `touse' ==1
		foreach v of local parsed_extravars {
			tempvar z`++zcount'
			/* noi di "extra embeding: `v'" */
			if substr("`v'",1,2) == "z." {

				sum `=substr("`v'",3,.)' if `touse' ==1
				gen double `z`zcount'' = (`=substr("`v'",3,.)' - r(mean))/r(sd)

				replace `zusable' = 0 if `z`zcount'' ==.
				/* replace `z`zcount'' = 0 if `touse' != 1 */
			}
			else {
				if real("`v'") !=. {
					noi di as error "`v' is not a variable name"
					error 198
				}
				gen double `z`zcount'' = `v' if `touse' ==1
				replace `zusable' = 0 if `z`zcount'' ==.
			}

			local zlist "`zlist' `z`zcount''"
		}
	}
	/* sum `zusable' */




	tempvar usable

	/*
	include missing algorithm:
	1. more relaxed usable
	2. deal within mata

	*/
	qui gen byte `usable' = `x'!=. & `touse' & `zusable' ==1
	/* noi sum `touse' */
	/* noi sum `usable' */
	if "`2'" != "" {
		qui replace `usable' =0  if `usable' ==1 &  `y' ==.
	}
	if (`missingdistance' !=0 | "`allowmissing'"=="allowmissing") {
		/* di "Reset usable due to allow missing" */
		qui replace `usable' = `touse'
	}

	local max_e = -1
	local min_e = .
	foreach i of numlist `e' {
		if `i'>`max_e' {
			local max_e = `i'
		}
		if `i' <`min_e' {
			local min_e = `i'
		}
	}
	if (`max_e' < 3 - `univariate') | (`min_e' < 3 - `univariate') {
		dis as error "Some of the proposed number of dimensions for embedding is too small."
		error 121
	}
	local esize = `max_e' - (1- `univariate')

	local mapping_0 "`x' `y' `zlist'"
	if `parsed_dt' == 1 {
		if `parsed_dtw' ==0 {
			qui sum `x' if `usable' ==1
			local xsd = r(sd)
			qui sum `dt_value' if `usable' ==1
			local tsd = r(sd)
			local parsed_dtw = `xsd'/`tsd'
			if `tsd' ==0 {
				// if there is no variance, no sampling required
				local parsed_dtw = 0
				local parsed_dt = 0
			}
		}
	}
	* If allow missing, use a wide definition of usable when generating manifold
	if (`missingdistance' !=0 | "`allowmissing'"=="allowmissing") {
		qui replace `usable' = `touse'
	}
	forvalues i=1/`=`esize'-1' {
		tempvar x_`i'
		qui gen double `x_`i'' = l`=`i'*`tau''.`x' if `usable' ==1
		qui replace `usable' = 0 if `x_`i'' ==.
		local mapping_`i' "`mapping_`=`i'-1'' `x_`i''"
		if `parsed_dt' ==1 {
			tempvar t_`i'
			// note: embeding does not include the status itself, it includes the gap between current obs with the last obs
			/* sum `usable' */
			/* di `dt_value'
			di `parsed_dtw' */
			/* sum `dt_value' */
			/* di "dt descript"
			sum `dt_value' l`=`i'-1'.`dt_value' if `usable' ==1 */
			qui gen double `t_`i'' = l`=`i'-1'.`dt_value'* `parsed_dtw' if `usable' ==1
			local mapping_`i' "`mapping_`i'' `t_`i''"
			/* di "mapping_`i': `mapping_`i''" */
			/* di "incorporate lag `i' in dt mapping" */
			/* sum l`=`i'-1'.`dt_value' `t_`i'' */
		}

	}
	qui replace `usable' = 0 if f`future_step'.`x' ==. & `usable' ==1
	/* di "`mapping_`=`esize'-1''" */
	/* sum `mapping_`=`esize'-1'' */

	if "`copredictvar'" != "" {
		if "`copredict'" == "" {
			di as error "The copredict() option is not specified"
			error 111
		}
		// temporary move to newt_co
		if `parsed_dt' == 1 {
			if "`original_id'" != ""{
				qui xtset `original_id' `newt_co'
			}
			else {
				qui tsset `newt_co'
			}
		}

		confirm new variable `copredict'
		tempvar co_train_set co_predict_set co_usable
		gen byte `co_train_set' = `usable' ==1
		gen byte `co_usable' = `touse' ==1
		* build prediction manifold
		tokenize "`copredictvar'"
		local co_x "`1'"
		local co_y "`2'"
		/* set trace on */
		foreach v in "x" "y" {
			if substr("`co_`v''",1,2) =="z." {
				tempvar co_`v'_new
				qui sum `=substr("`co_`v''",3,.)' if `touse' ==1
				qui gen double `co_`v'_new' = (`=substr("`co_`v''",3,.)' - r(mean))/r(sd)

				local co_`v' `co_`v'_new'
			}
			else if strpos("`co_`v''",".") >0 {
				tempvar co_`v'_new
				qui gen double `co_`v'_new' = `co_`v'' if `touse' == 1
				local co_`v' `co_`v'_new'
			}
		}
		/* set trace off */

		if (`univariate' == 1 & "`co_y'" != "") |  (`univariate' == 0 & "`co_y'" == "") {
			di as error "Coprediction does not match the main manifold construct"
			error 111
		}

		* z list
		tempvar co_zusable
		qui gen byte `co_zusable' = `touse' ==1
		local co_zlist_name ""
		local co_zlist ""
		local co_zcount = 0
		qui {
			foreach v of local parsed_extravars {
				tempvar z`++co_zcount'
				/* noi di "extra embeding: `v'" */
				if substr("`v'",1,2) == "z." {
					sum `=substr("`v'",3,.)' if `touse' ==1
					gen double `z`co_zcount'' = (`=substr("`v'",3,.)' - r(mean))/r(sd)
					replace `co_zusable' = 0 if `z`co_zcount'' ==.
					/* replace `z`co_zcount'' = 0 if `touse' != 1 */
				}
				else {
					if real("`v'") !=. {
						noi di as error "`v' is not a variable name"
						error 198
					}
					gen double `z`co_zcount'' = `v' if `touse' ==1
					replace `co_zusable' = 0 if `z`co_zcount'' ==.
				}
				local co_zlist_name "`co_zlist_name' `v'"
				local co_zlist "`co_zlist' `z`co_zcount''"
			}
		}

		local co_mapping_0 "`co_x' `co_y' `co_zlist'"
		qui replace `co_usable' = 0 if `co_x'==.
		if "`co_y'" != "" {
			qui replace `co_usable' = 0 if `co_y'==.
		}


		forvalues i=1/`=`esize'-1' {
			tempvar co_x_`i'
			qui gen double `co_x_`i'' = l`=`i'*`tau''.`co_x' if `co_usable' ==1
			qui replace `co_usable' = 0 if `co_x_`i'' ==.
			local co_mapping_`i' "`co_mapping_`=`i'-1'' `co_x_`i''"
			local co_mapping "`co_mapping_`i''"
			/* di "i=`i', x= `co_x_`i'', co_mapping "
			list `co_usable' */
			/* di "co_mapping_`i' w/o dt: `co_mapping_`i''" */
			if `parsed_dt' ==1 {
				if `codtweight' ==0 {
					local codtweight = `parsed_dtw'
				}

				tempvar t_`i'
				// note: embeding does not include the status itself, it includes the gap between current obs with the last obs
				/* di "dt descript"
				sum `dt_value' l`=`i'-1'.`dt_value' if `usable' ==1 */

				// parsed _dtw should match copredict
				qui gen double `t_`i'' = l`=`i'-1'.`dt_value_co'* `codtweight' if `co_usable' ==1
				local co_mapping_`i' "`co_mapping_`i'' `t_`i''"

				/* di "incorporate lag `i' in dt mapping"
				sum l`=`i'-1'.`dt_value' `t_`i''
				di "`mapping_`i'_name'" */
				local co_mapping "`co_mapping_`i''"
				/* di "co_mapping_`i' w dt: `co_mapping_`i''" */

			}
		}

		/* forvalues i=1/`=`esize'-1' {
			di "final map`i' : `mapping_`i''"
			di "final comap`i' : `co_mapping_`i''"
		} */

		/* gen co_usable = `co_usable' */
		gen byte `co_predict_set' = `co_usable' ==1

		//restore t
		if `parsed_dt' == 1 {
			if "`original_id'" != ""{
				qui xtset `original_id' `newt'
			}
			else {
				qui tsset `newt'
			}
		}
	}

	tempvar x_f x_p train_set predict_set
	local future_step = `tp'-1 + `tau' //predict the future value with an offset defined by tp
	qui gen double `x_f' = f`future_step'.`x' if `usable' ==1
	qui replace `usable' =0 if `x_f' ==.
	qui gen double `x_p' = .

	* split time by half - one for training and one for validation
	/* local train_start_t = `tmin'
	local train_finish_t = int((`tmin' + `tmax')/2)
	local validation_start_t = `train_finish_t' + 1
	local validation_finish_t = `tmax'


	gen byte `train_set' = `timevar'>= `train_start_t' & `timevar'<=`train_finish_t' & `usable' ==1
	gen byte `predict_set' = `timevar'>=`validation_start_t' & `timevar'<=`validation_finish_t' & `usable' ==1 */

	/* sum `usable' */
	if (`missingdistance' !=0 | "`allowmissing'"=="allowmissing") {
		qui {
			/* di "Reset usable due to allow missing" */
			replace `usable' = 0
			foreach v of local mapping_`=`esize'-1' {
				replace `usable' =1 if `v' !=. & `touse' ==1
			}
			if `missingdistance' <=0 {
				qui sum `x' if `usable' ==1
				local missingdistance = 2/sqrt(c(pi))*r(sd)
			}
			replace `x_f' = f`future_step'.`x' if `usable' ==1
			replace `usable' = 0 if `x_f' ==.
			/* di "missing distance: `missingdistance'" */
		}

	}
	/* sum `usable' `mapping_`=`esize'-1'' */


	tempvar u mae
	mat r = J(1,4,.)
	local round = max(`crossfold', `replicate')
	if `crossfold' >0 {
		qui count if `usable' ==1
		if `crossfold' > r(N)/ `esize' {
			di as error "Not enough observations for cross-validations"
			error 149
		}
		tempvar crossfoldu crossfoldunum
		qui gen double `crossfoldu' = runiform() if `usable' ==1
		qui egen `crossfoldunum'= rank(`crossfoldu'), unique
		/* sum `crossfoldunum' */
	}
	local no_of_runs = 0
	tempvar overlap
	if `round' > 1 & `dot' >0 {
		if  `replicate' > 1 {
			di "Replication progress (`replicate' in total)"
		}
		else if `crossfold' >1 {
			di "`crossfold'-fold cross-validation progress (`crossfold' in total)"
		}

		local finished_rep = 0
	}


	numlist "`e'"
	local e_size = wordcount("`=r(numlist)'")
	numlist "`theta'"
	local theta_size = wordcount("`=r(numlist)'")

	mat r = J(`=`round'*`theta_size'*`e_size'',4,.)

	forvalues t=1/`round' {
		/* timer on 1 */
		qui {
			cap drop `train_set' `predict_set' `overlap'

			if `crossfold' > 0 {
				gen byte `train_set' = mod(`crossfoldunum',`crossfold') != (`t' -1) & `usable' ==1
				gen byte `predict_set' = mod(`crossfoldunum',`crossfold') == (`t' -1) & `usable' ==1
				/* noi sum `train_set' `predict_set' */
			}
			else {
				* replicate mode
				if "`full'" == "full" {
					gen byte `train_set' = `usable' ==1
					gen byte `predict_set' = `train_set'
				}
				else {
					gen double `u' = runiform() if `usable' ==1
					sum `u',d
					gen byte `train_set' = `u' <r(p50) & `u' !=.
					gen byte `predict_set' = `u' >=r(p50) & `u' !=.
					drop `u'
				}

			}

			gen byte `overlap' = (`train_set' ==`predict_set') & (`predict_set' ==1)
			if "`full'" != "full" {
				assert `overlap' == 0 if `predict_set' ==1
			}




			count if `train_set' ==1
			local train_size = r(N)
			count if `predict_set' ==1
			local max_lib_size = min(`train_size',r(N))
			if `max_lib_size' < 1 {
				noi display as error "Invalid dimension or library specifications"
				error 9
			}
		}


		foreach i of numlist `e' {
			local manifold "mapping_`=`i'-2+`univariate''"
			local e_offset = wordcount("`mapping_`=`i'-2+`univariate'''") - `i'
			/* sum `mapping_`i'' */
			foreach j of numlist `theta' {
				/* noi sum ``manifold'' `x_f' `x_p' t if `train_set' ==1
				noi sum ``manifold'' `x_f' `x_p' t if `predict_set' ==1 */
				/* list ``manifold'' `x_f' `x_p' t if `train_set' ==1
				list ``manifold'' `x_f' `x_p' t if `predict_set' ==1 */
				if `k'> 0{
					local lib_size = min(`k',`train_size')
				}
				else if `k' == 0{
					local lib_size = `i' +`zcount' + (`parsed_dt' ==1) + cond("`algorithm'" =="smap",2,1)
				}
				else {
					local lib_size = `max_lib_size'
					/* di "full lib" */
				}
				if `lib_size' > `max_lib_size' {
					local lib_size = `max_lib_size'

				}
				if `k' != 0 {
					local cmdfootnote = "Note: Number of neighbours (k) is adjusted to `lib_size'" + char(10)
				}
				else if `k' != `lib_size' & `k' == 0 {
					local plus_amt = `zcount' + (`parsed_dt' ==1) + cond("`algorithm'" =="smap",2,1)
					local cmdfootnote = "Note: Number of neighbours (k) is set to E+`plus_amt'" + char(10)
				}
				local vars_save ""
				/* if ${EDM_DEBUG} {
					sum `overlap' ``manifold'' `x_f' `x_p' `train_set' `predict_set'  if `train_set' ==1
					sum ``manifold'' `x_f' if `train_set' ==1
					sum `x_p' if `predict_set' ==1
					di "`missingdistance'"
				} */

				/* ==== CODE FOR C PLUGIN ==== */


				/* di _rc */
				/* timer off 1
				timer on 2 */
				if `mata_mode' == 1 {
					// di "Mata Mode"
					mata: smap_block("``manifold''", "", "`x_f'", "`x_p'","`train_set'","`predict_set'",`j',`lib_size',"`overlap'", "`algorithm'", "`vars_save'","`force'", `missingdistance')
				}
				else {
					// di "Plugin Mode"
					if "`savesmap'"!="" & ("`algorithm'"=="smap"|"`algorithm'"=="llr") {
						local vsave_flag = 1
						// display "vsave_flag: " `vsave_flag'
						unab vars : `vars_save'
						local varssv `: word count `vars''
					}
					else {
						local vsave_flag = 0
						loc varssv=0
						// display "vsave_flag: " `vsave_flag'
					}

					local myvars ``manifold'' `x_f' `x_p' `train_set' `predict_set' `overlap' `vars_save'

					unab vars : ``manifold''
					local mani `: word count `vars''

					qui {
						ds ``manifold''
						local label_mani : word 1 of `r(varlist)'
						levelsof `label_mani'
					}
					local level_mani = r(r)
					//display "Number of distinct values in manifold: " `level_mani'

					if (`level_mani' <= 2) {
						local mani_metric "binary"
					}
					else {
						local mani_metric "euclidean"
					}
					//display "Manifold metric: `mani_metric'"

					local pmani_flag = 0
					//display "pmani_flag: " `pmani_flag'

					local pmani_metric = "`mani_metric'"
					//display "P_manifold metric: `pmani_metric'"

					plugin call smap_block_mdap `myvars', `j' `lib_size' "`algorithm'" "`force'" `missingdistance' `mani' `pmani_flag' `vsave_flag' `varssv' `nthreads' `verbosity' "`mani_metric'" "`pmani_metric'" `saveinputs'
				}


				/* ==== END CODE FOR C PLUGIN ==== */


				/* sum `x_f' `x_p' */

				qui gen double `mae' = abs( `x_p' - `x_f' )  if `predict_set' == 1
				qui sum `mae', meanonly
				local rmae = r(mean)
				drop `mae'
				local current_e =`i' + cond(`report_actuale'==1,`e_offset',0)
				/* noi sum */


				qui corr `x_f' `x_p' if `predict_set' == 1

				if "`predict'" != "" {
					cap gen double `predict' = `x_p'
					qui label variable `predict' "edm prediction result"
					cap replace `predict' = `x_p' if `x_p' !=.
				}

				/* noi di "E = `current_e', theta = `j': Correlation = `=r(rho)', MAE = `rmae'" */
				/* timer off 2
				timer on 3 */
				local ++no_of_runs
				mat r[`no_of_runs',1] = `current_e'
				mat r[`no_of_runs',2] = `j'
				mat r[`no_of_runs',3] = r(rho)
				mat r[`no_of_runs',4] = `rmae'
				/* timer off 3 */
				/* ereturn local rho = `=r(rho)'
				ereturn local mae = `rmae' */

			}

		}


		if `round' > 1 & `dot' >0 {
			local ++finished_rep
			if mod(`finished_rep',50*`dot') == 0 {
				di as text ". `finished_rep'"
			}
			else if mod(`finished_rep',`dot') == 0{
				di as text "." _c
			}
		}
		/* timer off 1
		timer off 2

		cap timer list
		qui replace timer_1 =r(t1) in `t'
		qui replace timer_2=r(t2) in `t'
		qui replace timer_3=r(t3) in `t'
		timer clear 1
		timer clear 2
		timer clear 3 */
	}
	if `round' > 1 & `dot' >0 {
		if mod(`finished_rep',50*`dot') != 0 {
			di ""
		}
	}
	if "`copredictvar'" !=""  {
		if `no_of_runs' ==1{
			/* di "overlap"
			sum `overlap' */
			qui replace `overlap' = 0
			qui replace `co_train_set' = 0 if `usable' ==0
			/* qui count if `co_train_set' ==1 */

			/* sum `usable' `co_train_set' `train_set'
			sum `overlap' ``manifold'' `x_f' `x_p' `co_train_set' `co_predict_set'  if `co_train_set' ==1
			sum `co_mapping' `co_predict_set' */
			/*
			sum `co_mapping'  if `co_train_set' ==1
			sum `co_train_set' ``manifold'' `x_f' `x_p' `co_train_set' `co_predict_set'  if `co_train_set' ==1 */
			/* list `co_mapping'
			list `co_usable'
			list  ID RowNr iPA iNA `usable' `co_train_set' `co_predict_set' */
			tempvar co_x_p
			qui gen double `co_x_p'=.

			if `mata_mode' == 1 {
				// di "Mata Mode"
				mata: smap_block("``manifold''", "`co_mapping'", "`x_f'", "`co_x_p'","`co_train_set'","`co_predict_set'",`theta',`lib_size',"`overlap'", "`algorithm'", "","`force'",`missingdistance')
			}
			else {
				// di "Plugin Mode"
				local myvars ``manifold'' `x_f' `co_x_p' `co_train_set' `co_predict_set' `overlap' `co_mapping' `vars_save'

				unab vars : ``manifold''
				local mani `: word count `vars''

				qui {
					ds ``manifold''
					local label_mani : word 1 of `r(varlist)'
					levelsof `label_mani'
				}
				local level_mani = r(r)
				//display "Number of distinct values in manifold: " `level_mani'

				if (`level_mani' <= 2) {
					local mani_metric "binary"
				}
				else {
					local mani_metric "euclidean"
				}
				//display "Manifold metric: `mani_metric'"

				unab vars : `co_mapping'
				local pmani `: word count `vars''

				qui {
					ds `co_mapping'
                    local label_pmani : word 1 of `r(varlist)'
					levelsof `label_pmani'
				}
				local level_pmani = r(r)
				//display "Number of distinct values in p_manifold: " `level_pmani'

				if (`level_pmani' <= 2) {
					local pmani_metric "binary"
				}
				else {
					local pmani_metric "euclidean"
				}
				//display "P-manifold metric: `pmani_metric'"

				local pmani_flag = 1
				/* display "pmani_flag: " `pmani_flag' */

				local vsave_flag = 0
				/* display "vsave_flag: " `vsave_flag' */

				plugin call smap_block_mdap `myvars', `theta' `lib_size' "`algorithm'" "`force'" `missingdistance' `mani' `pmani_flag' `vsave_flag' `pmani' `nthreads' `verbosity' "`mani_metric'" "`pmani_metric'" `saveinputs'
			}


			qui gen double `copredict' = `co_x_p'
			qui label variable `copredict' "edm copredicted  `copredictvar' using manifold `ori_x' `ori_y'"
		}
		else {
			di as error "Error: coprediction can only run with one specified manifold construct (no repetition etc.)" _c
			di as result ""
		}

	}

	/* mat r = r[2...,.] */

	mat cfull = r[1,3]
	local cfullname= subinstr("`ori_x'",".","/",.)
	matrix colnames cfull = `cfullname'
	matrix rownames cfull = rho
	/* mat list cfull */
	qui count if `usable' ==1
	scalar total_obs = r(N)
	ereturn post cfull, esample(`usable')
	ereturn scalar N = total_obs
	/* ereturn post r, esample(`usable') dep("`y'") properties("r") */
	ereturn local subcommand = "explore"
	ereturn local direction = "oneway"
	ereturn scalar univariate_main = `univariate'
	ereturn scalar e_offset = `e_offset'
	ereturn scalar report_actuale = `report_actuale'
	ereturn local x "`ori_x'"
	ereturn local y "`ori_y'"
	if `crossfold' >0 {
		ereturn local cmdfootnote "`cmdfootnote'Note: `crossfold'-fold cross validation results reported"
	}
	else {
		if "`full'" == "full" {
			ereturn local cmdfootnote "`cmdfootnote'Note: Full sample used for the computation"
		}
		else {
			ereturn local cmdfootnote "`cmdfootnote'Note: Random 50/50 split for training and validation data"
		}

	}
	ereturn matrix explore_result  = r
	ereturn local algorithm "`algorithm'"
	ereturn scalar tau = `tau'
	ereturn scalar replicate = `replicate'
	ereturn scalar crossfold = `crossfold'
	ereturn scalar rep_details = "`details'" == "details"
	ereturn scalar ci = `ci'
	ereturn local copredict ="`copredict'"
	ereturn local copredictvar ="`copredictvar'"
	ereturn scalar force_compute = "`force'" =="force"
	ereturn scalar panel =`ispanel'
	ereturn scalar dt =`parsed_dt'
	if (`missingdistance' !=0 | "`allowmissing'"=="allowmissing") {
		ereturn scalar missingdistance = `missingdistance'
	}
	if `parsed_dt' {
		ereturn scalar dtw =`parsed_dtw'
		ereturn local dtsave "`parsed_dtsave'"
		if "`parsed_extravars'" != "" {
			ereturn local extraembed = "`parsed_extravars' (+ time delta)"
		}
		else {
			ereturn local extraembed = "(time delta)"
		}

	}
	else {
		ereturn local extraembed = "`parsed_extravars'"
	}
	if "`dt'" =="dt" {
		sort `original_id' `original_t'
		qui xtset `original_id' `original_t'
		if "`original_id'" != ""{
			qui xtset `original_id' `original_t'
		}
		else {
			qui tsset `original_t'
		}
		if `parsed_dt' ==0 {
			ereturn local cmdfootnote "`cmdfootnote'Note: dt option is ignored due to lack of variations in time delta"
		}
	}
	ereturn local mode = cond(`mata_mode' ==1, "mata","plugin")
	edmDisplay

end


program define edmXmap, eclass sortpreserve
	syntax anything  [if], [e(integer 2)] [theta(real 1)] [Library(numlist)] [seed(integer 0)] [k(integer 0)] [ALGorithm(string)] [tau(integer 1)] [REPlicate(integer 1)] [SAVEsmap(string)] [DETails] [DIrection(string)] [Predict(name)] [CI(integer 0)] [tp(integer 0)] [COPredict(name)] [copredictvar(string)] [force] [EXTRAembed(string)] [ALLOWMISSing] [MISSINGdistance(real 0)] [dt] [DTWeight(real 0)] [DTSave(name)] [oneway] [savemanifold(name)] [CODTWeight(real 0)] [dot(integer 1)] [mata] [nthreads(integer 0)] [saveinputs(string)] [verbosity(integer 0)]
	* set seed
	if `seed' != 0 {
		set seed `seed'
	}
	if `tp' <0 {
		di as error "tp must be greater than or equal to 0"
		error 9
	}

	if "`oneway'" =="oneway" {
		if !inlist("`direction'","oneway","") {
			di as error "option oneway does not match direction() option"
			error 9
		}
		else {
			local direction "oneway"
		}
	}
	if "`direction'" != "oneway" & "`dtsave'" !="" {
		di as error "dtsave() option can only be used together with oneway"
		error 9
	}
	* check prediction save
	if "`predict'" !="" {
		confirm new variable `predict'
		if "`direction'" != "oneway" {
			dis as error "direction() option must be set to oneway if predicted values are to be saved."
			error 197
		}
	}


	* default values
	if "`e'" =="" {
		local e = "2"
	}
	if "`theta'" == ""{
		local theta = 1
	}
	local l_ori "`library'"
	if "`library'" == "" {
		local l = 0
	}
	else {
		local l  "`library'"
	}
	if !inlist("`algorithm'","smap","simplex","llr","") {
		dis as error "Not valid algorithm specification"
		error 121
	}
	if "`algorithm'" == "" {
		local algorithm "simplex"
	}
	else if ("`algorithm'" == "smap"|"`algorithm'" == "llr") {
		if "`savesmap'" != "" {
			cap sum `savesmap'*
			if _rc != 111 {
				dis as error "There should be no variable with existing prefix when savesmap() option is used"
				error 110
			}
		}
	}
	if "`savesmap'" != "" & !("`algorithm'" =="smap" | "`algorithm'" =="llr") {
		dis as error "savesmap() option should only be specified with S-map"
		error 119
	}
	if "`direction'" == ""  {
		local direction "both"
	}
	if !inlist("`direction'","both","oneway") {
		dis as error "direction() option should be either both or oneway"
		error 197
	}
	* identify data structure
	qui xtset
	if "`=r(panelvar)'" != "." {
		local ispanel =1
		local panel_id = r(panelvar)
	}
	else {
		local ispanel =0
	}
	qui xtset
	local tmax = r(tmax)
	local tmin = r(tmin)
	local tdelta = r(tdelta)
	local timevar "`=r(timevar)'"
	local total_t = (r(tmax)-r(tmin))/r(tdelta) + 1

	marksample touse
	markout `touse' `timevar' `panel_id'
	sort `panel_id' `timevar'


	edmPluginCheck, `mata'
	local mata_mode = r(mata_mode)
	if "${EDM_VERBOSITY}"!="" {
		local verbosity=${EDM_VERBOSITY}
	}
	if "${EDM_NTHREADS}"!="" {
		local nthreads=${EDM_NTHREADS}
	}

	* create manifold as variables
	tokenize "`anything'"

	local ori_x "`1'"
	local ori_y "`2'"
	if "`3'" != "" {
		error 103
	}

	if "`1'" =="" | "`2'" == "" {
		error 102
	}

	tempvar x y
	forvalues i=1/2 {
		if `i' ==1 {
			local currentv = "x"
		}
		else {
			local currentv = "y"
		}
		if substr("``i''",1,2) == "z." {
			/* local `i' "`=substr("``i''",3,.)'" */
			/* tab `touse' */
			qui sum `=substr("``i''",3,.)' if `touse' ==1
			qui gen double ``currentv'' = (`=substr("``i''",3,.)' - r(mean))/r(sd)
			/* qui replace ``currentv'' = 0 if `touse' != 1 */
			/* count if `n`i'' !=.
			di "normalisation triggered." */
		}
		else {
			if "``i''" != "" {
				qui gen double ``currentv'' = ``i''
				/* qui gen double ``currentv'' = ``i'' if `touse' == 1 */
				// the line below is needed for jackknife to work. Apparently no missing values can be in this line for unknown reason
				/* qui replace ``currentv'' = 0 if `touse' != 1 */

			}

		}

	}

	if (`e' < 1) {
		dis as error "Some of the proposed number of dimensions for embedding is too small."
		error 121
	}

	local esize = `e'
	local comap_constructed = 0

	mat r1 = J(1,4,.)
	mat r2 = J(1,4,.)
	local max_round = ("`direction'" == "both") + 1

	forvalues round=1/`max_round'{
		if `round' ==2 {
			local swap "`x'"
			local x "`y'"
			local y "`swap'"
		}

		local zcount = 0
		local zlist ""
		tempvar zusable
		edmExtractExtra `extraembed'
		/* return list */
		local parsed_extravars = strtrim(r(extravars))
		local parsed_dt = "`dt'" =="dt"
		local parsed_dtw = "`dtweight'"
		if "`dtsave'" != ""{
			confirm new variable `dtsave'
		}
		local parsed_dtsave = "`dtsave'"
		/* di "`parsed_dtsave'"
		di "parsed extra:`parsed_extravars':" */
		if `parsed_dt' == 1 {
			/* general algorithm for generating t patterns
			1. keep only touse
			2. drop all missings
			3. regenerate t
			4. generate oldt_pattern
			*/
			qui {

				// update main mainfold
				preserve
				keep if `touse' ==1
				if !(`missingdistance' !=0 | "`allowmissing'"=="allowmissing") {
					keep if `x' != .
				}
				qui xtset
				local original_t = r(timevar)
				if "`=r(panelvar)'" == "." {
					local original_id = ""
					local byori =""
				}
				else {
					local original_id = r(panelvar)
					local byori ="by `original_id': "
				}
				tempvar newt
				sort `original_id' `original_t'
				`byori' gen `newt' = _n
				if "`original_id'" != ""{
					qui xtset `original_id' `newt'
				}
				else {
					qui tsset `newt'
				}

				tempvar dt_value
				gen double `dt_value' = d.`original_t'
				/* sum `dt_value' */
				keep `original_id' `original_t' `newt' `dt_value'
				tempfile updatedt_main
				save `updatedt_main'
				restore

				// update copredict mainfold
				if "`copredictvar'" != "" {
					preserve
					keep if `touse' ==1
					if !(`missingdistance' !=0 | "`allowmissing'"=="allowmissing") {
						tokenize "`copredictvar'"
						local co_x "`1'"
						local co_y "`2'"
						foreach v in "x" "y" {
							tempvar co_`v'_new
							if substr("`co_`v''",1,2) =="z." {
								gen `co_`v'_new' = `=substr("`co_`v''",3,.)' if `touse' ==1
							}
							else {
								if "`co_`v''" !="" {
									gen `co_`v'_new' = `co_`v'' if `touse' ==1
								}
								else {
									continue
								}
							}
							keep if `co_`v'_new' !=.
						}
					}
					tempvar newt_co
					sort `original_id' `original_t'
					`byori' gen `newt_co' = _n
					if "`original_id'" != ""{
						qui xtset `original_id' `newt_co'
					}
					else {
						qui tsset `newt_co'
					}

					tempvar dt_value_co
					gen double `dt_value_co' = d.`original_t'
					/* sum `dt_value_co' */
					keep `original_id' `original_t' `newt_co' `dt_value_co'
					tempfile updatedt_co
					save `updatedt_co'
					restore
				}

				merge m:1 `original_id' `original_t' using `updatedt_main', assert(master match) nogen
				if "`copredictvar'" != "" {
					merge m:1 `original_id' `original_t' using `updatedt_co', assert(master match) nogen
				}
				/* tempvar mergevar
				merge m:1 `original_id' `original_t' using `updatedt_main', assert(master match) gen(`mergevar')

				noi {

					tab `mergevar'

					if "`original_id'" !="" {
						assert `mergevar' ==3 if  `original_t'!=. & `original_id'!=. & `touse' & `x' !=.
					}
					else {
						assert `mergevar' ==3 if  `original_t'!=. & `touse'  & `x' !=.
					}

				}

				drop `mergevar'
				*/
				/* merge m:1 `original_id' `original_t' using `updatedt_main', assert(master match)  */
				/* tab _merge */


				sum `original_t' `newt'
				sort `original_id' `newt'
				if "`original_id'" != ""{
					qui xtset `original_id' `newt'
				}
				else {
					qui tsset `newt'
				}
				if !inlist("`parsed_dtsave'","",".") {
					clonevar `parsed_dtsave' = `dt_value'
					qui label variable `parsed_dtsave' "Time delta (`original_t')"
				}
			}

		}

		qui gen byte `zusable' = `touse' ==1
		local zlist_name ""
		local zlist ""
		qui {
			foreach v of local parsed_extravars {
				tempvar z`++zcount'
				/* noi di "extra embeding: `v'" */
				if substr("`v'",1,2) == "z." {

					sum `=substr("`v'",3,.)' if `touse' ==1
					gen double `z`zcount'' = (`=substr("`v'",3,.)' - r(mean))/r(sd)
					replace `zusable' = 0 if `z`zcount'' ==.
					/* replace `z`zcount'' = 0 if `touse' != 1 */
				}
				else {
					if real("`v'") !=. {
						noi di as error "`v' is not a variable name"
						error 198
					}
					gen double `z`zcount'' = `v' if `touse' ==1
					replace `zusable' = 0 if `z`zcount'' ==.
				}
				local zlist_name "`zlist_name' `v'"
				local zlist "`zlist' `z`zcount''"
			}
		}

		/* for additional variables */
		/* if "`addition'" !="" {
			tempvar addition
			qui gen double `added_x' = `addition'
		} */

		/* local x = "`1'"
		local y = "`2'" */

		tempvar usable

		qui gen byte `usable' = `x'!=. & `touse' & f`tp'.`y' !=. & `zusable' == 1

		* mapping include variables and specified multivariates
		local mapping_0 "`x' `zlist'"
		local mapping_0_name "`=cond(`round'==1,"`ori_x'","`ori_y'")' `zlist_name'"
		/* di "`mapping_0_name'" */
		qui {
			if (`missingdistance' !=0 | "`allowmissing'"=="allowmissing") {
				/* di "Reset usable due to allow missing" */
				qui replace `usable' = `touse'
				/* sum `mapping_0' if `usable' */
			}
		}
		/* di "usable"
		sum `usable' */
		if `parsed_dt' == 1 {
			if `parsed_dtw' ==0 {
				qui sum `x' if `usable' ==1
				local xsd = r(sd)
				qui sum `dt_value' if `usable' ==1
				local tsd = r(sd)
				local parsed_dtw = `xsd'/`tsd'
				if `tsd' ==0 {
					// if there is no variance, no sampling required
					/* noi di "No sampling variation detected, dt option ignored" */
					local parsed_dtw = 0
					local parsed_dt = 0
				}
			}
			local parsed_dtw`round' = `parsed_dtw'
		}
		* If allow missing, use a wide definition of usable when generating manifold
		if (`missingdistance' !=0 | "`allowmissing'"=="allowmissing") {
			qui replace `usable' = `touse'
		}
		forvalues i=1/`=`esize'-1' {
			tempvar x_`i'
			/* sum `x' `usable'
			sum `x' if `usable' ==1 */

			qui gen double `x_`i'' = l`=`i'*`tau''.`x' if `usable' ==1
			/* sum `x_`i'' */
			/* pause on
			pause check */
			/* assert `x_`i'' !=0 */
			if !(`missingdistance' !=0 | "`allowmissing'"=="allowmissing") {
				qui replace `usable' = 0 if `x_`i'' ==.
			}

			local mapping_`i' "`mapping_`=`i'-1'' `x_`i''"
			local mapping_`i'_name "`mapping_`=`i'-1'_name' l`=`i'*`tau''.`=cond(`round'==1,"`ori_x'","`ori_y'")'"
			if `parsed_dt' ==1 {
				tempvar t_`i'
				// note: embeding does not include the status itself, it includes the gap between current obs with the last obs
				/* di "dt descript"
				sum `dt_value' l`=`i'-1'.`dt_value' if `usable' ==1 */
				qui gen double `t_`i'' = l`=`i'-1'.`dt_value'* `parsed_dtw' if `usable' ==1
				local mapping_`i' "`mapping_`i'' `t_`i''"
				local mapping_`i'_name "`mapping_`i'_name' dt`i'"
				/* di "incorporate lag `i' in dt mapping"
				sum l`=`i'-1'.`dt_value' `t_`i''
				di "`mapping_`i'_name'" */

			}
		}

		qui {
			if ((`missingdistance' !=0) | ("`allowmissing'"=="allowmissing")) {
				/* di "Reset usable due to allow missing" */
				replace `usable' = 0
				foreach v of local mapping_`=`esize'-1' {
					replace `usable' =1 if `v' !=. & `touse' ==1
				}
				if `missingdistance' <=0 {
					sum `x' if `usable' ==1
					local missingdistance = 2/sqrt(c(pi))*r(sd)
				}
				replace `usable' = 0 if f`tp'.`y' ==.
				di "missing distance: `missingdistance'"
				local missingdistance`round' = `missingdistance'
			}
			/* sum `usable' `mapping_`=`esize'-1'' */
		}


		if ("`copredictvar'" != "") & (`comap_constructed' ==0) {

			// temporary move to newt_co
			if `parsed_dt' == 1 {
				qui {
					if "`original_id'" != ""{
						qui xtset `original_id' `newt_co'
					}
					else {
						qui tsset `newt_co'
					}
				}
			}
			confirm new variable `copredict'
			tempvar co_train_set co_predict_set co_usable
			gen byte `co_train_set' = `usable' ==1
			gen byte `co_usable' = `touse' ==1
			* build prediction manifold
			tokenize "`copredictvar'"
			local co_x "`1'"
			local co_y "`2'"
			foreach v in "x" "y" {
				if substr("`co_`v''",1,2) =="z." {
					tempvar co_`v'_new
					qui sum `=substr("`co_`v''",3,.)' if `touse' ==1
					qui gen double `co_`v'_new' = (`=substr("`co_`v''",3,.)' - r(mean))/r(sd)
					local co_`v' `co_`v'_new'
				}
				else if strpos("`co_`v''",".") >0 {
					tempvar co_`v'_new
					qui gen double `co_`v'_new' = `co_`v'' if `touse' == 1
					local co_`v' `co_`v'_new'
				}
			}
			if ("`co_y'" == "") |  ("`co_x'" == "") {
				di as error "Coprediction does not match the main manifold construct"
				error 111
			}

			* z list
			tempvar co_zusable
			qui gen byte `co_zusable' = `touse' ==1
			local co_zlist_name ""
			local co_zlist ""
			local co_zcount = 0
			qui {
				foreach v of local parsed_extravars {
					tempvar z`++co_zcount'
					/* noi di "extra embeding: `v'" */
					if substr("`v'",1,2) == "z." {

						sum `=substr("`v'",3,.)' if `touse' ==1
						gen double `z`co_zcount'' = (`=substr("`v'",3,.)' - r(mean))/r(sd)
						replace `co_zusable' = 0 if `z`co_zcount'' ==.
						/* replace `z`co_zcount'' = 0 if `touse' != 1 */
					}
					else {
						if real("`v'") !=. {
							noi di as error "`v' is not a variable name"
							error 198
						}
						gen double `z`co_zcount'' = `v' if `touse' ==1
						replace `co_zusable' = 0 if `z`co_zcount'' ==.
					}
					local co_zlist_name "`co_zlist_name' `v'"
					local co_zlist "`co_zlist' `z`co_zcount''"
				}
			}

			* manifold of coprediction
			local co_mapping_0 "`co_x' `co_zlist'"
			qui replace `co_usable' = 0 if `co_x'==.
			forvalues i=1/`=`esize'-1' {
				tempvar co_x_`i'
				qui gen double `co_x_`i'' = l`=`i'*`tau''.`co_x' if `co_usable' ==1
				qui replace `co_usable' = 0 if `co_x_`i'' ==.
				local co_mapping_`i' "`co_mapping_`=`i'-1'' `co_x_`i''"
				local co_mapping "`co_mapping_`i''"
				if `parsed_dt' ==1 {
					if `codtweight' ==0 {
						// note: there are issues in recalculating the codtweight as the variable usable are not generated in the same way as cousable
						/* qui sum `co_x' if `co_usable' ==1
						local xsd = r(sd)
						qui sum `dt_value_co' if `co_usable' ==1
						local tsd = r(sd)
						local codtweight = `xsd'/`tsd'
						if `tsd' ==0 {
							// if there is no variance, no sampling required
							local codtweight = 0
						} */
						local codtweight = `parsed_dtw'
					}

					tempvar t_`i'
					// note: embeding does not include the status itself, it includes the gap between current obs with the last obs
					/* di "dt descript"
					sum `dt_value' l`=`i'-1'.`dt_value' if `usable' ==1 */

					// parsed _dtw should match copredict
					qui gen double `t_`i'' = l`=`i'-1'.`dt_value_co'* `codtweight' if `co_usable' ==1
					local co_mapping_`i' "`co_mapping_`i'' `t_`i''"

					/* di "incorporate lag `i' in dt mapping"
					sum l`=`i'-1'.`dt_value' `t_`i''
					di "`mapping_`i'_name'" */
					local co_mapping "`co_mapping_`i''"

				}
			}

			gen byte `co_predict_set' = `co_usable' ==1
			local comap_constructed =1


			//restore t
			if `parsed_dt' == 1 {
				qui {
					if "`original_id'" != ""{
						xtset `original_id' `newt'
					}
					else {
						tsset `newt'
					}
				}
			}

		}

		tempvar x_f x_p train_set predict_set

		qui gen double `x_f' = f`tp'.`y' if `usable' ==1
		qui gen double `x_p' = .

		qui gen byte `predict_set' = `usable'
		qui gen byte `train_set' = . // to be decided by library length

		tempvar u urank
		tempvar overlap


		local no_of_runs = 0
		if `replicate' > 1 & `round' == 1 & `dot' >0 {
			di "Replication progress (`=`replicate'*`max_round'' in total)"
			local finished_rep = 0
		}

		numlist "`e'"
		local e_size = wordcount("`=r(numlist)'")
		numlist "`theta'"
		local theta_size = wordcount("`=r(numlist)'")
		numlist "`l'"
		local l_size = wordcount("`=r(numlist)'")

		mat r`round' = J(`=`replicate'*`theta_size'*`e_size'*`l_size'',4,.)


		forvalues rep =1/`replicate' {
			cap drop `u' `urank'
			qui gen double `u' = runiform() if `usable' ==1
			qui egen `urank' =rank(`u') if `usable' ==1, unique
			qui count if `usable' ==1
			local urank_max = r(N)

			if "`l_ori'" =="0" | "`l_ori'"=="" {
				local l = `urank_max'
				/* di "l=`l', umax =`urank_max'" */
			}
			foreach i of numlist `e' {
				local manifold "mapping_`=`i'-1'"
				/* local manifold "mapping_`i'" */
				/* sum `mapping_`i'' */
				foreach j of numlist `theta' {
					/* sum ``manifold'' `x_f' `x_p' `train_set' `predict_set' */
					foreach lib_size of numlist `l' {
						/* di "l=`l', umax =`urank_max'" */
						/* sum ``manifold'' */
						if `lib_size'>`urank_max'{
							di as error "Library size exceeds the limit."
							error 1
							/* di "max triggered lib size `lib_size' , rankmax `urank_max'" */
							continue, break
						}
						else if `lib_size' <= `i' + 1 {
							/* noi display "Skipped library size `lib_size'" */
							di as error "Cannot estimate under the current library specification"
							error 1
						}

						qui replace `train_set' = `urank' <=`lib_size' & `usable' ==1
						qui count if `train_set' ==1
						local train_size = r(N)
						// detect k size
						if `k'> 0{
							local k_size = min(`k',`train_size' -1)
						}
						else if `k' == 0{
							local k_size = `i' +`zcount' + (`parsed_dt' ==1) + cond("`algorithm'" =="smap",2,1)
						}
						else if `k' < 0  {
							local k_size = `train_size' -1
							/* di "full lib" */
						}

						if `k' != 0 {
							local cmdfootnote = "Note: Number of neighbours (k) is adjusted to `k_size'" + char(10)
						}
						else if `k' != `k_size' & `k' == 0 {
							/* local cmdfootnote = "Note: Number of neighbours (k) is set to E+1" + char(10) */
						}

						local vars_save ""
						if "`savesmap'" != "" & ("`algorithm'" =="smap" | "`algorithm'" =="llr") {
							/* forvalues em=1/`=`esize'-1' {
								di "`mapping_`em'_name'"
								di "`mapping_`em''"
							} */

							local ii =0
							qui gen double `savesmap'`round'_b0_rep`rep' = .
							qui label variable `savesmap'`round'_b0_rep`rep' "constant in `=cond(`round'==1,"`ori_x'","`ori_y'")' predicting `=cond(`round'==1,"`ori_y'","`ori_x'")' S-map equation (rep `rep')"
							local vars_save "`vars_save' `savesmap'`round'_b0_rep`rep'"
							foreach name of local mapping_`=`esize'-1'_name {
								qui gen double `savesmap'`round'_b`++ii'_rep`rep' = .
								qui label variable `savesmap'`round'_b`ii'_rep`rep' "`name' predicting `=cond(`round'==1,"`ori_y'","`ori_x'")' or `=cond(`round'==1,"`ori_y'","`ori_x'")'|M(`=cond(`round'==1,"`ori_x'","`ori_y'")') S-map coefficient (rep `rep')"
								local vars_save "`vars_save' `savesmap'`round'_b`ii'_rep`rep'"
							}

							/* di "`vars_save'" */
						}
						/* di "``manifold''"
						di "``manifold'_name'" */
						qui gen byte `overlap' = `train_set' ==`predict_set' if `predict_set' ==1
						local last_theta =  `j'

						/* if ${EDM_DEBUG} {
							sum `overlap' ``manifold'' `x_f' `x_p' `train_set' `predict_set'  if `train_set' ==1
							sum ``manifold'' `x_f' if `train_set' ==1
							sum `x_p' if `predict_set' ==1
							di "`missingdistance'"
						} */

						if "`savemanifold'" !="" {
							local counter = 1
							foreach v of varlist ``manifold'' {
								cap gen double `savemanifold'`round'_`counter' = `v'
								if _rc!=0 {
									di as error "Cannot save the manifold using variable `savemanifold'`round'_`counter' - is the prefix used already?"
									exit(100)
								}
								local ++counter
							}


						}

						/* ==== CODE FOR C PLUGIN ==== */

						if `mata_mode' == 1 {
							// di "Mata Mode"
							mata: smap_block("``manifold''","", "`x_f'", "`x_p'","`train_set'","`predict_set'",`j',`k_size', "`overlap'", "`algorithm'","`vars_save'","`force'",`missingdistance')
						}
						else {
							// di "Plugin Mode"
							if "`savesmap'"!="" & ("`algorithm'"=="smap"|"`algorithm'"=="llr") {
								local vsave_flag = 1
								// display "vsave_flag: " `vsave_flag'
								unab vars : `vars_save'
								local varssv `: word count `vars''
							}
							else {
								local vsave_flag = 0
								loc varssv=0
								// display "vsave_flag: " `vsave_flag'
							}

							local myvars ``manifold'' `x_f' `x_p' `train_set' `predict_set' `overlap' `vars_save'

							unab vars : ``manifold''
							local mani `: word count `vars''

							qui {
								ds ``manifold''
								local label_mani : word 1 of `r(varlist)'
								levelsof `label_mani'
							}
							local level_mani = r(r)
							//display "Number of distinct values in manifold: " `level_mani'

							if (`level_mani' <= 2) {
								local mani_metric "binary"
							}
							else {
								local mani_metric "euclidean"
							}
							//display "Manifold metric: `mani_metric'"

							local pmani_flag = 0
							//display "pmani_flag: " `pmani_flag'

							local pmani_metric = "`mani_metric'"
							//display "P_manifold metric: `pmani_metric'"

							plugin call smap_block_mdap `myvars', `j' `k_size' "`algorithm'" "`force'" `missingdistance' `mani' `pmani_flag' `vsave_flag' `varssv' `nthreads' `verbosity' "`mani_metric'" "`pmani_metric'" `saveinputs'
						}

						/* ==== END CODE FOR C PLUGIN ==== */
						/* assert `x_p' !=. if `predict_set'==1
						assert `x_f' !=. if `predict_set'==1 */
						/* sum `x_f' `x_p' */
						tempvar mae
						qui gen double `mae' = abs( `x_p' - `x_f' ) if `predict_set' == 1
						qui sum `mae'
						local rmae = r(mean)
						drop `mae'
						local current_e =`i'
						/* scatter `x_f' `x_p' */
						qui corr `x_f' `x_p' if `predict_set' == 1
						if "`predict'" != "" {
							cap gen double `predict' = `x_p'
							qui label variable `predict' "edm prediction result"
							cap replace `predict' = `x_p' if `x_p' !=.
						}
						/* di "E = `current_e', theta = `j': Correlation = `=r(rho)'" */
						local ++no_of_runs
						mat r`round'[`no_of_runs',1] = `round'
						mat r`round'[`no_of_runs',2] = `lib_size'
						mat r`round'[`no_of_runs',3] = r(rho)
						mat r`round'[`no_of_runs',4] = `rmae'
						drop `overlap'

					}


				}
			}
			if `replicate' > 1 & `dot' >0 {
				local ++finished_rep
				if mod(`finished_rep',50*`dot') == 0 {
					di as text ". `finished_rep'"
				}
				else if mod(`finished_rep',`dot') == 0{
					di as text "." _c
				}
			}

		}


		* reset the panel structure

		if "`dt'" =="dt" {
			sort `original_id' `original_t'
			qui xtset `original_id' `original_t'
			if "`original_id'" != ""{
				qui xtset `original_id' `original_t'
			}
			else {
				qui tsset `original_t'
			}
			if `parsed_dt' ==0 {
				if "`direction'" == "oneway" {
					local cmdfootnote "`cmdfootnote'Note: dt option is ignored due to lack of variations in time delta"
				}
				else {
					local cmdfootnote "Note: dt option is ignored in at least one direction"
				}

			}
		}


		/* mat list r */
		/* mat r`round' = r`round'[2...,.] */
		/* mat list r`round' */

	}
	if `replicate' > 1 & `dot' >0 {
		if mod(`finished_rep',50*`dot') != 0 {
			di ""
		}
	}
	if "`copredictvar'" !="" {
		if `no_of_runs' ==1{
			qui gen byte `overlap' = 0
			qui replace `co_train_set' = 0 if `usable' ==0
			tempvar co_x_p
			qui gen double `co_x_p'=.


			//check whether dt transformation is required for copredict?
			// extract t for copredict variables -> add to copredict extras
			// set to new id t for mainfold construction


			/* sum ``manifold'' `co_mapping'
			sum `co_train_set' `co_predict_set'
			di "co_usable: `co_usable'"
			di "usable: `usable'"
			pause on
			pause test */

			/* di "start mata" */

			/* ==== CODE FOR C PLUGIN ==== */

			if `mata_mode' == 1 {
				// di "Mata Mode"
				mata: smap_block("``manifold''","`co_mapping'", "`x_f'", "`co_x_p'","`co_train_set'","`co_predict_set'",`last_theta',`k_size', "`overlap'", "`algorithm'","","`force'",`missingdistance')
			}
			else {
				// di "Plugin Mode"
				local myvars ``manifold'' `x_f' `co_x_p' `co_train_set' `co_predict_set' `overlap' `co_mapping' `vars_save'
				unab vars : ``manifold''
				local mani `: word count `vars''
				
				qui {
					ds ``manifold''
					local label_mani : word 1 of `r(varlist)'
					levelsof `label_mani'
				}
				local level_mani = r(r)
				//display "Number of distinct values in manifold: " `level_mani'

				if (`level_mani' <= 2) {
					local mani_metric "binary"
				}
				else {
					local mani_metric "euclidean"
				}
				//display "Manifold metric: `mani_metric'"
				
				unab vars : `co_mapping'
				local pmani `: word count `vars''
				
				qui {
					ds `co_mapping'
                    local label_pmani : word 1 of `r(varlist)'
					levelsof `label_pmani'
				}
				local level_pmani = r(r)
				//display "Number of distinct values in p_manifold: " `level_pmani'

				if (`level_pmani' <= 2) {
					local pmani_metric "binary"
				}
				else {
					local pmani_metric "euclidean"
				}
				//display "P-manifold metric: `pmani_metric'"
				
				local pmani_flag = 1
				// display "pmani_flag: " `pmani_flag'
				local vsave_flag = 0
				// display "vsave_flag: " `vsave_flag'
				plugin call smap_block_mdap `myvars', `last_theta' `k_size' "`algorithm'" "`force'" `missingdistance' `mani' `pmani_flag' `vsave_flag' `pmani' `nthreads' `verbosity' "`mani_metric'" "`pmani_metric'" `saveinputs'
			}

			/* ==== END CODE FOR C PLUGIN ==== */



			/* di "done back to do" */
			qui gen double `copredict' = `co_x_p'
			qui label variable `copredict' "edm copredicted  `copredictvar' using manifold `ori_x' `ori_y'"
		}
		else {
			di as error "Error: coprediction can only run with one specified manifold construct (no repetition etc.)" _c
			di as result ""
		}
	}
	/* mat list r1
	mat list r2 */

	mat cfull = (r1[1,3],r2[1,3])

	/* local cfullname = subinstr("`ori_y'|M(`ori_x') `ori_x'|M(`ori_y')",".","/",.) */
	local name1 = subinstr("`ori_y'|M(`ori_x')",".","/",.)
	local name2 = subinstr("`ori_x'|M(`ori_y')",".","/",.)
	local shortened = 1
	forvalues n =1/2 {
		if strlen("`name`n''") > 32 {
			local name`n' = substr("`name`n''",1,29) + "~`shortened'"
			local ++shortened
		}
	}
	matrix colnames cfull = `name1' `name2'
	matrix rownames cfull = rho
	/* mat list cfull */
	qui count if `usable' ==1
	scalar total_obs = r(N)
	/* mat list cfull */
	if "`direction'" == "oneway" {
		mat cfull = cfull[1...,1]
	}
	/* mat list cfull */
	ereturn post cfull, esample(`usable')
	ereturn scalar N = total_obs
	ereturn local subcommand = "xmap"
	ereturn matrix xmap_1  = r1
	if "`direction'" == "both" {
		ereturn matrix xmap_2  = r2
	}
	// the actual size of e should be main e + dt + extras
	ereturn scalar e_main = `e'
	ereturn scalar e_actual = wordcount("``manifold''")
	ereturn scalar e_offset = wordcount("``manifold''") - `e'
	ereturn scalar theta = `theta'
	ereturn local x "`ori_x'"
	ereturn local y "`ori_y'"
	ereturn local algorithm "`algorithm'"
	ereturn local cmdfootnote "`cmdfootnote'"
	ereturn scalar tau = `tau'
	ereturn scalar replicate = `replicate'
	ereturn scalar rep_details = "`details'" == "details"
	ereturn local direction = "`direction'"
	ereturn scalar ci = `ci'
	ereturn local copredict ="`copredict'"
	ereturn local copredictvar ="`copredictvar'"
	ereturn scalar force_compute = "`force'" =="force"
	ereturn local extraembed = "`extraembed'"
	ereturn scalar panel =`ispanel'
	ereturn scalar dt =`parsed_dt'
	if (`missingdistance' !=0 | "`allowmissing'"=="allowmissing") {
		ereturn scalar missingdistance = `missingdistance1'
		ereturn scalar missingdistance1 = `missingdistance1'
		if "`direction'" == "both" {
			ereturn scalar missingdistance2 = `missingdistance2'
		}
	}
	if `parsed_dt' {
		ereturn scalar dtw =`parsed_dtw1'
		ereturn scalar dtw1 =`parsed_dtw1'
		if "`direction'" == "both" {
			ereturn scalar dtw2 =`parsed_dtw2'
		}
		ereturn local dtsave "`parsed_dtsave'"
		if "`parsed_extravars'" != "" {
			ereturn local extraembed = "`parsed_extravars' (+ time delta)"
		}
		else {
			ereturn local extraembed = "(time delta)"
		}

	}
	else {
		ereturn local extraembed = "`parsed_extravars'"
	}


	ereturn local mode = cond(`mata_mode' ==1, "mata","plugin")
	edmDisplay
end


program define edmDisplay, eclass
/*


Emperical Dynamic Model
Univariate simplex projection with manifold construct x and its lag values
------------------------------------
| E | theta | rho |
| 2 | 0.1 | 0.993 |
------------------------------------
*/

	display _n "Empirical Dynamic Modelling"
	local diopts "`options'"
	local fmt "%12.5g"
	local fmtprop "%8.3f"
	local ci_counter = 1
	if e(subcommand) =="explore" {
		if e(univariate_main) ==1 {
			if !inlist("`=e(extraembed)'","",".") {
				di as text "Multivariate mapping with `=e(x)' and its lag values"
			}
			else {
				di as text "Univariate mapping with `=e(x)' and its lag values"
			}
		}
		else {
			di as text "Multivariate mapping with `=e(x)', its lag values, and `=e(y)'"
		}
		if !inlist("`=e(extraembed)'","",".") {
			di as text "Additional variable" _c
			di cond(wordcount("`=e(extraembed)'")>1,"s","") _c
			di " in the embedding: `=e(extraembed)'"
		}
		if e(missingdistance)>0 & e(missingdistance)!= .{
			di as text "Missing values are assumed to have a distance of " _c
			di `:di %8.2g `=e(missingdistance)'' _c
			di " with all values."
		}
		if ((e(replicate) == 1 & e(crossfold) <=0) | e(rep_details) == 1) {
			di as txt "{hline 68}"
			display as text %18s cond(e(report_actuale)==1,"Actual E","E")  _c
			display as text %16s "theta"  _c
			display as text %16s "rho"  _c
			display as text %16s "MAE"
			di as txt "{hline 68}"

			mat r = e(explore_result)
			local nr = rowsof(r)
			local kr = colsof(r)
			forvalues i = 1/ `nr' {
				forvalues j=1/`kr' {
					if `j'==1 {
						local dformat "%18s"
					}
					else {
						local dformat "%16s"
					}
					display as result `dformat' `"`:display `fmt' r[`i',`j'] '"' _c
				}
				display " "
			}
			di as txt "{hline 68}"
		}
		else {

			di as txt "{hline 70}"
			di as text %22s " " _c
			di as txt "{hline 9} rho {hline 9}  " _c
			di as txt "{hline 9} MAE {hline 9}"
			/* di as txt "{hline 70}" */
			display as text %9s cond(e(report_actuale)==1,"Actual E","E")  _c
			display as text %9s "theta"  _c
			display as text %13s "Mean"  _c
			display as text %13s "Std. Dev."  _c
			display as text %13s "Mean"  _c
			display as text %13s "Std. Dev."
			di as txt "{hline 70}"
			local dformat "%13s"

			// process the return matrix
			tempname reported_r r buffer summary_r
			mat `r' = e(explore_result)
			local nr = rowsof(`r')
			local kr = colsof(`r')
			mat `reported_r' = J(`nr',1,0)
			mat `summary_r' = J(1,6,.)
			/* mat list `r'
			mat list `reported_r' */
			forvalues i = 1/ `nr' {
				mat `buffer' = J(1,2,.)
				if `reported_r'[`i',1] ==1 {
					continue
				}
				local base_E = `r'[`i',1]
				local base_theta = `r'[`i',2]
				forvalues j=1/`nr' {
					if `reported_r'[`j',1] ==0 {
						if `r'[`j',1] == `base_E' & `r'[`j',2] == `base_theta' {
							mat `buffer' = (`buffer'\ `=`r'[`j',3]',`=`r'[`j',4]')
							mat `reported_r'[`j',1] =1
							/* mat list `buffer' */
						}
					}
				}
				/* noi mat list `buffer' */
				// now get the mean and st
				tempname mat_mean mat_sd
				mata: st_matrix("`mat_sd'", diagonal(sqrt(variance(st_matrix("`buffer'"))))')
				/* if changes to standard error */
				/* mata: st_matrix("`mat_sd'", diagonal(sqrt(variance(st_matrix("`buffer'"))))/sqrt(`nr')') */
				mata: st_matrix("`mat_mean'", mean(st_matrix("`buffer'")))

				di as result %9s  `"`: display %9.0g `r'[`i',1] '"' _c
				di as result %9s  `"`: display %9.5g `r'[`i',2] '"' _c
				forvalues j=1/2{
					display as result `dformat' `"`:display `fmt' `mat_mean'[1,`j'] '"' _c
					display as result `dformat' `"`:display `fmt' `mat_sd'[1,`j'] '"' _c
				}
				mat `summary_r' = (`summary_r'\ `=`r'[`i',1]',`=`r'[`i',2]', `=`mat_mean'[1,1]',`=`mat_sd'[1,1]', `=`mat_mean'[1,2]',`=`mat_sd'[1,2]')
				/* mat list `summary_r' */
				di ""
				if `=e(ci)'>0 & `=e(ci)'<100 {
					/* set trace on */
					edmDisplayCI , mat(`buffer') ci(`=e(ci)')
					local type1 "rho"
					local type2 "mae"
					forvalues j=1/2 {
						foreach t_type in "lb_mean" "ub_mean" "lb_pco" "ub_pco" "lb_pce" "ub_pce" {
							ereturn scalar `t_type'_`type`j''`ci_counter' =r(`t_type'_`type`j'')
						}
					}
					local ++ci_counter

				}


			}
			mat `summary_r'=`summary_r'[2...,.]
			ereturn matrix summary = `summary_r'
			di as txt "{hline 70}"
			di as text "Note: Results from `=max(`=e(replicate)',`=e(crossfold)')' runs"


		}


		if e(e_offset) != 0 {
			di as text "Note: Actual E is higher than the specified E due to extras"
		}
		di as text ustrtrim(e(cmdfootnote))
		/* di as txt "Note: E is the embedding dimension" */


	}
	else if e(subcommand) == "xmap" {
		di as txt "Convergent Cross-mapping result for variables {bf:`=e(x)'} and {bf:`=e(y)'}"
		if !inlist("`=e(extraembed)'","",".") {
			di as text "Additional variable" _c
			di cond(wordcount("`=e(extraembed)'")>1,"s","") _c
			di " in the embedding: `=e(extraembed)'"
		}
		if e(missingdistance)>0 & e(missingdistance)!= .{
			di as text "Missing values are assumed to have a distance of " _c
			/* di `:di %8.2g `=e(missingdistance)'' _c */

			if `=e(missingdistance1)' != `=e(missingdistance2)' & `=e(missingdistance1)' !=. & e(direction) != "oneway" {
				di `:di %8.2g `=e(missingdistance1)'' _c
				di " and " _c
				di `:di %8.2g `=e(missingdistance2)''
			}
			else {
				di `:di %8.2g `=e(missingdistance)'' _c
				di " with all values."
			}

		}
		local direction1 = "`=e(y)' ~ `=e(y)'|M(`=e(x)')"
		local direction2 = "`=e(x)' ~ `=e(x)'|M(`=e(y)')"
		forvalues i=1/2{
			if strlen("`direction`i''")>26 {
				local direction`i' = substr("`direction`i''",1,24) + ".."
			}
		}
		local mapp_col_length = min(28, max(strlen("`direction1'"), strlen("`direction2'")) +3)
		local line_length = 50 + `mapp_col_length'
		if (e(replicate) == 1 | e(rep_details) == 1) {
			* the case of no replication
			di as txt "{hline `line_length'}"
			display as text %`mapp_col_length's "Mapping"  _c
			display as text %16s "Library size"  _c
			display as text %16s "rho"  _c
			display as text %16s "MAE"
			di as txt "{hline `line_length'}"
			local max_round= 1+ (e(direction) =="both")
			forvalues round=1/`max_round'{
				if `round' ==1 {
					mat r = e(xmap_1)
				}
				else {
					mat r = e(xmap_2)
				}

				local nr = rowsof(r)
				local kr = colsof(r)


				forvalues i = 1/ `nr' {

					forvalues j=1/`kr' {
						if `j' == 1 {
							display as result %`mapp_col_length's "`direction`=r[`i',`j']''" _c
						}
						else {
							display as result %16s `"`:display `fmt' r[`i',`j'] '"' _c
						}
					}
					display " "
				}
			}

			di as txt "{hline `line_length'}"
		}
		else {
			* the case of replication
			di as txt "{hline `line_length'}"
			display as text %`mapp_col_length's "Mapping"  _c
			display as text %16s "Lib size"  _c
			display as text %16s "Mean rho"  _c
			display as text %16s "Std. Dev."
			di as txt "{hline `line_length'}"
			local dformat "%16s"

			// process the return matrix
			tempname reported_r r buffer summary_r

			forvalues round=1/2{
				if `round' ==1 {
					mat `r' = e(xmap_1)
				}
				else {
					mat `r' = e(xmap_2)
					if e(direction) =="oneway" {
						continue, break
					}

				}

				local nr = rowsof(`r')
				local kr = colsof(`r')
				mat `reported_r' = J(`nr',1,0)

				mat `summary_r' = J(1,6,.)
				/* mat list `r'
				mat list `reported_r' */
				forvalues i = 1/ `nr' {
					mat `buffer' = J(1,2,.)
					if `reported_r'[`i',1] ==1 {
						continue
					}
					local base_direction = `r'[`i',1]
					local base_L = `r'[`i',2]
					forvalues j=1/`nr' {
						if `reported_r'[`j',1] ==0 {
							if `r'[`j',1] == `base_direction' & `r'[`j',2] == `base_L' {
								mat `buffer' = (`buffer'\ `=`r'[`j',3]',`=`r'[`j',4]')
								mat `reported_r'[`j',1] =1
								/* mat list `buffer' */
							}
						}
					}
					// now get the mean and st
					tempname mat_mean mat_sd
					/* mat list `buffer' */
					mata: st_matrix("`mat_sd'", diagonal(sqrt(variance(st_matrix("`buffer'"))))')
					mata: st_matrix("`mat_mean'", mean(st_matrix("`buffer'")))

					di as result %`mapp_col_length's "`direction`base_direction''" _c
					di as result `dformat' `"`: display `fmt' `r'[`i',2] '"' _c
					forvalues j=1/1{
						display as result `dformat' `"`:display `fmt' `mat_mean'[1,`j'] '"' _c
						display as result `dformat' `"`:display `fmt' `mat_sd'[1,`j'] '"' _c
					}
					mat `summary_r' = (`summary_r'\ `=`r'[`i',1]',`=`r'[`i',2]', `=`mat_mean'[1,1]',`=`mat_sd'[1,1]', `=`mat_mean'[1,2]',`=`mat_sd'[1,2]')
					/* mat list `summary_r' */
					display ""
					if `=e(ci)'>0 & `=e(ci)'<100 {
						edmDisplayCI , mat(`buffer') ci(`=e(ci)') maxr(1)
						local type1 "rho"
						local type2 "mae"
						forvalues j=1/1 {
							foreach t_type in "lb_mean" "ub_mean" "lb_pco" "ub_pco" "lb_pce" "ub_pce" {
								ereturn scalar `t_type'_`type`j''`ci_counter' =r(`t_type'_`type`j'')
							}
						}
						local ++ci_counter
					}
				}
			}
			mat `summary_r'=`summary_r'[2...,.]
			ereturn matrix summary = `summary_r'
			di as txt "{hline `line_length'}"
			di as text "Note: Results from `=e(replicate)' replications"
		}



		if "`=e(cmdfootnote)'" != "." {
			di as text ustrtrim(e(cmdfootnote))
		}
		di as txt "Note: The embedding dimension E is `=e(e_actual)'" _c
		/* set trace on */
		if e(e_main) != e(e_actual) {
			di " (including `=e(e_offset)' extra`=cond(e(e_offset)>1,"s","")')"
		}
		else {
			di ""
		}
		/* di as txt "Note: The embedding dimension E is `=e(e)', theta (distance weight) is `=e(theta)'" */
	}
	/* if `=e(ci)'>0 & `=e(ci)'<100 {
		di as text "Note: CI is estimated based on mean +/- " _c
		local sdm:display %10.2f `=invnormal(1-(100-`=e(ci)')/200)'
		di trim("`sdm'") _c
		di "*std / sqrt(`=e(replicate)')"
	} */
	if `=e(force_compute)'==1 {
		di as txt "Note: -force- option is specified. The estimate may not be derived from the specified k."
	}
	if `=e(dt)' ==1 {
		di as txt "Note: Embedding includes the delta of the time variable with a weight of " _c
		if `=e(dtw1)' != `=e(dtw2)' & `=e(dtw2)' !=. & e(direction) !="oneway" {
			di `:di %8.2g `=e(dtw1)'' _c
			di " and " _c
			di `:di %8.2g `=e(dtw2)''
		}
		else {
			di `:di %8.2g `=e(dtw)''
		}
	}

	/* di as txt "For more information, please refer to {help edm:help file} and the article." */
end



program define edmExtractExtra, rclass
	syntax [anything]
	/* di "anything: `anything'" */
	return local extravars = " `anything'"

end

capture mata mata drop smap_block()
mata:
mata set matastrict on
void smap_block(string scalar manifold, string scalar p_manifold, string scalar prediction, string scalar result, string scalar train_use, string scalar predict_use, real scalar theta, real scalar l, string scalar skip_obs, string scalar algorithm, string scalar vars_save, string scalar force, real scalar missingdistance)
{
	real scalar force_compute, k, i
	force_compute = force == "force" // check whether we need to force the computation if k is too high
	real matrix M, Mp, y, ystar,S
	st_view(M, ., tokens(manifold), train_use) //base manifold
	st_view(y, ., prediction, train_use) // known prediction of the base manifold
	st_view(ystar, ., result, predict_use)
	if (p_manifold != "") {
		//data used for prediction is different than the source manifold
		st_view(Mp, ., tokens(p_manifold), predict_use)
	}
	else {
		st_view(Mp, ., tokens(manifold), predict_use)
	}

	st_view(S, ., skip_obs, predict_use)

	/* matlist(M) */
	/* matlist(st_data(., train_use, .)) */
	/* matlist(Mp) */

	if (l <= 0) { //default value of local library size
		k = cols(M)
		l = k + 1 // local library size (E+1) + itself
	}

	real matrix B
	real scalar save_mode
	if (vars_save != "") {
		st_view(B, ., tokens(vars_save), predict_use)
		/* matlist(B) */
		save_mode = 1
	}
	else {
		save_mode = 0
	}
	real scalar n
	n = rows(Mp)
	/* matlist(l) */
	real rowvector b

	for(i=1;i<=n;i++) {
		/* sprintf("%g %g",i,S[i]) */
		b= Mp[i,.]
		ystar[i] = mf_smap_single(M,b,y,l,theta,S[i],algorithm, save_mode*i, B, force_compute,missingdistance)
		/* ystar[i] =1 */
		/* sprintf("%g",ystar[i]) */
	}
	/* return */
	/* sprintf("done") */
}
end


capture mata mata drop mf_smap_single()
mata:
mata set matastrict on
real scalar mf_smap_single(real matrix M, real rowvector b, real colvector y, real scalar l, real scalar theta, real scalar skip_obs, string scalar algorithm, real scalar save_index, real matrix Beta_smap, real scalar force_compute, real scalar missingdistance)
{
	/* real scalar mf_smap_single(real matrix M, real rowvector b, real colvector y, real scalar l, real scalar theta, real scalar skip_obs, string scalar algorithm, real scalar save_index, real matrix Beta_smap, transmorphic scalar Acache) */

	/* M : manifold matrix
	b : the vector used for prediction
	y: existing predicted value for M (same number of rows with M)
	l : library size
	theta: exponential weighting parameter
	skip_obs: number of closest neighbours to skip (to exclude itself sometimes) */

	/* sprintf("begin") */
	real colvector d, w, a
	real colvector ind, v
	real scalar i,j,n,r,n_ls
	n = rows(M)
	d = J(n, 1, 0)
	/* matlist(b) */
	for(i=1;i<=n;i++) {
		a= M[i,.] - b
		/* matlist(a) */
		if (missingdistance !=0) {
			a=editvalue(a,., missingdistance)
		}
		// d is squared distance
		d[i] = a*a'
		/* sprintf("%g",d[i]) */
	}



	/* sprintf("d") */
	/* matlist(d) */
	minindex(d, l+skip_obs, ind, v)
	// create weights for each point in the library
	/* matlist(d[1...5]) */
	/* matlist(l)
	matlist(ind)
	matlist(d[ind[1+skip_obs]])
	matlist(d[ind[1]])
	matlist(d[ind[2]])
	matlist(d[ind[3]]) */

	// find the smallest non-zero distance
	real scalar d_base
	real scalar pre_adj_skip_obs
	pre_adj_skip_obs = skip_obs
	for(j=1;j<=l;j++) {
		if (d[ind[j+skip_obs]] == 0) {
			skip_obs++
		}
		else {
			break
		}
	}
	if (pre_adj_skip_obs!=skip_obs) {
		minindex(d, l+skip_obs, ind, v)
	}
	if (d[ind[1+skip_obs]] == 0) {
		d= editvalue(d, 0,.)
		/* sprintf("search failed") */
		/* skip_obs++ */
		skip_obs = 0
		minindex(d, l+skip_obs, ind, v)
	}
	d_base = d[ind[1+skip_obs]]
	/* if (d_base ==0) {
		sprintf("error")
	} */
	/* sprintf("dbase %g with %g",d_base, skip_obs) */
	w = J(l+skip_obs, 1, .)
	if (rows(ind)<l+skip_obs) {
		if (force_compute==1) {
			l=rows(ind)-skip_obs // change l to match neighbor size
			/* sprintf("library size has been reduced for some observations")	 */
			if (l<=0) {
				sprintf("Insufficient number of unique observations in the dataset even with -force- option.")
				exit(error(503))
			}
		}
		else {
			sprintf("Insufficient number of unique observations, consider tweaking the values of E, k or use -force- option")
			exit(error(503))
		}
	}
	// note the w, X_ls, y_ls matrix are larger than necessary, the first skip_obs rows are not used
	r = 0
	/* matlist(y)
	matlist(ind)
	matlist(w) */
/*
	matlist(ind)
	matlist(w) */

	if (algorithm == "" | algorithm == "simplex") {
		for(j=1+skip_obs;j<=l+skip_obs;j++) {
			w[j] = exp(-theta*(d[ind[j]] / d_base)^(1/2))
		}
		w = w/sum(w)
		for(j=1+skip_obs;j<=l+skip_obs;j++) {
			/* sprintf("%g", j)  */
			/* sprintf("%g %g %g", j, y[ind[j]],w[j])  */
			r = r +  y[ind[j]] * w[j]
			/* sprintf("%g %g", y[ind[j]], w[j])  */
		}
		/* sprintf("%g", r)  */
		return(r)
	}
	else if (algorithm =="smap" | algorithm =="llr") {

		real colvector y_ls, b_ls, w_ls
		real matrix X_ls, XpXi
		real rowvector x_pred
		real scalar mean_w

		for(j=1+skip_obs;j<=l+skip_obs;j++) {
			w[j] = d[ind[j]] ^ (1/2)
		}
		mean_w = mean(w)
		for(j=1+skip_obs;j<=l+skip_obs;j++) {
			w[j] = exp(-theta*(w[j] / mean_w))
		}

		y_ls = J(l, 1, .)
		X_ls = J(l, cols(M), .)
		w_ls = J(l, 1, .)

		real scalar rowc
		rowc = 0
		/* sprintf("start") */
		for(j=1+skip_obs;j<=l+skip_obs;j++) {
			if (hasmissing(y[ind[j]]) | hasmissing(M[ind[j],.])) {
				continue
			}
			rowc++
			if (algorithm == "llr") {
				y_ls[rowc]    = y[ind[j]]
				/* matlist(X_ls[j,.]) */
				X_ls[rowc,.]    = M[ind[j],.]
				/* matlist(X_ls[j,.]) */
				w_ls[rowc] = w[j]
			}
			else if (algorithm =="smap") {
				y_ls[rowc]    = y[ind[j]] * w[j]
				X_ls[rowc,.]    = M[ind[j],.] * w[j]
				w_ls[rowc] = w[j]
			}
		}
		if (rowc ==0) {
			return(.)
		}
		/* sprintf("done")
		matlist(rowc) */

		y_ls =y_ls[1..rowc]
		X_ls =X_ls[1..rowc,.]
		w_ls = w_ls[1..rowc]

		n_ls   = rows(X_ls)
		// add constant
		X_ls    = w_ls,X_ls

		/* matlist(w)
		printf("cutw")
		matlist(w_ls[1..rowc])
		matlist(X_ls) */

		/* matlist(X_ls) */
		/* printf("w")
		matlist(w) */

		/* X_ls    = X_ls,J(n_ls,1,1) */

		if (algorithm == "llr") {
			XpXi = quadcross(X_ls, w_ls, X_ls)
			XpXi = invsym(XpXi)
			b_ls    = XpXi*quadcross(X_ls, w_ls, y_ls)
		}
		else {
			b_ls = svsolve(X_ls, y_ls)
		}
/*
		matlist(skip_obs)
		matlist(w)
		matlist(X_ls)
		matlist(y_ls)
		matlist(b_ls) */

		/* if (hasmissing(b_ls) | b_ls[1]==0 ){
			sprintf("colinearity?")
			matlist(X_ls)
			matlist(y_ls)
			matlist(w_ls)
			matlist(b_ls)
		} */

		if (save_index>0) {
			Beta_smap[save_index,.] = editvalue(b_ls',0,.)
			/* Beta_smap[save_index,.] = b_ls' */
		}
		/* matlist(b)
		matlist(b_ls) */
		x_pred = 1,editvalue(b,.,0)

		/* matlist(x_pred)
		matlist(b_ls) */
		r = x_pred * b_ls

		/* sprintf("R")
		matlist(r) */
		return(r)
	}

}
end

//smap block C implementation
cap program smap_block_mdap, plugin using(edm_`=c(os)'_x`=c(bit)'.plugin)


/*
capture mata mata drop matlist()
mata:
void matlist(
	real matrix X,
	| string scalar fmt
	)
{
	real scalar     i, j, wd, rw, cw
	string scalar   sfmt

	if (fmt=="") fmt = "%g"
	wd = strlen(sprintf(fmt,-1/3))

	if (length(X)==0) return

	rw = trunc(log10(rows(X))) + 1
	cw = trunc(log10(cols(X))) + 1
	wd = max((cw,wd)) + 2
	sfmt = "%"+strofreal(wd)+"s"

	printf("{txt}"+(2+rw+1+1)*" ")
	for (j=1;j<=cols(X);j++) {
		printf(sfmt+" ", sprintf("%g", j))
	}
	printf("  \n")
	printf((2+rw+1)*" " + "{c TLC}{hline " +
		strofreal((wd+1)*cols(X)+1) + "}{c TRC}\n")
	for (i=1;i<=rows(X);i++) {
		printf("{txt}  %"+strofreal(rw)+"s {c |}{res}", sprintf("%g", i))
		for (j=1;j<=cols(X);j++) {
			printf(sfmt+" ",sprintf(fmt, X[i,j]))
		}
		printf(" {txt}{c |}\n")
	}
	printf((2+rw+1)*" " + "{c BLC}{hline " +
		strofreal((wd+1)*cols(X)+1) + "}{c BRC}\n")
}
end  */
