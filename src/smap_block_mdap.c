/* version 2.1, 09 Sep 2020, Edoardo Tescari, Melbourne Data Analytics Platform,
   The University of Melbourne, e.tescari@unimelb.edu.au */

#include "stplugin.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_linalg.h>

/* internal functions */

ST_double mf_smap_single(ST_int rowsm, ST_int colsm, ST_double (*)[],\
			 ST_double*, ST_double*, ST_int, ST_int, ST_double,\
			 char*, ST_int, ST_double*, ST_int, ST_int);

ST_int minindex(ST_int, ST_double*, ST_int, ST_int*);

void quicksortind(ST_double*, ST_int*, ST_int, ST_int);

/* global variable placeholder for missing values */

ST_double missval = 1.0e+100;

/* OpenMP routines */

int omp_get_num_procs(void);

void omp_set_num_threads(int num_threads);

/*
Example call to the plugin:

local myvars ``manifold'' `co_mapping' `x_f' `x_p' `train_set' `predict_set' `overlap' `vars_save'

unab vars : ``manifold''
local mani `: word count `vars''

local pmani_flag = 0
                               
local vsave_flag = 0

plugin call smap_block_mdap `myvars', `j' `lib_size' "`algorithm'" "`force'" `missingdistance' `mani' `pmani_flag' `vsave_flag'
*/

STDLL stata_call(int argc, char *argv[])
{
  ST_int nvars, nobs, first, last, mani, pmani_flag, pmani, smaploc;
  ST_int Mpcol, l, vsave_flag, save_mode, varssv, theta;

  ST_double value, *train_use, *predict_use, *skip_obs;
  ST_double *y, *S, *Bi, *ystar;

  ST_int i, j, h, force_compute, missingdistance;
  ST_int count_train_set, count_predict_set, allocbim, obsi, obsj;
  
  char temps[500], algorithm[500];
  
  /* header of the plugin */
  SF_display("\n");
  SF_display("====================\n");
  SF_display("Start of the plugin\n");
  SF_display("\n");

  /* overview of variables and arguments passed and observations in sample */
  nvars = SF_nvars();
  nobs = SF_nobs();
  first = SF_in1();
  last  = SF_in2();
  sprintf(temps,"number of vars & obs = %i, %i\n",nvars,nobs);
  SF_display(temps);
  sprintf(temps,"first and last obs in sample = %i, %i\n",first,last);
  SF_display(temps);
  SF_display("\n");

  for(i=0;i < argc; i++) {
    sprintf(temps,"arg %i: ",i);
    SF_display(temps);
    SF_display(argv[i]);
    SF_display("\n");
  }
  SF_display("\n");
  
  /* allocation of variable force_compute based on fourth argument */
  if (strcmp(argv[3],"force") == 0)
    force_compute = 1;
  else
    force_compute = 0;
  sprintf(temps,"force compute = %i\n",force_compute);
  SF_display(temps);
  SF_display("\n");

  /* allocation of string variable algorithm based on third argument */
  sprintf(algorithm,"%s",argv[2]);
  sprintf(temps,"algorithm = %s\n",algorithm);
  SF_display(temps);
  SF_display("\n");

  /* allocation of variable missingdistance based on fifth argument */
  missingdistance = atoi(argv[4]);
  sprintf(temps,"missing distance = %i\n",missingdistance);
  SF_display(temps);
  SF_display("\n");

  /* allocation of number of columns in manifold */
  mani = atoi(argv[5]);
  sprintf(temps,"number of variables in manifold = %i \n",mani);
  SF_display(temps);
  SF_display("\n");
  
  /* allocation of train_use, predict_use and skip_obs variables */
  train_use = (ST_double*)malloc(sizeof(ST_double)*nobs);
  predict_use = (ST_double*)malloc(sizeof(ST_double)*nobs);
  skip_obs = (ST_double*)malloc(sizeof(ST_double)*nobs);
  if ((train_use == NULL) || (predict_use == NULL) || (skip_obs == NULL)) {
    sprintf(temps,"Insufficient memory\n");
    SF_error(temps);
    return((ST_retcode)909);
  }
  
  count_train_set = 0;
  count_predict_set = 0;
  for(i=1; i<=(last-first+1); i++) {
    SF_vdata(mani+3, i, &value);
    train_use[i-1] = value;
    if (value == 1.) count_train_set++;
    if (SF_is_missing(value)) {
      /* missing value */
      train_use[i-1] = missval;
    }
    SF_vdata(mani+4, i, &value);
    predict_use[i-1] = value;
    if (value == 1.) count_predict_set++;
    if (SF_is_missing(value)) {
      /* missing value */
      predict_use[i-1] = missval;
    }
    SF_vdata(mani+5, i, &value);
    skip_obs[i-1] = value;
    if (SF_is_missing(value)) {
      /* missing value */
      skip_obs[i-1] = missval;
    }
  }
  sprintf(temps,"train set obs: %i\n",count_train_set);
  SF_display(temps);
  sprintf(temps,"predict set obs: %i\n",count_predict_set);
  SF_display(temps);
  SF_display("\n");
  
  /* allocation of matrices M and y */
  ST_double (*M)[mani] = malloc(count_train_set*sizeof(*M)); 
  y = (ST_double*)malloc(sizeof(ST_double)*count_train_set);
  if ((*M == NULL) || (y == NULL)) {
    sprintf(temps,"Insufficient memory\n");
    SF_error(temps);
    return((ST_retcode)909);
  }

  obsi = 0;
  for(i=0; i<nobs; i++) {
    if (train_use[i] == 1.) {  
      for(j=0; j<mani; j++) {
        SF_vdata(j+1, i+1, &value);
        M[obsi][j] = value;
        if (SF_is_missing(value)) {
          /* missing value */
          M[obsi][j] = missval;
        }
      }
      SF_vdata(j+1, i+1, &value);
      y[obsi] = value;
      if (SF_is_missing(value)) {
        /* missing value */
        y[obsi] = missval;
      }
      obsi++;
    }
  }
  
  /* allocation of matrices Mp, S, ystar */
  pmani_flag = atoi(argv[6]); /* contains the flag for p_manifold */
  sprintf(temps,"p_manifold flag = %i \n",pmani_flag);
  SF_display(temps);

  if (pmani_flag == 1) {
    pmani = atoi(argv[8]); /* contains the number of columns in p_manifold */
    sprintf(temps,"number of variables in p_manifold = %i \n",pmani);
    SF_display(temps);
    SF_display("\n");
    Mpcol = pmani;
  } else {
    SF_display("\n");
    Mpcol = mani;
  }
  ST_double (*Mp)[Mpcol] = malloc(count_predict_set*sizeof(*Mp));
  S = (ST_double*)malloc(sizeof(ST_double)*count_predict_set);
  if ((*Mp == NULL) || (S == NULL)) {
    sprintf(temps,"Insufficient memory\n");
    SF_error(temps);
    return((ST_retcode)909);
  }
  
  if (pmani_flag == 1) {
    smaploc = mani+5+Mpcol+1;
    obsi = 0;
    for(i=0; i<nobs; i++) {
      if (predict_use[i] == 1.) {
	obsj = 0;
        for(j=mani+5; j<mani+5+Mpcol; j++) {
          SF_vdata(j+1, i+1, &value);
          Mp[obsi][obsj] = value;
          if (SF_is_missing(value)) {
            /* missing value */
            Mp[obsi][obsj] = missval;
          }
	  obsj++;
        }
        S[obsi] = skip_obs[i];
        obsi++;
      }
    }
  } else {
    smaploc = mani+5+1;
    obsi = 0;
    for(i=0; i<nobs; i++) {
      if (predict_use[i] == 1.) {
        for(j=0; j<Mpcol; j++) {
          SF_vdata(j+1, i+1, &value);
          Mp[obsi][j] = value;
          if (SF_is_missing(value)) {
            /* missing value */
            Mp[obsi][j] = missval;
          }
        }
        S[obsi] = skip_obs[i];
        obsi++;
      }
    }
  }
  
  l = atoi(argv[1]); /* contains l */
  if (l <= 0) {
    l = mani + 1;
  }
  sprintf(temps,"l = %i \n",l);
  SF_display(temps);
  SF_display("\n");

  vsave_flag = atoi(argv[7]); /* contains the flag for vars_save */
  sprintf(temps,"vars_save flag = %i \n",vsave_flag);
  SF_display(temps);

  if (vsave_flag == 1) { /* flag savesmap is ON */
    varssv = atoi(argv[8]); /* contains the number of columns
                               in smap coefficents */
    sprintf(temps,"coumns in smap coefficents = %i \n",varssv);
    SF_display(temps);
    allocbim = count_predict_set;
    save_mode = 1;
  } else { /* flag savesmap is OFF */
    varssv = 1; /* to avoid allocating arrays Bi and Bi_map of zero dimension */
    allocbim = 1; /* to minimise the size of Bi_map, not used in this case */
    save_mode = 0;
  }
  Bi = (ST_double*)malloc(sizeof(ST_double)*varssv);
  ST_double (*Bi_map)[varssv] = malloc(allocbim*sizeof(*Bi_map));
  if ((Bi == NULL) || (*Bi_map == NULL)) {
    sprintf(temps,"Insufficient memory\n");
    SF_error(temps);
    return((ST_retcode)909);
  }
  if (vsave_flag != 1) {
    varssv = 0; /* reset varssv to 0 in case flag savesmap is OFF */
  }
  sprintf(temps,"save_mode = %i \n",save_mode);
  SF_display(temps);
  SF_display("\n");

  ystar = (ST_double*)malloc(sizeof(ST_double)*count_predict_set);
  if (ystar == NULL) {
    sprintf(temps,"Insufficient memory\n");
    SF_error(temps);
    return((ST_retcode)909);
  }
  
  theta = atoi(argv[0]); /* contains value of theta = first argument */
  sprintf(temps,"theta = %i\n",theta); /* CHECK TYPE OF theta */
  SF_display(temps);
  SF_display("\n");
  
  /* setting the number of OpenMP threads = number of processors available */ 
  omp_set_num_threads(omp_get_num_procs());
  sprintf(temps,"Using %i OpenMP threads \n",omp_get_num_procs());
  SF_display(temps);
  SF_display("\n");
  
  /* OpenMP loop with call to mf_smap_single function */
  #pragma omp parallel
  {
     ST_double *b = Mp[i];
     ST_int o;
    
    #pragma omp for
    for (i=0; i<count_predict_set; i++) {

      ystar[i] = mf_smap_single(count_train_set,mani,M,b,y,l,theta,S[i],\
			        algorithm,save_mode*(i+1),Bi,force_compute,\
			        missingdistance);
      
      for (o=0; o<varssv; o++) {
        Bi_map[i][o] = Bi[o];
      }
    }
  }

  /*for (i=0; i<count_predict_set; i++) {
    sprintf(temps,"ystar[%i] = %12.10f \n",i, ystar[i]);
    SF_display(temps);
  }*/
  
  /* returning the value of ystar (and smap coefficients) to Stata */
  j=0;
  for (i=0; i < nobs; i++) {
    if (predict_use[i] == 1) {
      if (ystar[j] != missval) {
        SF_vstore(mani+2,i+1,ystar[j]);
      } else {
	/* returning a missing value */
	SF_vstore(mani+2,i+1,SV_missval);
      }
      if (vsave_flag == 1) {
	for (h=0; h < varssv; h++) {
          if (Bi_map[j][h] != missval) {
            SF_vstore(smaploc+h,i+1,Bi_map[j][h]);
	  } else {
	    SF_vstore(smaploc+h,i+1,SV_missval);
	  }
	}
      }
      j++;
    }
  }
  
  /* deallocation of matrices and arrays before exiting the plugin */
  free(train_use);
  free(predict_use);
  free(skip_obs);
  free(M);
  free(y);
  free(Mp);
  free(S);
  free(Bi);
  free(Bi_map);
  free(ystar);
  
  /* footer of the plugin */
  SF_display("\n");
  SF_display("End of the plugin\n");
  SF_display("====================\n");
  SF_display("\n");
  
  return(0);

}

ST_double mf_smap_single(ST_int rowsm, ST_int colsm, ST_double (*M)[colsm],\
			 ST_double b[], ST_double y[], ST_int l, ST_int theta,\
			 ST_double skip_obs, char *algorithm,\
			 ST_int save_index, ST_double Beta_smap[],\
			 ST_int force_compute, ST_int missingdistance)
{
  ST_double *d, *a, *w;
  ST_int *ind;

  ST_double value, pre_adj_skip_obs, d_base, sumw, r;

  ST_int i, j, numind, boolmiss;
  
  char temps[500];

  d = (ST_double*)malloc(sizeof(ST_double)*rowsm);
  a = (ST_double*)malloc(sizeof(ST_double)*colsm);
  ind = (ST_int*)malloc(sizeof(ST_int)*rowsm);
  if ((d == NULL) || (a == NULL) || (ind == NULL)) {
    sprintf(temps,"Insufficient memory\n");
    SF_error(temps);
    return((ST_retcode)909);
  }

  for (i=0; i<rowsm; i++) {
    value = 0.;
    boolmiss = 0;
    for (j=0; j<colsm; j++) {
      if ((M[i][j] == missval) || (b[j] == missval)) {
        if (missingdistance !=0) {
          a[j] = (ST_double)missingdistance;
	  value = value + a[j]*a[j];
	} else {
          boolmiss = 1;
	}
      } else {
        a[j] = M[i][j] - b[j];
	value = value + a[j]*a[j]; 
      }
    }
    if (boolmiss == 1) {
      d[i] = missval;
    } else {
      d[i] = value;
    }
    ind[i] = i;
  }
  
  numind = minindex(rowsm,d,l+(int)skip_obs,ind);

  pre_adj_skip_obs = skip_obs;

  for (j=0; j<l; j++) {
    if (d[ind[j+(int)skip_obs]] == 0.) {
      skip_obs++;
    } else {
      break;
    }
  }

  if (pre_adj_skip_obs!=skip_obs) {
    numind = minindex(rowsm,d,l+(int)skip_obs,ind);
  }

  if (d[ind[(int)skip_obs]] == 0.) {
    for (i=0; i<rowsm; i++) {
      if (d[i] == 0.) {
        d[i] = missval;
      }
    }
    skip_obs = 0.;
    numind = minindex(rowsm,d,l+(int)skip_obs,ind);
  }
  
  d_base = d[ind[(int)skip_obs]];

  if (numind < l+(int)skip_obs) {
    if (force_compute == 1) {
      l = numind - (int)skip_obs;
      if (l <= 0) {
	sprintf(temps,"Insufficient number of unique observations in the "
                      "dataset even with -force- option\n");
        SF_error(temps);
        return((ST_retcode)503);
      }
    } else {
      sprintf(temps,"Insufficient number of unique observations, consider "
                    "tweaking the values of E, k or use -force- option\n");
      SF_error(temps);
      return((ST_retcode)503);
    }
  }

  w = (ST_double*)malloc(sizeof(ST_double)*(l+(int)skip_obs));
  if (w == NULL) {
    sprintf(temps,"Insufficient memory\n");
    SF_error(temps);
    return((ST_retcode)909);
  }

  sumw = 0.;
  r = 0.;
  if ((strcmp(algorithm,"") == 0) || (strcmp(algorithm,"simplex") == 0)) {
    for(j=(int)skip_obs; j<l+(int)skip_obs; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = exp(-(ST_double)theta*pow((d[ind[j]] / d_base),(0.5))); */
      w[j] = exp(-(ST_double)theta*sqrt(d[ind[j]] / d_base));
      sumw = sumw + w[j];
    }
    for(j=(int)skip_obs; j<l+(int)skip_obs; j++) {
      r = r + y[ind[j]] * (w[j] / sumw);
    }
    /* deallocation of matrices and arrays before exiting the function */
    free(d);
    free(a);
    free(ind);
    free(w);

    /* returning the value of ystar[j] */
    return(r);
    
  } else if ((strcmp(algorithm,"smap") == 0) ||\
	     (strcmp(algorithm,"llr") == 0)) { 
 
    ST_double *y_ls, *w_ls, *b_ls, mean_w;

    ST_int rowc, bocont;

    y_ls = (ST_double*)malloc(sizeof(ST_double)*l);
    w_ls = (ST_double*)malloc(sizeof(ST_double)*l);
    ST_double (*X_ls)[colsm] = malloc(l*sizeof(*X_ls));
    if ((y_ls == NULL) || (w_ls == NULL) || (*X_ls == NULL)) {
      sprintf(temps,"Insufficient memory\n");
      SF_error(temps);
      return((ST_retcode)909);
    }

    mean_w = 0.;
    for(j=(int)skip_obs; j<l+(int)skip_obs; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = pow(d[ind[j]],0.5); */
      w[j] = sqrt(d[ind[j]]);
      mean_w = mean_w + w[j];
    }
    mean_w = mean_w / (ST_double)l;
    for(j=(int)skip_obs; j<l+(int)skip_obs; j++) {
      w[j] = exp(-(ST_double)theta * (w[j] / mean_w));
    }
    
    rowc = -1;
    for(j=(int)skip_obs; j<l+(int)skip_obs; j++) {
      if (y[ind[j]] == missval) {
	continue;
      }
      bocont = 0;
      for (i=0; i<colsm; i++) {
        if (M[ind[j]][i] == missval) {
	  bocont = 1;
        }
      }
      if (bocont == 1) {
	continue;
      }
      rowc++;
      if (strcmp(algorithm,"llr") == 0) {
	/* llr algorithm is not needed at this stage */
	sprintf(temps,"llr algorithm not yet implemented\n");
        SF_error(temps);
        return((ST_retcode)908);

      } else if (strcmp(algorithm,"smap") == 0) {
	y_ls[rowc] = y[ind[j]] * w[j];
	w_ls[rowc] = w[j];
	for (i=0; i<colsm; i++) {
	  X_ls[rowc][i] = M[ind[j]][i] * w[j];
	}
      }
    }
    if (rowc == -1) {
      /* deallocation of matrices and arrays before exiting the function */
      free(d);
      free(a);
      free(ind);
      free(w);
      free(y_ls);
      free(w_ls);
      free(X_ls);
      
      /* return missing value flag to ystar[j] */
      return(missval);
      
    }

    gsl_matrix *X_ls_cj = gsl_matrix_alloc(rowc+1,colsm+1);
    gsl_vector *y_ls_cj = gsl_vector_alloc(rowc+1);
    if ((X_ls_cj == NULL) || (y_ls_cj == NULL)) {
      sprintf(temps,"Insufficient memory\n");
      SF_error(temps);
      return((ST_retcode)909);
    }
    for(i=0; i<rowc+1; i++) {
      gsl_matrix_set(X_ls_cj,i,0,w_ls[i]);
      gsl_vector_set(y_ls_cj,i,y_ls[i]);
      for (j=1; j<colsm+1; j++) {
	gsl_matrix_set(X_ls_cj,i,j,X_ls[i][j-1]);
      }
    }
    
    if (strcmp(algorithm,"llr") == 0) {
	/* llr algorithm is not needed at this stage */
	sprintf(temps,"llr algorithm not yet implemented\n");
        SF_error(temps);
        return((ST_retcode)908);
    } else {
      gsl_matrix *V = gsl_matrix_alloc(colsm+1,colsm+1);
      gsl_vector *Esse = gsl_vector_alloc(colsm+1);
      gsl_vector *ics = gsl_vector_alloc(colsm+1);
      if ((V == NULL) || (Esse == NULL) || (ics == NULL)) {
        sprintf(temps,"Insufficient memory\n");
        SF_error(temps);
        return((ST_retcode)909);
      }

      /* singular value decomposition (SVD) of X_ls_cj, using gsl libraries */

      /* TOBE ADDED: benchmark which one of the following methods work best*/
      /*Golub-Reinsch SVD algorithm*/
      /*gsl_linalg_SV_decomp(X_ls_cj,V,Esse,ics);*/

      /* one-sided Jacobi orthogonalization method */
      gsl_linalg_SV_decomp_jacobi(X_ls_cj,V,Esse);

      /* setting to zero extremely small values of Esse to avoid
	 underflow errors */
      for (j=0; j<colsm+1; j++) {
        if (gsl_vector_get(Esse,j) < 1.0e-12) {
	  gsl_vector_set(Esse,j,0.);
        }
      }
      
      /* function to solve X_ls_cj * ics = y_ls_cj and return ics,
	 using gsl libraries */
      gsl_linalg_SV_solve(X_ls_cj,V,Esse,y_ls_cj,ics);

      /* saving ics coefficients if savesmap option enabled */
      if (save_index > 0) {  
        for (j=0; j<colsm+1; j++) {
	  if (gsl_vector_get(ics,j) == 0.) {
	    Beta_smap[j] = missval;
	  } else {
	    Beta_smap[j] = gsl_vector_get(ics,j);
	  }
        }
      }
      
      r = gsl_vector_get(ics,0);
      for (j=1; j<colsm+1; j++) {
        if (b[j-1] == missval) {
	  b[j-1] = 0.;
        }
        r = r + b[j-1] * gsl_vector_get(ics,j);
      }
      
      /* deallocation of matrices and arrays before exiting the function */
      free(d);
      free(a);
      free(ind);
      free(w);
      free(y_ls);
      free(w_ls);
      free(X_ls);
      gsl_matrix_free(X_ls_cj);
      gsl_vector_free(y_ls_cj);
      gsl_matrix_free(V);
      gsl_vector_free(Esse);
      gsl_vector_free(ics);
   
      /* returning the value of ystar[j] */
      return(r);
      
    }
  }
  
  /* returning the result to the main program, added just to avoid
     a warning when compiling, not really necessary */
  return(-1.);
  
}

/* NOTE: in mata, minindex(v,k,i,w) returns in i and w the indices of the
   k minimums of v. The internal function minindex below only returns i +
   the number of k minimums and does not return w, as w is not used in the
   original edm code */
ST_int minindex(ST_int rvect, ST_double vect[], ST_int k,\
	      ST_int ind[])
{
  ST_int i, j, contin, numind, count_ord, *subind;

  ST_double tempval, *temp_ind;

  char temps[500];
  
  quicksortind(vect,ind,0,rvect-1);
  
  tempval = vect[ind[0]];
  contin = 0;
  numind = 0;
  count_ord = 1;
  i = 1;
  while ((contin < k) && (i < rvect)) {
    if (vect[ind[i]] != tempval) {
      tempval = vect[ind[i]];
      if (count_ord > 1) {
	/* here I reorder the indexes from low to high in case of
	   repeated values */
        temp_ind = (ST_double*)malloc(sizeof(ST_double)*count_ord);
	subind = (ST_int*)malloc(sizeof(ST_int)*count_ord);
	if ((temp_ind == NULL) || (subind == NULL)) {
          sprintf(temps,"Insufficient memory\n");
          SF_error(temps);
          return((ST_retcode)909);
        }
        for (j=0; j<count_ord; j++) {
	  temp_ind[j] = (ST_double)ind[i-1-j];
	  subind[j] = j;
	}
	quicksortind(temp_ind,subind,0,count_ord-1);
        for (j=0; j<count_ord; j++) {
	  ind[i-1-j] = (ST_int)temp_ind[subind[count_ord-1-j]];
	}
	free(temp_ind);
	free(subind);
	count_ord = 1;
      }
      contin++;
      numind++;
      i++;
    } else {
      numind++;
      count_ord++;
      i++;
      if (i == rvect) {
	/* here I check whether I reached the end of the array */
        if (count_ord > 1) {
	  /* here I reorder the indexes from low to high in case of
	     repeated values */
          temp_ind = (ST_double*)malloc(sizeof(ST_double)*count_ord);
	  subind = (ST_int*)malloc(sizeof(ST_int)*count_ord);
          if ((temp_ind == NULL) || (subind == NULL)) {
            sprintf(temps,"Insufficient memory\n");
            SF_error(temps);
            return((ST_retcode)909);
          }
	  for (j=0; j<count_ord; j++) {
	    temp_ind[j] = (ST_double)ind[i-1-j];
	    subind[j] = j;
	  }
	  quicksortind(temp_ind,subind,0,count_ord-1);
          for (j=0; j<count_ord; j++) {
	    ind[i-1-j] = (ST_int)temp_ind[subind[count_ord-1-j]];
	  }
	  free(temp_ind);
	  free(subind);
        }
      }
    }
  }

  /* returning the number of k minimums (and indices via ind) */
  return numind;
  
}

/* function that returns the sorted indices of an array */
void quicksortind(ST_double A[], ST_int I[], ST_int lo, ST_int hi)
{
  while (lo < hi) {
    ST_double pivot = A[I[lo + (hi - lo) / 2]];
    ST_int t;
    ST_int i = lo - 1;
    ST_int j = hi + 1;
    while (1) {
      while (A[I[++i]] < pivot);
      while (A[I[--j]] > pivot);
      if (i >= j)
        break;
      t = I[i];
      I[i] = I[j];
      I[j] = t;
    }
    /* avoid stack overflow */
    if((j - lo) < (hi - j)) {
      quicksortind(A, I, lo, j);
      lo = j+1;
    } else {
      quicksortind(A, I, j + 1, hi);
      hi = j;
    }
  }
}
