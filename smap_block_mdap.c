/* version 1, 22 July 2020, Edoardo Tescari, Melbourne Data Analytics Platform,
   The University of Melbourne, e.tescari@unimelb.edu.au */

#include "stplugin.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* internal functions */

ST_double mf_smap_single(ST_int rowsm, ST_int colsm, ST_double (*)[],\
			 ST_double*, ST_double*, ST_int, ST_int, ST_double,\
			 char*, ST_int, ST_int, ST_int);

ST_int minindex(ST_int, ST_double*, ST_int, ST_int*);

void quicksortind(ST_double*, ST_int*, ST_int, ST_int);

/* global variables d and ind used in minindex and quicksortind functions */

ST_double *d;
ST_int *ind;

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
  ST_int nvars, nobs, first, last, mani, pmani_flag;
  ST_int Mpcol, l, vsave_flag, save_mode, theta;
  ST_retcode rc;

  ST_double value, *train_use, *predict_use, *skip_obs;
  ST_double *y, *ystar, *S, *b;

  ST_int i, j, force_compute, missingdistance, count_obs, obsi;
  ST_double missval = -1.; /* placeholder for missing values */
  
  char temps[500], algorithm[500];
  
  /* header of the plugin */
  SF_display("\n");
  SF_display("====================\n");
  SF_display("Start of the plugin\n");
  SF_display("\n");

  /* overview on variables and arguments passed and observations in sample */
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
  
  /* rc holds the return code that the plugin will return to Stata */
  rc = (ST_retcode) 0;
  
  /* allocation of train_use, predict_use and skip_obs variables */

  train_use = (ST_double*)malloc(sizeof(ST_double)*nobs);
  predict_use = (ST_double*)malloc(sizeof(ST_double)*nobs);
  skip_obs = (ST_double*)malloc(sizeof(ST_double)*nobs);

  if ((train_use == NULL) || (predict_use == NULL) || (skip_obs == NULL)) {
    sprintf(temps,"Insufficient memory\n");
    SF_error(temps);
    return( (ST_retcode)909);
  }
  
  count_obs = 0;
  
  for(i=1; i<=(last-first+1); i++) {
    SF_vdata(5, i, &value);
    train_use[i-1] = value;
    if (SF_is_missing(value)) {
      /* missing value */
      train_use[i-1] = missval;
    }
    SF_vdata(6, i, &value);
    predict_use[i-1] = value;
    if (value == 1) count_obs++;
    if (SF_is_missing(value)) {
      /* missing value */
      predict_use[i-1] = missval;
    }
    SF_vdata(7, i, &value);
    skip_obs[i-1] = value;
    if (SF_is_missing(value)) {
      /* missing value */
      skip_obs[i-1] = missval;
    }
  }
  sprintf(temps,"observations selected: %i\n",count_obs);
  SF_display(temps);
  SF_display("\n");
  
  /* allocation of matrices M and y */

  mani = atoi(argv[5]); /* contains the number of columns in manifold */
  sprintf(temps,"number of variables in manifold = %i \n",mani);
  SF_display(temps);
  SF_display("\n");
  
  ST_double (*M)[mani] = malloc(count_obs*sizeof(*M)); 
  y = (ST_double*)malloc(sizeof(ST_double)*count_obs);
  if ((*M == NULL) || (y == NULL)) {
    sprintf(temps,"Insufficient memory\n");
    SF_error(temps);
    return( (ST_retcode)909);
  }

  obsi = 0;
  for(i=0; i<nobs; i++) {
    if (train_use[i] == 1) {  
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
  
  /* allocation of matrices ystar, Mp, S and b */

  pmani_flag = atoi(argv[6]); /* contains the flag for p_manifold */
  sprintf(temps,"p_manifold flag = %i \n",pmani_flag);
  SF_display(temps);
  SF_display("\n");
  
  /* TO BE ADDED */
  /* allocation of Mp according to mani or p_mani flags*/
  Mpcol = mani; /* for now, to be changed according to the flag */
  
  ST_double (*Mp)[Mpcol] = malloc(count_obs*sizeof(*Mp));
  S = (ST_double*)malloc(sizeof(ST_double)*count_obs);
  if ((*Mp == NULL) || (S == NULL)) {
    sprintf(temps,"Insufficient memory\n");
    SF_error(temps);
    return( (ST_retcode)909);
  }
  
  if (pmani_flag == 1) {

    /* TO BE ADDED */
    /* st_view(Mp, ., tokens(p_manifold), predict_use) */
    
  } else {

    obsi = 0;
    for(i=0; i<nobs; i++) {
      if (predict_use[i] == 1) {  
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

  if (vsave_flag == 1) {

    /* TO BE ADDED */
    /* st_view(B, ., tokens(vars_save), predict_use)*/
    save_mode = 1; /*CHECK TYPE OF save_mode*/
    
  } else {
    save_mode = 0; /* CHECK TYPE OF save_mode */
  }
  sprintf(temps,"save_mode = %i \n",save_mode);
  SF_display(temps);
  SF_display("\n");

  ystar = (ST_double*)malloc(sizeof(ST_double)*count_obs);
  b = (ST_double*)malloc(sizeof(ST_double)*mani);

  theta = atoi(argv[0]); /* contains value of theta = first argument */
  sprintf(temps,"theta = %i \n",theta); /* CHECK TYPE OF theta */
  SF_display(temps);
  SF_display("\n");
  
  /* loop with call to mf_smap_single function */
  for (i=0; i<count_obs; i++) {
    for (j=0; j<Mpcol; j++) {
      b[j] = Mp[i][j];
    }

    /* TO BE ADDED case when save_mode = 1 and matrix B is allocated */
    
    ystar[i] = mf_smap_single(count_obs,mani,M,b,y,l,theta,S[i],algorithm,\
			      save_mode*i,force_compute,missingdistance);

    //sprintf(temps,"ystar[%i] = %12.10f \n",i, ystar[i]);
    //SF_display(temps);
  }

  /* returning the value of ystar to Stata */
  /* TO BE ADDED: check variable number in varlist */
  j=0;
  for (i=0; i < nobs; i++) {
    if (predict_use[i] == 1) {
      SF_vstore(4,i+1,ystar[j]);
      j++;
    }
  }
  
  /* deallocation of matrices and arrays before exiting the plugin */
  free(train_use);
  free(predict_use);
  free(skip_obs);
  free(M);
  free(y);
  free(ystar);
  free(Mp);
  free(S);
  free(b);
  
  /* footer of the plugin */
  SF_display("\n");
  SF_display("End of the plugin\n");
  SF_display("====================\n");
  SF_display("\n");
  
  return(0);

}

/* TO BE ADDED: passing matrix B when save_mode = 1 */
ST_double mf_smap_single(ST_int rowsm, ST_int colsm, ST_double (*M)[colsm],\
			 ST_double b[], ST_double y[], ST_int l, ST_int theta,\
			 ST_double skip_obs, char *algorithm,\
			 ST_int save_index, ST_int force_compute,\
			 ST_int missingdistance)
{
  ST_double *a, *w;
  ST_double value, pre_adj_skip_obs, d_base, sumw, r;

  ST_int i, j, numind;
  
  char temps[500];

  d = (ST_double*)malloc(sizeof(ST_double)*rowsm);
  a = (ST_double*)malloc(sizeof(ST_double)*colsm);
  ind = (ST_int*)malloc(sizeof(ST_int)*rowsm);
  
  for (i=0; i<rowsm; i++) {
    value = 0.;
    for (j=0; j<colsm; j++) {
      a[j] = M[i][j] - b[j];
      if (missingdistance !=0) {

	/* TO BE ADDED: HANDLING OF MISSING VALUES */
	
      }
      value = value + a[j]*a[j]; 
    }
    d[i] = value;
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

    /* TO BE ADDED: HANDLING OF MISSING VALUES */
    
  }

  d_base = d[ind[(int)skip_obs]];

  if (numind < l+(int)skip_obs) {
    if (force_compute == 1) {
      l = numind - (int)skip_obs;
      if (l <= 0) {
	sprintf(temps,"Insufficient number of unique observations in the\
                       dataset even with -force- option\n");
        SF_error(temps);
        return( (ST_retcode)503);
      }
    } else {
      sprintf(temps,"Insufficient number of unique observations, consider\
                     tweaking the values of E, k or use -force- option\n");
      SF_error(temps);
      return( (ST_retcode)503);
    }
  }

  w = (ST_double*)malloc(sizeof(ST_double)*(l+(int)skip_obs));
  sumw = 0.;
  for (j=(int)skip_obs; j<l+(int)skip_obs; j++) {
    w[j] = exp(-(ST_double)theta*pow((d[ind[j]] / d_base),(0.5)));
    sumw = sumw + w[j];
  }
  for (j=(int)skip_obs; j<l+(int)skip_obs; j++) {
    w[j] = w[j]/sumw;
  }

  r = 0.;
  if ((strcmp(algorithm,"") == 0) || (strcmp(algorithm,"simplex") == 0)) {
    for(j=(int)skip_obs; j<l+(int)skip_obs; j++) {
      r = r + y[ind[j]] * w[j];
    }
    return(r);
  }
  
  /* deallocation of matrices and arrays before exiting the function */
  free(d);
  free(a);
  free(ind);
  free(w);

  /* returning the result to the main program */
  return b[1];
  
}

/* NOTE: in mata, minindex(v,k,i,w) returns in i and w the indices of the
   k minimums of v. The internal function minindex below only returns i
   and does not return w, as w is not used in the original edm code */
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
  count_ord = 0;
  i = 1;
  while ((contin < k) && (i < rvect)) {
    if (vect[ind[i]] != tempval) {
      tempval = vect[ind[i]];
      if (count_ord > 1) {
	/* here I reorder the indexes from low to high in case of
	   repeated values */
        temp_ind = (ST_double*)malloc(sizeof(ST_double)*count_ord);
	subind = (ST_int*)malloc(sizeof(ST_int)*count_ord);
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
	count_ord = 0;
      }
      contin++;
      numind++;
      count_ord++;
      i++;
    } else {
      numind++;
      count_ord++;
      i++;
      if (i == rvect) {
	/* here I chek if I reached the end of the array */
        if (count_ord > 1) {
	  /* here I reorder the indexes from low to high in case of
	     repeated values */
          temp_ind = (ST_double*)malloc(sizeof(ST_double)*count_ord);
	  subind = (ST_int*)malloc(sizeof(ST_int)*count_ord);
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
