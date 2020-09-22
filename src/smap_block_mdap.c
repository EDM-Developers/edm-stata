/* version 2.1, 09 Sep 2020, Edoardo Tescari, Melbourne Data Analytics Platform,
   The University of Melbourne, e.tescari@unimelb.edu.au */

/* Suppress Windows problems with sprintf etc. functions. */
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#define SUCCESS 0
#define INVALID_ALGORITHM 400
#define INSUFFICIENT_UNIQUE 503
#define NOT_IMPLEMENTED 908
#define MALLOC_ERROR 909

#include "stplugin.h"

#include <gsl/gsl_linalg.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* global variable placeholder for missing values */

ST_double missval = 1.0e+100;

/* internal functions */

ST_retcode print_error(ST_retcode rc)
{
  char temps[500];

  switch (rc) {
    case MALLOC_ERROR:
      sprintf(temps, "Insufficient memory\n");
      break;
    case NOT_IMPLEMENTED:
      sprintf(temps, "Method is not yet implemented\n");
      break;
    case INSUFFICIENT_UNIQUE:
      sprintf(temps, "Insufficient number of unique observations, consider "
                     "tweaking the values of E, k or use -force- option\n");
      break;
    case INVALID_ALGORITHM:
      sprintf(temps, "Invalid algorithm argument\n");
      break;
  }

  if (rc != SUCCESS) {
    SF_error(temps);
  }

  return rc;
}

static void free_matrix(ST_double** M, ST_int nrow)
{
  if (M != NULL) {
    for (ST_int i = 0; i < nrow; i++) {
      if (M[i] != NULL) {
        free(M[i]);
      }
    }
    free(M);
  }
}

static ST_double** alloc_matrix(ST_int nrow, ST_int ncol)
{
  if (nrow == 0 || ncol == 0) {
    return NULL;
  }

  ST_double** M = calloc(nrow, sizeof(ST_double*));
  if (M != NULL) {
    for (ST_int i = 0; i < nrow; i++) {
      M[i] = malloc(ncol * (sizeof(ST_double)));
      if (M[i] == NULL) {
        free_matrix(M, nrow);
        return NULL;
      }
    }
  }
  return M;
}

/* function that returns the sorted indices of an array */
static void quicksortind(ST_double A[], ST_int I[], ST_int lo, ST_int hi)
{
  while (lo < hi) {
    ST_double pivot = A[I[lo + (hi - lo) / 2]];
    ST_int t;
    ST_int i = lo - 1;
    ST_int j = hi + 1;
    while (1) {
      while (A[I[++i]] < pivot) {};
      while (A[I[--j]] > pivot) {};
      if (i >= j)
        break;
      t = I[i];
      I[i] = I[j];
      I[j] = t;
    }
    /* avoid stack overflow */
    if ((j - lo) < (hi - j)) {
      quicksortind(A, I, lo, j);
      lo = j + 1;
    } else {
      quicksortind(A, I, j + 1, hi);
      hi = j;
    }
  }
}

/* NOTE: in mata, minindex(v,k,i,w) returns in i and w the indices of the
   k minimums of v. The internal function minindex below only returns i +
   the number of k minimums and does not return w, as w is not used in the
   original edm code */
static ST_int minindex(ST_int rvect, ST_double vect[], ST_int k, ST_int ind[])
{
  ST_int i, j, contin, numind, count_ord, *subind;

  ST_double tempval, *temp_ind;

  quicksortind(vect, ind, 0, rvect - 1);

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
        temp_ind = (ST_double*)malloc(sizeof(ST_double) * count_ord);
        subind = (ST_int*)malloc(sizeof(ST_int) * count_ord);
        if ((temp_ind == NULL) || (subind == NULL)) {
          return print_error(MALLOC_ERROR);
        }
        for (j = 0; j < count_ord; j++) {
          temp_ind[j] = (ST_double)ind[i - 1 - j];
          subind[j] = j;
        }
        quicksortind(temp_ind, subind, 0, count_ord - 1);
        for (j = 0; j < count_ord; j++) {
          ind[i - 1 - j] = (ST_int)temp_ind[subind[count_ord - 1 - j]];
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
          temp_ind = (ST_double*)malloc(sizeof(ST_double) * count_ord);
          subind = (ST_int*)malloc(sizeof(ST_int) * count_ord);
          if ((temp_ind == NULL) || (subind == NULL)) {
            return print_error(MALLOC_ERROR);
          }
          for (j = 0; j < count_ord; j++) {
            temp_ind[j] = (ST_double)ind[i - 1 - j];
            subind[j] = j;
          }
          quicksortind(temp_ind, subind, 0, count_ord - 1);
          for (j = 0; j < count_ord; j++) {
            ind[i - 1 - j] = (ST_int)temp_ind[subind[count_ord - 1 - j]];
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

static ST_retcode mf_smap_single(ST_int rowsm, ST_int colsm, ST_double** M, ST_double b[], ST_double y[], ST_int l,
                                 ST_double theta, ST_double skip_obs, char* algorithm, ST_int save_mode, ST_double Bi[],
                                 ST_int varssv, ST_int force_compute, ST_double missingdistance, ST_double* ystar)
{
  ST_double *d, *a, *w;
  ST_int* ind;

  ST_double value, pre_adj_skip_obs, d_base, sumw, r;

  ST_int i, j, numind, boolmiss;

  d = (ST_double*)malloc(sizeof(ST_double) * rowsm);
  a = (ST_double*)malloc(sizeof(ST_double) * colsm);
  ind = (ST_int*)malloc(sizeof(ST_int) * rowsm);
  if ((d == NULL) || (a == NULL) || (ind == NULL)) {
    return MALLOC_ERROR;
  }

  for (i = 0; i < rowsm; i++) {
    value = 0.;
    boolmiss = 0;
    for (j = 0; j < colsm; j++) {
      if ((M[i][j] == missval) || (b[j] == missval)) {
        if (missingdistance != 0) {
          a[j] = missingdistance;
          value = value + a[j] * a[j];
        } else {
          boolmiss = 1;
        }
      } else {
        a[j] = M[i][j] - b[j];
        value = value + a[j] * a[j];
      }
    }
    if (boolmiss == 1) {
      d[i] = missval;
    } else {
      d[i] = value;
    }
    ind[i] = i;
  }

  numind = minindex(rowsm, d, l + (int)skip_obs, ind);

  pre_adj_skip_obs = skip_obs;

  for (j = 0; j < l; j++) {
    if (d[ind[j + (int)skip_obs]] == 0.) {
      skip_obs++;
    } else {
      break;
    }
  }

  if (pre_adj_skip_obs != skip_obs) {
    numind = minindex(rowsm, d, l + (int)skip_obs, ind);
  }

  if (d[ind[(int)skip_obs]] == 0.) {
    for (i = 0; i < rowsm; i++) {
      if (d[i] == 0.) {
        d[i] = missval;
      }
    }
    skip_obs = 0.;
    numind = minindex(rowsm, d, l + (int)skip_obs, ind);
  }

  d_base = d[ind[(int)skip_obs]];

  if (numind < l + (int)skip_obs) {
    if (force_compute == 1) {
      l = numind - (int)skip_obs;
      if (l <= 0) {
        return INSUFFICIENT_UNIQUE;
      }
    } else {
      return INSUFFICIENT_UNIQUE;
    }
  }

  w = (ST_double*)malloc(sizeof(ST_double) * (l + (int)skip_obs));
  if (w == NULL) {
    return MALLOC_ERROR;
  }

  sumw = 0.;
  r = 0.;
  if ((strcmp(algorithm, "") == 0) || (strcmp(algorithm, "simplex") == 0)) {
    for (j = (int)skip_obs; j < l + (int)skip_obs; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = exp(-theta*pow((d[ind[j]] / d_base),(0.5))); */
      w[j] = exp(-theta * sqrt(d[ind[j]] / d_base));
      sumw = sumw + w[j];
    }
    for (j = (int)skip_obs; j < l + (int)skip_obs; j++) {
      r = r + y[ind[j]] * (w[j] / sumw);
    }
    /* deallocation of matrices and arrays before exiting the function */
    free(d);
    free(a);
    free(ind);
    free(w);

    /* save the value of ystar[j] */
    *ystar = r;
    return SUCCESS;

  } else if ((strcmp(algorithm, "smap") == 0) || (strcmp(algorithm, "llr") == 0)) {

    ST_double mean_w, *y_ls, *w_ls, **X_ls;
    ST_int rowc, bocont;

    y_ls = (ST_double*)malloc(sizeof(ST_double) * l);
    w_ls = (ST_double*)malloc(sizeof(ST_double) * l);
    X_ls = alloc_matrix(l, colsm);
    if ((y_ls == NULL) || (w_ls == NULL) || (X_ls == NULL)) {
      return MALLOC_ERROR;
    }

    mean_w = 0.;
    for (j = (int)skip_obs; j < l + (int)skip_obs; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = pow(d[ind[j]],0.5); */
      w[j] = sqrt(d[ind[j]]);
      mean_w = mean_w + w[j];
    }
    mean_w = mean_w / (ST_double)l;
    for (j = (int)skip_obs; j < l + (int)skip_obs; j++) {
      w[j] = exp(-theta * (w[j] / mean_w));
    }

    rowc = -1;
    for (j = (int)skip_obs; j < l + (int)skip_obs; j++) {
      if (y[ind[j]] == missval) {
        continue;
      }
      bocont = 0;
      for (i = 0; i < colsm; i++) {
        if (M[ind[j]][i] == missval) {
          bocont = 1;
        }
      }
      if (bocont == 1) {
        continue;
      }
      rowc++;
      if (strcmp(algorithm, "llr") == 0) {
        /* llr algorithm is not needed at this stage */
        return NOT_IMPLEMENTED;

      } else if (strcmp(algorithm, "smap") == 0) {
        y_ls[rowc] = y[ind[j]] * w[j];
        w_ls[rowc] = w[j];
        for (i = 0; i < colsm; i++) {
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
      free_matrix(X_ls, l);

      /* save the missing value flag to ystar[j] */
      *ystar = missval;
      return SUCCESS;
    }

    gsl_matrix* X_ls_cj = gsl_matrix_alloc(rowc + 1, colsm + 1);
    gsl_vector* y_ls_cj = gsl_vector_alloc(rowc + 1);
    if ((X_ls_cj == NULL) || (y_ls_cj == NULL)) {
      return MALLOC_ERROR;
    }
    for (i = 0; i < rowc + 1; i++) {
      gsl_matrix_set(X_ls_cj, i, 0, w_ls[i]);
      gsl_vector_set(y_ls_cj, i, y_ls[i]);
      for (j = 1; j < colsm + 1; j++) {
        gsl_matrix_set(X_ls_cj, i, j, X_ls[i][j - 1]);
      }
    }

    if (strcmp(algorithm, "llr") == 0) {
      /* llr algorithm is not needed at this stage */
      return NOT_IMPLEMENTED;
    } else {
      gsl_matrix* V = gsl_matrix_alloc(colsm + 1, colsm + 1);
      gsl_vector* Esse = gsl_vector_alloc(colsm + 1);
      gsl_vector* ics = gsl_vector_alloc(colsm + 1);
      if ((V == NULL) || (Esse == NULL) || (ics == NULL)) {
        return MALLOC_ERROR;
      }

      /* singular value decomposition (SVD) of X_ls_cj, using gsl libraries */
      if (rowc + 1 < colsm + 1) {
        /* GSL's SVD crashes for one kind of rectangular matrices */
        return NOT_IMPLEMENTED;
      }

      /* TO BE ADDED: benchmark which one of the following methods work best*/
      /*Golub-Reinsch SVD algorithm*/
      /*gsl_linalg_SV_decomp(X_ls_cj,V,Esse,ics);*/

      /* one-sided Jacobi orthogonalization method */
      gsl_linalg_SV_decomp_jacobi(X_ls_cj, V, Esse);

      /* setting to zero extremely small values of Esse to avoid
         underflow errors */
      for (j = 0; j < colsm + 1; j++) {
        if (gsl_vector_get(Esse, j) < 1.0e-12) {
          gsl_vector_set(Esse, j, 0.);
        }
      }

      /* function to solve X_ls_cj * ics = y_ls_cj and return ics,
         using gsl libraries */
      gsl_linalg_SV_solve(X_ls_cj, V, Esse, y_ls_cj, ics);

      /* saving ics coefficients if savesmap option enabled */
      if (save_mode) {
        for (j = 0; j < varssv; j++) {
          if (gsl_vector_get(ics, j) == 0.) {
            Bi[j] = missval;
          } else {
            Bi[j] = gsl_vector_get(ics, j);
          }
        }
      }

      r = gsl_vector_get(ics, 0);
      for (j = 1; j < colsm + 1; j++) {
        if (b[j - 1] == missval) {
          b[j - 1] = 0.;
        }
        r = r + b[j - 1] * gsl_vector_get(ics, j);
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

      /* save the value of ystar[j] */
      *ystar = r;
      return SUCCESS;
    }
  }

  return INVALID_ALGORITHM;
}

/* OpenMP routines */

ST_retcode mf_smap_loop(ST_int count_predict_set, ST_int count_train_set, ST_double** Bi_map, ST_int mani,
                        ST_double** M, ST_double** Mp, ST_double* y, ST_int l, ST_double theta, ST_double* S,
                        char* algorithm, ST_int save_mode, ST_int varssv, ST_int force_compute, ST_double missingdistance,
                        ST_double* ystar)
{

  /* OpenMP loop with call to mf_smap_single function */
  ST_retcode* rc = malloc(count_predict_set * sizeof(ST_retcode));
  ST_int i;

#pragma omp parallel for
  for (i = 0; i < count_predict_set; i++) {
    ST_double* Bi = save_mode ? Bi_map[i] : NULL;
    rc[i] = mf_smap_single(count_train_set, mani, M, Mp[i], y, l, theta, S[i], algorithm, save_mode, Bi, varssv,
                           force_compute, missingdistance, &(ystar[i]));
  }

  /* Check if any mf_smap_single call failed, and if so find the most serious error */
  ST_retcode maxError = 0;
  for (i = 0; i < count_predict_set; i++) {
    if (rc[i] > maxError) {
      maxError = rc[i];
    }
  }

  free(rc);

  return print_error(maxError);
}

/*
Example call to the plugin:

local myvars ``manifold'' `co_mapping' `x_f' `x_p' `train_set' `predict_set' `overlap' `vars_save'

unab vars : ``manifold''
local mani `: word count `vars''

local pmani_flag = 0

local vsave_flag = 0

plugin call smap_block_mdap `myvars', `j' `lib_size' "`algorithm'" "`force'" `missingdistance' `mani' `pmani_flag'
`vsave_flag'
*/

STDLL stata_call(int argc, char* argv[])
{
  ST_int nvars, nobs, first, last, mani, pmani_flag, pmani, smaploc;
  ST_int Mpcol, l, vsave_flag, save_mode, varssv;

  ST_double value, theta, missingdistance, *train_use, *predict_use, *skip_obs;
  ST_double **M, **Mp, **Bi_map;
  ST_double *y, *S, *ystar;
  ST_int i, j, h, force_compute, nthreads;
  ST_int count_train_set, count_predict_set, obsi, obsj;

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
  last = SF_in2();
  sprintf(temps, "number of vars & obs = %i, %i\n", nvars, nobs);
  SF_display(temps);
  sprintf(temps, "first and last obs in sample = %i, %i\n", first, last);
  SF_display(temps);
  SF_display("\n");

  for (i = 0; i < argc; i++) {
    sprintf(temps, "arg %i: ", i);
    SF_display(temps);
    SF_display(argv[i]);
    SF_display("\n");
  }
  SF_display("\n");

  theta = atof(argv[0]); /* contains value of theta = first argument */
  sprintf(temps, "theta = %6.4f\n", theta);
  SF_display(temps);
  SF_display("\n");

  /* allocation of string variable algorithm based on third argument */
  sprintf(algorithm, "%s", argv[2]);
  sprintf(temps, "algorithm = %s\n", algorithm);
  SF_display(temps);
  SF_display("\n");

  /* allocation of variable force_compute based on fourth argument */
  if (strcmp(argv[3], "force") == 0)
    force_compute = 1;
  else
    force_compute = 0;
  sprintf(temps, "force compute = %i\n", force_compute);
  SF_display(temps);
  SF_display("\n");

  /* allocation of variable missingdistance based on fifth argument */
  missingdistance = atof(argv[4]);
  sprintf(temps, "missing distance = %f\n", missingdistance);
  SF_display(temps);
  SF_display("\n");

  /* allocation of number of columns in manifold */
  mani = atoi(argv[5]);
  sprintf(temps, "number of variables in manifold = %i \n", mani);
  SF_display(temps);
  SF_display("\n");

  /* allocation of train_use, predict_use and skip_obs variables */
  train_use = (ST_double*)malloc(sizeof(ST_double) * nobs);
  predict_use = (ST_double*)malloc(sizeof(ST_double) * nobs);
  skip_obs = (ST_double*)malloc(sizeof(ST_double) * nobs);
  if ((train_use == NULL) || (predict_use == NULL) || (skip_obs == NULL)) {
    return print_error(MALLOC_ERROR);
  }

  count_train_set = 0;
  count_predict_set = 0;
  for (i = 1; i <= (last - first + 1); i++) {
    SF_vdata(mani + 3, i, &value);
    train_use[i - 1] = value;
    if (value == 1.)
      count_train_set++;
    if (SF_is_missing(value)) {
      /* missing value */
      train_use[i - 1] = missval;
    }
    SF_vdata(mani + 4, i, &value);
    predict_use[i - 1] = value;
    if (value == 1.)
      count_predict_set++;
    if (SF_is_missing(value)) {
      /* missing value */
      predict_use[i - 1] = missval;
    }
    SF_vdata(mani + 5, i, &value);
    skip_obs[i - 1] = value;
    if (SF_is_missing(value)) {
      /* missing value */
      skip_obs[i - 1] = missval;
    }
  }
  sprintf(temps, "train set obs: %i\n", count_train_set);
  SF_display(temps);
  sprintf(temps, "predict set obs: %i\n", count_predict_set);
  SF_display(temps);
  SF_display("\n");

  /* allocation of matrices M and y */
  M = alloc_matrix(count_train_set, mani);
  y = (ST_double*)malloc(sizeof(ST_double) * count_train_set);
  if ((M == NULL) || (y == NULL)) {
    return print_error(MALLOC_ERROR);
  }

  obsi = 0;
  for (i = 0; i < nobs; i++) {
    if (train_use[i] == 1.) {
      for (j = 0; j < mani; j++) {
        SF_vdata(j + 1, i + 1, &value);
        M[obsi][j] = value;
        if (SF_is_missing(value)) {
          /* missing value */
          M[obsi][j] = missval;
        }
      }
      SF_vdata(j + 1, i + 1, &value);
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
  sprintf(temps, "p_manifold flag = %i \n", pmani_flag);
  SF_display(temps);

  if (pmani_flag == 1) {
    pmani = atoi(argv[8]); /* contains the number of columns in p_manifold */
    sprintf(temps, "number of variables in p_manifold = %i \n", pmani);
    SF_display(temps);
    SF_display("\n");
    Mpcol = pmani;
  } else {
    SF_display("\n");
    Mpcol = mani;
  }
  Mp = alloc_matrix(count_predict_set, Mpcol);
  S = (ST_double*)malloc(sizeof(ST_double) * count_predict_set);
  if ((Mp == NULL) || (S == NULL)) {
    return print_error(MALLOC_ERROR);
  }

  if (pmani_flag == 1) {
    smaploc = mani + 5 + Mpcol + 1;
    obsi = 0;
    for (i = 0; i < nobs; i++) {
      if (predict_use[i] == 1.) {
        obsj = 0;
        for (j = mani + 5; j < mani + 5 + Mpcol; j++) {
          SF_vdata(j + 1, i + 1, &value);
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
    smaploc = mani + 5 + 1;
    obsi = 0;
    for (i = 0; i < nobs; i++) {
      if (predict_use[i] == 1.) {
        for (j = 0; j < Mpcol; j++) {
          SF_vdata(j + 1, i + 1, &value);
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
  sprintf(temps, "l = %i \n", l);
  SF_display(temps);
  SF_display("\n");

  vsave_flag = atoi(argv[7]); /* contains the flag for vars_save */

  if (vsave_flag == 1) { /* flag savesmap is ON */
    save_mode = 1;
    varssv = atoi(argv[8]); /* contains the number of columns
                               in smap coefficents */
    Bi_map = alloc_matrix(count_predict_set, varssv);
    if (Bi_map == NULL) {
      return print_error(MALLOC_ERROR);
    }
    sprintf(temps, "columns in smap coefficents = %i \n", varssv);
    SF_display(temps);
  } else { /* flag savesmap is OFF */
    save_mode = 0;
    Bi_map = NULL;
    varssv = 0;
  }

  sprintf(temps, "save_mode = %i \n", save_mode);
  SF_display(temps);
  SF_display("\n");

  ystar = (ST_double*)malloc(sizeof(ST_double) * count_predict_set);
  if (ystar == NULL) {
    return print_error(MALLOC_ERROR);
  }

  /* setting the number of OpenMP threads */
  nthreads = atoi(argv[9]);
  sprintf(temps, "Requested %i OpenMP threads \n", nthreads);
  SF_display(temps);
  nthreads = nthreads <= 0 ? omp_get_num_procs() : nthreads;
  sprintf(temps, "Using %i OpenMP threads \n", nthreads);
  SF_display(temps);
  SF_display("\n");

  omp_set_num_threads(nthreads);

  ST_retcode maxError = mf_smap_loop(count_predict_set, count_train_set, Bi_map, mani, M, Mp, y, l, theta, S, algorithm,
                                     save_mode, varssv, force_compute, missingdistance, ystar);

  /* If there are no errors, return the value of ystar (and smap coefficients) to Stata */
  if (maxError == 0) {
    j = 0;
    for (i = 0; i < nobs; i++) {
      if (predict_use[i] == 1) {
        if (ystar[j] != missval) {
          SF_vstore(mani + 2, i + 1, ystar[j]);
        } else {
          /* returning a missing value */
          SF_vstore(mani + 2, i + 1, SV_missval);
        }
        if (save_mode) {
          for (h = 0; h < varssv; h++) {
            if (Bi_map[j][h] != missval) {
              SF_vstore(smaploc + h, i + 1, Bi_map[j][h]);
            } else {
              SF_vstore(smaploc + h, i + 1, SV_missval);
            }
          }
        }
        j++;
      }
    }
  }

  /* deallocation of matrices and arrays before exiting the plugin */
  free(train_use);
  free(predict_use);
  free(skip_obs);
  free_matrix(M, count_train_set);
  free(y);
  free_matrix(Mp, count_predict_set);
  free(S);
  if (save_mode) {
    free_matrix(Bi_map, count_predict_set);
  }
  free(ystar);

  /* footer of the plugin */
  SF_display("\n");
  SF_display("End of the plugin\n");
  SF_display("====================\n");
  SF_display("\n");

  return (maxError);
}
