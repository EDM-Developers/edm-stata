/* version 2.1, 09 Sep 2020, Edoardo Tescari, Melbourne Data Analytics Platform,
   The University of Melbourne, e.tescari@unimelb.edu.au */

/* Suppress Windows problems with sprintf etc. functions. */
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include "edm.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* internal functions */

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
          return MALLOC_ERROR;
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
            return MALLOC_ERROR;
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

static ST_retcode mf_smap_single(ST_int rowsm, ST_int colsm, const gsl_matrix* M, const gsl_vector* b,
                                 const ST_double y[], ST_int l, ST_double theta, ST_int skip_obs, char* algorithm,
                                 ST_int save_mode, ST_int varssv, ST_int force_compute, ST_double missingdistance,
                                 ST_double* ystar, gsl_vector* Bi)
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
      if ((gsl_matrix_get(M, i, j) == MISSING) || (gsl_vector_get(b, j) == MISSING)) {
        if (missingdistance != 0) {
          a[j] = missingdistance;
          value = value + a[j] * a[j];
        } else {
          boolmiss = 1;
        }
      } else {
        a[j] = gsl_matrix_get(M, i, j) - gsl_vector_get(b, j);
        value = value + a[j] * a[j];
      }
    }
    if (boolmiss == 1) {
      d[i] = MISSING;
    } else {
      d[i] = value;
    }
    ind[i] = i;
  }

  numind = minindex(rowsm, d, l + skip_obs, ind);

  pre_adj_skip_obs = skip_obs;

  for (j = 0; j < l; j++) {
    if (d[ind[j + skip_obs]] == 0.) {
      skip_obs++;
    } else {
      break;
    }
  }

  if (pre_adj_skip_obs != skip_obs) {
    numind = minindex(rowsm, d, l + skip_obs, ind);
  }

  if (d[ind[skip_obs]] == 0.) {
    for (i = 0; i < rowsm; i++) {
      if (d[i] == 0.) {
        d[i] = MISSING;
      }
    }
    skip_obs = 0;
    numind = minindex(rowsm, d, l + skip_obs, ind);
  }

  d_base = d[ind[skip_obs]];

  if (numind < l + skip_obs) {
    if (force_compute == 1) {
      l = numind - skip_obs;
      if (l <= 0) {
        return INSUFFICIENT_UNIQUE;
      }
    } else {
      return INSUFFICIENT_UNIQUE;
    }
  }

  w = (ST_double*)malloc(sizeof(ST_double) * (l + skip_obs));
  if (w == NULL) {
    return MALLOC_ERROR;
  }

  sumw = 0.;
  r = 0.;
  if ((strcmp(algorithm, "") == 0) || (strcmp(algorithm, "simplex") == 0)) {
    for (j = skip_obs; j < l + skip_obs; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = exp(-theta*pow((d[ind[j]] / d_base),(0.5))); */
      w[j] = exp(-theta * sqrt(d[ind[j]] / d_base));
      sumw = sumw + w[j];
    }
    for (j = skip_obs; j < l + skip_obs; j++) {
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

    gsl_matrix* X_ls;
    ST_double mean_w, *y_ls, *w_ls;
    ST_int rowc, bocont;
    X_ls = gsl_matrix_alloc(l, colsm);
    y_ls = (ST_double*)malloc(sizeof(ST_double) * l);
    w_ls = (ST_double*)malloc(sizeof(ST_double) * l);
    if ((y_ls == NULL) || (w_ls == NULL)) {
      return MALLOC_ERROR;
    }

    mean_w = 0.;
    for (j = skip_obs; j < l + skip_obs; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = pow(d[ind[j]],0.5); */
      w[j] = sqrt(d[ind[j]]);
      mean_w = mean_w + w[j];
    }
    mean_w = mean_w / (ST_double)l;
    for (j = skip_obs; j < l + skip_obs; j++) {
      w[j] = exp(-theta * (w[j] / mean_w));
    }

    rowc = -1;
    for (j = skip_obs; j < l + skip_obs; j++) {
      if (y[ind[j]] == MISSING) {
        continue;
      }
      bocont = 0;
      for (i = 0; i < colsm; i++) {
        if (gsl_matrix_get(M, ind[j], i) == MISSING) {
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
          gsl_matrix_set(X_ls, rowc, i, gsl_matrix_get(M, ind[j], i) * w[j]);
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
      gsl_matrix_free(X_ls);

      /* save the missing value flag to ystar[j] */
      *ystar = MISSING;
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
        gsl_matrix_set(X_ls_cj, i, j, gsl_matrix_get(X_ls, i, j - 1));
      }
    }

    if (strcmp(algorithm, "llr") == 0) {
      /* llr algorithm is not needed at this stage */
      return NOT_IMPLEMENTED;
    } else {
      gsl_vector* ics = gsl_vector_alloc(colsm + 1);

      /* singular value decomposition (SVD) of X_ls_cj, using gsl libraries */
      if (X_ls_cj->size1 >= X_ls_cj->size2) {
        gsl_matrix* V = gsl_matrix_alloc(colsm + 1, colsm + 1);
        gsl_vector* Esse = gsl_vector_alloc(colsm + 1);

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

        gsl_matrix_free(V);
        gsl_vector_free(Esse);
      } else {
        // X_ls_cj is underdetermined (less rows than columns) so find the
        // least-squares solution using an LQ decomposition.
        gsl_vector* tau = gsl_vector_alloc(X_ls_cj->size1);

        // First, find the LQ decomposition of X_ls_cj in-place.
        gsl_linalg_LQ_decomp(X_ls_cj, tau);

        gsl_vector* residuals = gsl_vector_alloc(y_ls_cj->size);
        gsl_linalg_LQ_lssolve(X_ls_cj, tau, y_ls_cj, ics, residuals);

        gsl_vector_free(tau);
        gsl_vector_free(residuals);
      }

      /* saving ics coefficients if savesmap option enabled */
      if (save_mode) {
        for (j = 0; j < varssv; j++) {
          if (gsl_vector_get(ics, j) == 0.) {
            gsl_vector_set(Bi, j, MISSING);
          } else {
            gsl_vector_set(Bi, j, gsl_vector_get(ics, j));
          }
        }
      }

      r = gsl_vector_get(ics, 0);
      for (j = 1; j < colsm + 1; j++) {
        if (gsl_vector_get(b, j - 1) != MISSING) {
          r += gsl_vector_get(b, j - 1) * gsl_vector_get(ics, j);
        }
      }

      /* deallocation of matrices and arrays before exiting the function */
      free(d);
      free(a);
      free(ind);
      free(w);
      free(y_ls);
      free(w_ls);
      gsl_matrix_free(X_ls);
      gsl_matrix_free(X_ls_cj);
      gsl_vector_free(y_ls_cj);
      gsl_vector_free(ics);

      /* save the value of ystar[j] */
      *ystar = r;
      return SUCCESS;
    }
  }

  return INVALID_ALGORITHM;
}

/* OpenMP routines */
DLL ST_retcode mf_smap_loop(ST_int count_predict_set, ST_int count_train_set, ST_int mani, gsl_matrix* M,
                            gsl_matrix* Mp, ST_double* y, ST_int l, ST_double theta, ST_double* S, char* algorithm,
                            ST_int save_mode, ST_int varssv, ST_int force_compute, ST_double missingdistance,
                            ST_double* ystar, gsl_matrix* Bi_map)
{

  /* OpenMP loop with call to mf_smap_single function */
  ST_retcode* rc = malloc(count_predict_set * sizeof(ST_retcode));
  ST_int i;

#pragma omp parallel for
  for (i = 0; i < count_predict_set; i++) {

    gsl_vector_view Bi_view;
    gsl_vector* Bi = NULL;
    if (save_mode) {
      Bi_view = gsl_matrix_row(Bi_map, i);
      Bi = &Bi_view.vector;
    }

    gsl_vector_const_view Mpi_view = gsl_matrix_const_row(Mp, i);
    const gsl_vector* Mpi = &Mpi_view.vector;

    rc[i] = mf_smap_single(count_train_set, mani, M, Mpi, y, l, theta, (int) S[i], algorithm, save_mode, varssv,
                           force_compute, missingdistance, &(ystar[i]), Bi);
  }

  /* Check if any mf_smap_single call failed, and if so find the most serious error */
  ST_retcode maxError = 0;
  for (i = 0; i < count_predict_set; i++) {
    if (rc[i] > maxError) {
      maxError = rc[i];
    }
  }

  free(rc);

  return maxError;
}
