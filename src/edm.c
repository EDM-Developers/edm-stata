/* version 2.1, 09 Sep 2020, Edoardo Tescari, Melbourne Data Analytics Platform,
   The University of Melbourne, e.tescari@unimelb.edu.au */

#include "edm.h"
#include <gsl/gsl_linalg.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* internal functions */

/* function that returns the sorted indices of an array */
static void quicksortind(double A[], int I[], int lo, int hi)
{
  while (lo < hi) {
    double pivot = A[I[lo + (hi - lo) / 2]];
    int t;
    int i = lo - 1;
    int j = hi + 1;
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
static int minindex(int rvect, double vect[], int k, int ind[])
{
  int i, j, contin, numind, count_ord, *subind;

  double tempval, *temp_ind;

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
        temp_ind = (double*)malloc(sizeof(double) * count_ord);
        subind = (int*)malloc(sizeof(int) * count_ord);
        if ((temp_ind == NULL) || (subind == NULL)) {
          return MALLOC_ERROR;
        }
        for (j = 0; j < count_ord; j++) {
          temp_ind[j] = (double)ind[i - 1 - j];
          subind[j] = j;
        }
        quicksortind(temp_ind, subind, 0, count_ord - 1);
        for (j = 0; j < count_ord; j++) {
          ind[i - 1 - j] = (int)temp_ind[subind[count_ord - 1 - j]];
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
          temp_ind = (double*)malloc(sizeof(double) * count_ord);
          subind = (int*)malloc(sizeof(int) * count_ord);
          if ((temp_ind == NULL) || (subind == NULL)) {
            return MALLOC_ERROR;
          }
          for (j = 0; j < count_ord; j++) {
            temp_ind[j] = (double)ind[i - 1 - j];
            subind[j] = j;
          }
          quicksortind(temp_ind, subind, 0, count_ord - 1);
          for (j = 0; j < count_ord; j++) {
            ind[i - 1 - j] = (int)temp_ind[subind[count_ord - 1 - j]];
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

static retcode mf_smap_single(const gsl_matrix* M, const gsl_vector* b, const double y[], int l, double theta,
                              int skip_obs, char* algorithm, bool save_mode, int varssv, bool force_compute,
                              double missingdistance, double* ystar, gsl_vector* Bi)
{
  bool missing;
  int i, j, numind;
  int* ind;
  double value, pre_adj_skip_obs, d_base, sumw, r;
  double *d, *a, *w;

  d = (double*)malloc(sizeof(double) * M->size1);
  a = (double*)malloc(sizeof(double) * M->size2);
  ind = (int*)malloc(sizeof(int) * M->size1);
  if ((d == NULL) || (a == NULL) || (ind == NULL)) {
    return MALLOC_ERROR;
  }

  for (i = 0; i < M->size1; i++) {
    value = 0.;
    missing = false;
    for (j = 0; j < M->size2; j++) {
      if ((gsl_matrix_get(M, i, j) == MISSING) || (gsl_vector_get(b, j) == MISSING)) {
        if (missingdistance != 0) {
          a[j] = missingdistance;
          value = value + a[j] * a[j];
        } else {
          missing = true;
          break;
        }
      } else {
        a[j] = gsl_matrix_get(M, i, j) - gsl_vector_get(b, j);
        value = value + a[j] * a[j];
      }
    }
    if (missing) {
      d[i] = MISSING;
    } else {
      d[i] = value;
    }
    ind[i] = i;
  }

  numind = minindex((int)M->size1, d, l + skip_obs, ind);

  pre_adj_skip_obs = skip_obs;

  for (j = 0; j < l; j++) {
    if (d[ind[j + skip_obs]] == 0.) {
      skip_obs++;
    } else {
      break;
    }
  }

  if (pre_adj_skip_obs != skip_obs) {
    numind = minindex((int)M->size1, d, l + skip_obs, ind);
  }

  if (d[ind[skip_obs]] == 0.) {
    for (i = 0; i < M->size1; i++) {
      if (d[i] == 0.) {
        d[i] = MISSING;
      }
    }
    skip_obs = 0;
    numind = minindex((int)M->size1, d, l + skip_obs, ind);
  }

  d_base = d[ind[skip_obs]];

  if (numind < l + skip_obs) {
    if (force_compute) {
      l = numind - skip_obs;
      if (l <= 0) {
        return INSUFFICIENT_UNIQUE;
      }
    } else {
      return INSUFFICIENT_UNIQUE;
    }
  }

  w = (double*)malloc(sizeof(double) * (l + skip_obs));
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
    bool anyMissing;
    gsl_matrix* X_ls;
    double mean_w, *y_ls, *w_ls;
    int rowc;
    X_ls = gsl_matrix_alloc(l, M->size2);
    y_ls = (double*)malloc(sizeof(double) * l);
    w_ls = (double*)malloc(sizeof(double) * l);
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
    mean_w = mean_w / (double)l;
    for (j = skip_obs; j < l + skip_obs; j++) {
      w[j] = exp(-theta * (w[j] / mean_w));
    }

    rowc = -1;
    for (j = skip_obs; j < l + skip_obs; j++) {
      if (y[ind[j]] == MISSING) {
        continue;
      }
      anyMissing = false;
      for (i = 0; i < M->size2; i++) {
        if (gsl_matrix_get(M, ind[j], i) == MISSING) {
          anyMissing = true;
          break;
        }
      }
      if (anyMissing) {
        continue;
      }
      rowc++;
      if (strcmp(algorithm, "llr") == 0) {
        /* llr algorithm is not needed at this stage */
        return NOT_IMPLEMENTED;

      } else if (strcmp(algorithm, "smap") == 0) {
        y_ls[rowc] = y[ind[j]] * w[j];
        w_ls[rowc] = w[j];
        for (i = 0; i < M->size2; i++) {
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

    // Pull out the first 'rowc+1' elements of the y_ls vector
    gsl_vector_const_view y_ls_cj_view = gsl_vector_const_view_array(y_ls, rowc + 1);
    const gsl_vector* y_ls_cj = &y_ls_cj_view.vector;

    // Concatenate the column vector 'w' with 'X_ls', keeping only
    // the first 'rowc+1' rows.
    gsl_matrix* X_ls_cj = gsl_matrix_alloc(rowc + 1, M->size2 + 1);
    for (i = 0; i < rowc + 1; i++) {
      gsl_matrix_set(X_ls_cj, i, 0, w_ls[i]);
      for (j = 1; j < X_ls->size2 + 1; j++) {
        gsl_matrix_set(X_ls_cj, i, j, gsl_matrix_get(X_ls, i, j - 1));
      }
    }

    if (strcmp(algorithm, "llr") == 0) {
      /* llr algorithm is not needed at this stage */
      return NOT_IMPLEMENTED;
    } else {
      gsl_vector* ics = gsl_vector_alloc(M->size2 + 1);

      /* singular value decomposition (SVD) of X_ls_cj, using gsl libraries */
      if (X_ls_cj->size1 >= X_ls_cj->size2) {
        gsl_matrix* V = gsl_matrix_alloc(M->size2 + 1, M->size2 + 1);
        gsl_vector* Esse = gsl_vector_alloc(M->size2 + 1);

        /* TO BE ADDED: benchmark which one of the following methods work best*/
        /*Golub-Reinsch SVD algorithm*/
        /*gsl_linalg_SV_decomp(X_ls_cj,V,Esse,ics);*/

        /* one-sided Jacobi orthogonalization method */
        gsl_linalg_SV_decomp_jacobi(X_ls_cj, V, Esse);

        /* setting to zero extremely small values of Esse to avoid
        underflow errors */
        for (j = 0; j < Esse->size; j++) {
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
      for (j = 1; j < M->size2 + 1; j++) {
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
      gsl_vector_free(ics);

      /* save the value of ystar[j] */
      *ystar = r;
      return SUCCESS;
    }
  }

  return INVALID_ALGORITHM;
}

/* OpenMP routines */

DLL retcode mf_smap_loop(int count_predict_set, int count_train_set, int mani, int Mpcol, double* flat_M,
                         double* flat_Mp, double* y, int l, double theta, double* S, char* algorithm, bool save_mode,
                         int varssv, bool force_compute, double missingdistance, double* ystar, double* flat_Bi_map)
{
  /* Create GSL matrixes which are views of the supplied flattened matrices */
  gsl_matrix_view M_view = gsl_matrix_view_array(flat_M, count_train_set, mani);
  gsl_matrix* M = &(M_view.matrix);

  gsl_matrix_view Mp_view = gsl_matrix_view_array(flat_Mp, count_predict_set, Mpcol);
  gsl_matrix* Mp = &(Mp_view.matrix);

  gsl_matrix_view Bi_map_view = gsl_matrix_view_array(flat_Bi_map, count_predict_set, varssv);
  gsl_matrix* Bi_map = &Bi_map_view.matrix;

  /* OpenMP loop with call to mf_smap_single function */
  retcode* rc = malloc(Mp->size1 * sizeof(retcode));
  int i;

#pragma omp parallel for
  for (i = 0; i < Mp->size1; i++) {

    gsl_vector_view Bi_view;
    gsl_vector* Bi = NULL;
    if (save_mode) {
      Bi_view = gsl_matrix_row(Bi_map, i);
      Bi = &Bi_view.vector;
    }

    gsl_vector_const_view Mpi_view = gsl_matrix_const_row(Mp, i);
    const gsl_vector* Mpi = &Mpi_view.vector;

    rc[i] = mf_smap_single(M, Mpi, y, l, theta, (int)S[i], algorithm, save_mode, varssv, force_compute, missingdistance,
                           &(ystar[i]), Bi);
  }

  /* Check if any mf_smap_single call failed, and if so find the most serious error */
  retcode maxError = 0;
  for (i = 0; i < Mp->size1; i++) {
    if (rc[i] > maxError) {
      maxError = rc[i];
    }
  }

  free(rc);

  return maxError;
}
