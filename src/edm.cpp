/* version 2.1, 09 Sep 2020, Edoardo Tescari, Melbourne Data Analytics Platform,
   The University of Melbourne, e.tescari@unimelb.edu.au */

#include "edm.h"
#include <gsl/gsl_linalg.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <Eigen/SVD>
#include <algorithm> // std::partial_sort
#include <numeric>   // std::iota
#include <vector>

/* internal functions */

/* minindex(v,k) returns the indices of the k minimums of v.  */
std::vector<size_t> minindex(const std::vector<double>& v, int k)
{
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  if (k >= (int)v.size()) {
    k = (int)v.size();
  }

  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

  return idx;
}

retcode mf_smap_single(const gsl_matrix* M, const gsl_vector* b, const double y[], int l, double theta, int skip_obs,
                       char* algorithm, bool save_mode, int varssv, bool force_compute, double missingdistance,
                       double* ystar, gsl_vector* Bi)
{

  int i, j;
  double d_base, sumw, r;
  double* w;
  auto d = std::vector<double>(M->size1, 0.0);

  double diff;
  int validDistances = 0;
  for (i = 0; i < M->size1; i++) {
    double dist = 0.;
    bool missing = false;
    int numMissingDims = 0;
    for (j = 0; j < M->size2; j++) {
      if ((gsl_matrix_get(M, i, j) == MISSING) || (gsl_vector_get(b, j) == MISSING)) {
        if (missingdistance == 0) {
          missing = true;
          break;
        }
        numMissingDims += 1;
      } else {
        diff = gsl_matrix_get(M, i, j) - gsl_vector_get(b, j);
        dist = dist + diff * diff;
      }
    }
    // If the distance between M_i and b is 0 before handling missing values,
    // then keep it at 0. Otherwise, add in the correct number of missingdistance's.
    if (dist != 0) {
      dist += numMissingDims * missingdistance * missingdistance;
    }

    if (missing || dist == 0.) {
      d[i] = MISSING;
    } else {
      d[i] = dist;
      validDistances += 1;
    }
  }

  // If we only look at distances which are non-zero and non-missing,
  // do we have enough of them to find 'l' neighbours?
  if (l > validDistances) {
    if (force_compute) {
      l = validDistances;
      if (l == 0) {
        return INSUFFICIENT_UNIQUE;
      }
    } else {
      return INSUFFICIENT_UNIQUE;
    }
  }

  std::vector<size_t> ind = minindex(d, l);

  d_base = d[ind[0]];
  w = (double*)malloc(sizeof(double) * l);
  if (w == NULL) {
    return MALLOC_ERROR;
  }

  sumw = 0.;
  r = 0.;
  if ((strcmp(algorithm, "") == 0) || (strcmp(algorithm, "simplex") == 0)) {
    for (j = 0; j < l; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = exp(-theta*pow((d[ind[j]] / d_base),(0.5))); */
      w[j] = exp(-theta * sqrt(d[ind[j]] / d_base));
      sumw = sumw + w[j];
    }
    for (j = 0; j < l; j++) {
      r = r + y[ind[j]] * (w[j] / sumw);
    }
    /* deallocation of matrices and arrays before exiting the function */
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
    for (j = 0; j < l; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = pow(d[ind[j]],0.5); */
      w[j] = sqrt(d[ind[j]]);
      mean_w = mean_w + w[j];
    }
    mean_w = mean_w / (double)l;
    for (j = 0; j < l; j++) {
      w[j] = exp(-theta * (w[j] / mean_w));
    }

    rowc = -1;
    for (j = 0; j < l; j++) {
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
      free(w);
      free(y_ls);
      free(w_ls);
      gsl_matrix_free(X_ls);

      /* save the missing value flag to ystar[j] */
      *ystar = MISSING;
      return SUCCESS;
    }

    // Pull out the first 'rowc+1' elements of the y_ls vector and
    // concatenate the column vector 'w' with 'X_ls', keeping only
    // the first 'rowc+1' rows.
    Eigen::VectorXd y_ls_cj(rowc + 1);
    Eigen::MatrixXd X_ls_cj(rowc + 1, M->size2 + 1);

    for (i = 0; i < rowc + 1; i++) {
      y_ls_cj(i) = y_ls[i];
      X_ls_cj(i, 0) = w_ls[i];
      for (j = 1; j < X_ls->size2 + 1; j++) {
        X_ls_cj(i, j) = gsl_matrix_get(X_ls, i, j - 1);
      }
    }

    if (strcmp(algorithm, "llr") == 0) {
      /* llr algorithm is not needed at this stage */
      return NOT_IMPLEMENTED;
    } else {
      Eigen::BDCSVD<Eigen::MatrixXd> svd(X_ls_cj, Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::VectorXd ics = svd.solve(y_ls_cj);

      /* saving ics coefficients if savesmap option enabled */
      if (save_mode) {
        for (j = 0; j < varssv; j++) {
          if (ics(j) == 0.) {
            gsl_vector_set(Bi, j, MISSING);
          } else {
            gsl_vector_set(Bi, j, ics(j));
          }
        }
      }

      r = ics(0);
      for (j = 1; j < M->size2 + 1; j++) {
        if (gsl_vector_get(b, j - 1) != MISSING) {
          r += gsl_vector_get(b, j - 1) * ics(j);
        }
      }

      /* deallocation of matrices and arrays before exiting the function */
      free(w);
      free(y_ls);
      free(w_ls);
      gsl_matrix_free(X_ls);

      /* save the value of ystar[j] */
      *ystar = r;
      return SUCCESS;
    }
  }

  return INVALID_ALGORITHM;
}

/* OpenMP routines */

retcode mf_smap_loop(int count_predict_set, int count_train_set, int mani, int Mpcol, double* flat_M, double* flat_Mp,
                     double* y, int l, double theta, double* S, char* algorithm, bool save_mode, int varssv,
                     bool force_compute, double missingdistance, double* ystar, double* flat_Bi_map)
{
  /* Create GSL matrixes which are views of the supplied flattened matrices */
  gsl_matrix_view M_view = gsl_matrix_view_array(flat_M, count_train_set, mani);
  gsl_matrix* M = &(M_view.matrix);

  gsl_matrix_view Mp_view = gsl_matrix_view_array(flat_Mp, count_predict_set, Mpcol);
  gsl_matrix* Mp = &(Mp_view.matrix);

  gsl_matrix_view Bi_map_view = gsl_matrix_view_array(flat_Bi_map, count_predict_set, varssv);
  gsl_matrix* Bi_map = &Bi_map_view.matrix;

  /* OpenMP loop with call to mf_smap_single function */
  retcode* rc = (retcode*)malloc(Mp->size1 * sizeof(retcode));
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
