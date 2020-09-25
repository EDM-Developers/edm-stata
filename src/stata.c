/* Suppress Windows problems with sprintf etc. functions. */
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include "edm.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DUMP_INPUT
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

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

/*
 * Count the number of rows that aren't being filtered out
 * by Stata's 'if' or 'in' expressions.
 */
static int num_if_in_rows()
{
  int num = 0;
  for (ST_int i = SF_in1(); i <= SF_in2(); i++) {
    if (SF_ifobs(i)) {
      num += 1;
    }
  }
  return num;
}

/*
 * Read in columns from Stata (i.e. what Stata calls variables).
 *
 * Starting from column number 'j0', read in 'numCols' of columns.
 * The result is stored in the 'out' variable, and the column sum in 'outSum'.
 *
 * If 'filter' is not NULL, we consider each row 'i' only if 'filter[i]'
 * evaluates to true. To allocate properly the correct amount, pass in
 * the 'numFiltered' argument which is the total number of rows which are
 * true in the filter.
 */
static ST_retcode stata_columns_filtered(const ST_double* filter, int numFiltered, ST_int j0, int numCols, double** out,
                                         double* outSum)
{
  // Allocate space for the matrix of data from Stata
  int numRows = (filter == NULL) ? num_if_in_rows() : numFiltered;
  double* M = (double*)malloc(sizeof(double) * numRows * numCols);
  if (M == NULL) {
    return print_error(MALLOC_ERROR);
  }

  int ind = 0; // Flattened index of M matrix
  ST_retcode rc = 0;
  ST_double value = 0;
  ST_double sum = 0;

  int r = 0; // Count each row that isn't filtered by Stata 'if'
  for (ST_int i = SF_in1(); i <= SF_in2(); i++) {
    if (SF_ifobs(i)) {                   // Skip rows according to Stata's 'if'
      if (filter == NULL || filter[r]) { // Skip rows given our own filter
        for (ST_int j = j0; j < j0 + numCols; j++) {
          if (rc = SF_vdata(j, i, &value)) {
            free(M);
            return rc;
          }

          // Set missing values to MISSING
          if (!SF_is_missing(value)) {
            M[ind] = value;
            sum += value;
          } else {
            M[ind] = MISSING;
          }
          ind += 1;
        }
      }
      r += 1;
    }
  }

  *out = M;
  if (outSum != NULL) {
    *outSum = sum;
  }

  return SUCCESS;
}

/*
 * Write data to columns in Stata (i.e. what Stata calls variables).
 *
 * Starting from column number 'j0', write 'numCols' of columns.
 * The data being written is in the 'toSave' parameter, which is a
 * flattened row-major array.
 *
 * If 'filter' is not NULL, we consider each row 'i' only if 'filter[i]'
 * evaluates to true.
 */
static ST_retcode write_stata_columns_filtered(const ST_double* filter, ST_int j0, int numCols, const double* toSave)
{
  int ind = 0; // Index of y vector
  ST_retcode rc = 0;
  ST_double value = 0;

  int r = 0; // Count each row that isn't filtered by Stata 'if'
  for (ST_int i = SF_in1(); i <= SF_in2(); i++) {
    if (SF_ifobs(i)) {                   // Skip rows according to Stata's 'if'
      if (filter == NULL || filter[r]) { // Skip rows given our own filter
        for (ST_int j = j0; j < j0 + numCols; j++) {
          // Convert MISSING back to Stata's missing value
          value = (toSave[ind] == MISSING) ? SV_missval : toSave[ind];
          if (rc = SF_vstore(j, i, value)) {
            return rc;
          }
          ind += 1;
        }
      }
      r += 1;
    }
  }

  return SUCCESS;
}

static ST_retcode stata_column_filtered(const ST_double* filter, int numFiltered, ST_int j, double** out,
                                        double* outSum)
{
  return stata_columns_filtered(filter, numFiltered, j, 1, out, outSum);
}

static ST_retcode stata_column(ST_int j, double** out, double* outSum)
{
  return stata_columns_filtered(NULL, -1, j, 1, out, outSum);
}

static ST_retcode write_stata_column_filtered(const ST_double* filter, ST_int j, const double* toSave)
{
  return write_stata_columns_filtered(filter, j, 1, toSave);
}

ST_retcode train_manifold(const ST_double* train_use, int count_train_set, int mani, double** out)
{
  return stata_columns_filtered(train_use, count_train_set, 1, mani, out, NULL);
}

ST_retcode train_y(const ST_double* train_use, int count_train_set, int mani, double** out)
{
  return stata_column_filtered(train_use, count_train_set, mani + 1, out, NULL);
}

ST_retcode predict_manifold(const ST_double* predict_use, int count_predict_set, int mani, double** out)
{
  return stata_columns_filtered(predict_use, count_predict_set, 1, mani, out, NULL);
}

ST_retcode train_set(int mani, ST_double** out, double* count_train_set) // a.k.a. co_train_set
{
  return stata_column(mani + 3, out, count_train_set);
}

ST_retcode predict_set(int mani, ST_double** out, double* count_predict_set) // a.k.a. co_predict_set
{
  return stata_column(mani + 4, out, count_predict_set);
}

ST_retcode skip_set(const ST_double* predict_use, int count_predict_set, int mani, ST_double** out) // a.k.a. overlap
{
  return stata_column_filtered(predict_use, count_predict_set, mani + 5, out, NULL);
}

// a.k.a. co_mapping
ST_retcode predict_manifold_pmani(const ST_double* predict_use, int count_predict_set, int mani, int pmani,
                                  double** out)
{
  return stata_columns_filtered(predict_use, count_predict_set, mani + 6, pmani, out, NULL);
}

ST_retcode write_ystar(const ST_double* predict_use, int mani, const double* ystar)
{
  return write_stata_column_filtered(predict_use, mani + 2, ystar);
}

ST_retcode write_smap_coefficients(const ST_double* predict_use, int mani, bool pmani_flag, int pmani, int varssv,
                                   const double* Bi_map)
{
  ST_int j0 = mani + 5 + 1 + (int)pmani_flag * pmani;
  return write_stata_columns_filtered(predict_use, j0, varssv, Bi_map);
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

DLL ST_retcode stata_call(int argc, char* argv[])
{
  bool force_compute, pmani_flag, save_mode;
  char temps[500], *algorithm;
  ST_retcode rc;
  ST_int mani, pmani, l, varssv;
  ST_int i, nthreads;
  ST_int count_train_set, count_predict_set, Mpcol;
  ST_double theta, missingdistance;
  ST_double *train_use, *predict_use, *y, *S, *ystar;
  gsl_matrix *M, *Mp, *Bi_map;

  /* header of the plugin */
  SF_display("\n");
  SF_display("====================\n");
  SF_display("Start of the plugin\n");
  SF_display("\n");

  /* overview of variables and arguments passed and observations in sample */
  sprintf(temps, "number of vars & obs = %i, %i\n", SF_nvars(), SF_nobs());
  SF_display(temps);
  sprintf(temps, "first and last obs in sample = %i, %i\n", SF_in1(), SF_in2());
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
  algorithm = argv[2];
  sprintf(temps, "algorithm = %s\n", algorithm);
  SF_display(temps);
  SF_display("\n");

  /* allocation of variable force_compute based on fourth argument */
  force_compute = (strcmp(argv[3], "force") == 0);
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

  /* allocation of train_use, predict_use and S (prev. skip_obs) variables */
  ST_double sum;
  if (rc = train_set(mani, &train_use, &sum)) {
    return print_error(rc);
  }
  count_train_set = (int)sum;
  if (rc = predict_set(mani, &predict_use, &sum)) {
    return print_error(rc);
  }
  count_predict_set = (int)sum;
  if (rc = skip_set(predict_use, count_predict_set, mani, &S)) {
    return print_error(rc);
  }

  sprintf(temps, "train set obs: %i\n", count_train_set);
  SF_display(temps);
  sprintf(temps, "predict set obs: %i\n", count_predict_set);
  SF_display(temps);
  SF_display("\n");

  /* allocation of matrices M and y */
  ST_double* flat_M = NULL;
  if (rc = train_manifold(train_use, count_train_set, mani, &flat_M)) {
    return print_error(rc);
  }
  gsl_matrix_view M_view = gsl_matrix_view_array(flat_M, count_train_set, mani);
  M = &(M_view.matrix);

  if (rc = train_y(train_use, count_train_set, mani, &y)) {
    return print_error(rc);
  }

  /* allocation of matrices Mp, S, ystar */
  pmani_flag = atoi(argv[6]); /* contains the flag for p_manifold */
  sprintf(temps, "p_manifold flag = %i \n", pmani_flag);
  SF_display(temps);

  pmani = 0;
  if (pmani_flag) {
    pmani = atoi(argv[8]); /* contains the number of columns in p_manifold */
    sprintf(temps, "number of variables in p_manifold = %i \n", pmani);
    SF_display(temps);
    Mpcol = pmani;
  } else {
    Mpcol = mani;
  }
  SF_display("\n");

  ST_double* flat_Mp = NULL;
  gsl_matrix_view Mp_view;
  if (pmani_flag) {
    if (rc = predict_manifold_pmani(predict_use, count_predict_set, mani, pmani, &flat_Mp)) {
      return rc;
    }
    Mp_view = gsl_matrix_view_array(flat_Mp, count_predict_set, pmani);
  } else {
    if (rc = predict_manifold(predict_use, count_predict_set, mani, &flat_Mp)) {
      return rc;
    }
    Mp_view = gsl_matrix_view_array(flat_Mp, count_predict_set, mani);
  }
  Mp = &(Mp_view.matrix);

  l = atoi(argv[1]); /* contains l */
  if (l <= 0) {
    l = mani + 1;
  }
  sprintf(temps, "l = %i \n", l);
  SF_display(temps);
  SF_display("\n");

  save_mode = atoi(argv[7]); /* contains the flag for vars_save */

  double* flat_Bi_map = NULL;
  gsl_matrix_view Bi_map_view;
  Bi_map = NULL;

  if (save_mode) {          /* flag savesmap is ON */
    varssv = atoi(argv[8]); /* contains the number of columns
                               in smap coefficents */
    flat_Bi_map = malloc(sizeof(ST_double) * count_predict_set * varssv);
    if (flat_Bi_map == NULL) {
      return print_error(MALLOC_ERROR);
    }
    Bi_map_view = gsl_matrix_view_array(flat_Bi_map, count_predict_set, varssv);
    Bi_map = &Bi_map_view.matrix;

    sprintf(temps, "columns in smap coefficents = %i \n", varssv);
    SF_display(temps);

  } else { /* flag savesmap is OFF */
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

#ifdef DUMP_INPUT
  // Here we want to dump the input so we can use it without stata for
  // debugging and profiling purposes.
  if (argc >= 11) {
    hid_t fid = H5Fcreate(argv[10], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    H5LTset_attribute_int(fid, "/", "count_train_set", &count_train_set, 1);
    H5LTset_attribute_int(fid, "/", "count_predict_set", &count_predict_set, 1);
    H5LTset_attribute_int(fid, "/", "Mpcol", &Mpcol, 1);
    H5LTset_attribute_int(fid, "/", "mani", &mani, 1);

    H5LTmake_dataset_double(fid, "y", 1, (hsize_t[]){ count_train_set }, y);

    H5LTset_attribute_int(fid, "/", "l", &l, 1);
    H5LTset_attribute_double(fid, "/", "theta", &theta, 1);

    H5LTmake_dataset_double(fid, "S", 1, (hsize_t[]){ count_predict_set }, S);

    H5LTset_attribute_string(fid, "/", "algorithm", algorithm);
    H5LTset_attribute_int(fid, "/", "save_mode", (int*)&save_mode, 1);
    H5LTset_attribute_int(fid, "/", "force_compute", (int*)&force_compute, 1);
    H5LTset_attribute_int(fid, "/", "varssv", &varssv, 1);
    H5LTset_attribute_double(fid, "/", "missingdistance", &missingdistance, 1);

    H5LTmake_dataset_double(fid, "flat_Mp", 1, (hsize_t[]){ count_predict_set * Mpcol }, flat_Mp);
    H5LTmake_dataset_double(fid, "flat_M", 1, (hsize_t[]){ count_train_set * mani }, flat_M);

    H5Fclose(fid);
  }
#endif

  rc = mf_smap_loop(count_predict_set, count_train_set, mani, M, Mp, y, l, theta, S, algorithm, save_mode, varssv,
                    force_compute, missingdistance, ystar, Bi_map);

  /* If there are no errors, return the value of ystar (and smap coefficients) to Stata */
  if (rc == SUCCESS) {
    write_ystar(predict_use, mani, ystar);
    if (save_mode) {
      write_smap_coefficients(predict_use, mani, pmani_flag, pmani, varssv, flat_Bi_map);
    }
  } else {
    print_error(rc);
  }

  /* deallocation of matrices and arrays before exiting the plugin */
  free(train_use);
  free(predict_use);
  free(S);
  free(flat_M);
  free(y);
  free(flat_Mp);
  if (save_mode) {
    free(flat_Bi_map);
  }
  free(ystar);

  /* footer of the plugin */
  SF_display("\n");
  SF_display("End of the plugin\n");
  SF_display("====================\n");
  SF_display("\n");

  return rc;
}
