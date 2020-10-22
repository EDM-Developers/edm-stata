// Suppress Windows problems with sprintf etc. functions.
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include "edm.h"
#include "stplugin.h"
#include <cstdlib>
#include <omp.h>
#include <stdexcept>
#include <stdio.h>
#include <string.h>

#ifdef DUMP_INPUT
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

void display(char* s)
{
  SF_display(s);
}

void display(const char* s)
{
  SF_display((char*)s);
}

void error(const char* s)
{
  SF_error((char*)s);
}

void print_error(ST_retcode rc)
{
  switch (rc) {
    case TOO_FEW_VARIABLES:
      error("edm plugin call requires 11 or 12 arguments\n");
      break;
    case TOO_MANY_VARIABLES:
      error("edm plugin call requires 11 or 12 arguments\n");
      break;
    case MALLOC_ERROR:
      error("Insufficient memory\n");
      break;
    case NOT_IMPLEMENTED:
      error("Method is not yet implemented\n");
      break;
    case INSUFFICIENT_UNIQUE:
      error("Insufficient number of unique observations, consider "
            "tweaking the values of E, k or use -force- option\n");
      break;
    case INVALID_ALGORITHM:
      error("Invalid algorithm argument\n");
      break;
  }
}

/*
 * Count the number of rows that aren't being filtered out
 * by Stata's 'if' or 'in' expressions.
 */
int num_if_in_rows()
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
ST_retcode stata_columns_filtered_and_sum(const ST_double* filter, int numFiltered, ST_int j0, int numCols,
                                          double** out, double* outSum)
{
  // Allocate space for the matrix of data from Stata
  int numRows = (filter == NULL) ? num_if_in_rows() : numFiltered;
  double* M = (double*)malloc(sizeof(double) * numRows * numCols);
  if (M == NULL) {
    return MALLOC_ERROR;
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
          rc = SF_vdata(j, i, &value);
          if (rc) {
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
ST_retcode write_stata_columns_filtered(const ST_double* filter, ST_int j0, int numCols, const double* toSave)
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
          rc = SF_vstore(j, i, value);
          if (rc) {
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

/* Read some columns from Stata skipping some rows */
ST_retcode stata_columns_filtered(const ST_double* filter, int numFiltered, ST_int j, int numCols, double** out)
{
  return stata_columns_filtered_and_sum(filter, numFiltered, j, numCols, out, NULL);
}

/* Read a single column from Stata skipping some rows */
ST_retcode stata_column_filtered(const ST_double* filter, int numFiltered, ST_int j, double** out)
{
  return stata_columns_filtered(filter, numFiltered, j, 1, out);
}

/* Read a single column from Stata and calculate the column sum */
ST_retcode stata_column_and_sum(ST_int j, double** out, double* outSum)
{
  return stata_columns_filtered_and_sum(NULL, -1, j, 1, out, outSum);
}

/*  Write a single column to Stata while skipping some rows */
ST_retcode write_stata_column_filtered(const ST_double* filter, ST_int j, const double* toSave)
{
  return write_stata_columns_filtered(filter, j, 1, toSave);
}

/* Print to the Stata console the inputs to the plugin  */
void print_debug_info(int argc, char* argv[], ST_double theta, char* algorithm, bool force_compute,
                      ST_double missingdistance, ST_int mani, ST_int count_train_set, ST_int count_predict_set,
                      bool pmani_flag, ST_int pmani, ST_int l, bool save_mode, ST_int varssv, ST_int nthreads)
{
  char temps[500];

  // Header of the plugin
  display("\n====================\n");
  display("Start of the plugin\n\n");

  // Overview of variables and arguments passed and observations in sample
  sprintf(temps, "number of vars & obs = %i, %i\n", SF_nvars(), SF_nobs());
  display(temps);
  sprintf(temps, "first and last obs in sample = %i, %i\n\n", SF_in1(), SF_in2());
  display(temps);

  for (int i = 0; i < argc; i++) {
    sprintf(temps, "arg %i: %s\n", i, argv[i]);
    display(temps);
  }
  display("\n");

  sprintf(temps, "theta = %6.4f\n\n", theta);
  display(temps);
  sprintf(temps, "algorithm = %s\n\n", algorithm);
  display(temps);
  sprintf(temps, "force compute = %i\n\n", force_compute);
  display(temps);
  sprintf(temps, "missing distance = %f\n\n", missingdistance);
  display(temps);
  sprintf(temps, "number of variables in manifold = %i\n\n", mani);
  display(temps);
  sprintf(temps, "train set obs: %i\n", count_train_set);
  display(temps);
  sprintf(temps, "predict set obs: %i\n\n", count_predict_set);
  display(temps);
  sprintf(temps, "p_manifold flag = %i\n", pmani_flag);
  display(temps);

  if (pmani_flag) {
    sprintf(temps, "number of variables in p_manifold = %i\n", pmani);
    display(temps);
  }
  display("\n");

  sprintf(temps, "l = %i\n\n", l);
  display(temps);

  if (save_mode) {
    sprintf(temps, "columns in smap coefficents = %i\n", varssv);
    display(temps);
  }

  sprintf(temps, "save_mode = %i\n\n", save_mode);
  display(temps);

  sprintf(temps, "Requested %s OpenMP threads\n", argv[9]);
  display(temps);
  sprintf(temps, "Using %i OpenMP threads\n\n", nthreads);
  display(temps);
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
ST_retcode edm(int argc, char* argv[])
{
  if (argc < 11) {
    return TOO_FEW_VARIABLES;
  }
  if (argc > 12) {
    return TOO_MANY_VARIABLES;
  }

  ST_double theta = atof(argv[0]);
  ST_int l = atoi(argv[1]);
  char* algorithm = argv[2];
  bool force_compute = (strcmp(argv[3], "force") == 0);
  ST_double missingdistance = atof(argv[4]);
  ST_int mani = atoi(argv[5]);     // number of columns in the manifold
  bool pmani_flag = atoi(argv[6]); // contains the flag for p_manifold
  bool save_mode = atoi(argv[7]);
  ST_int pmani = atoi(argv[8]);  // contains the number of columns in p_manifold
  ST_int varssv = atoi(argv[8]); // number of columns in smap coefficents
  ST_int nthreads = atoi(argv[9]);
  ST_int verbosity = atoi(argv[10]);

  // Default number of neighbours is E + 1
  if (l <= 0) {
    l = mani + 1;
  }

  // Default number of threads is the number of cores available
  if (nthreads <= 0) {
    nthreads = omp_get_num_procs();
  }

  // Allocation of train_use, predict_use and S (prev. skip_obs) variables.
  ST_double *train_use, *predict_use, *S;
  ST_int count_train_set, count_predict_set;

  ST_double sum;
  ST_int stataVarNum = mani + 3;
  ST_retcode rc = stata_column_and_sum(stataVarNum, &train_use, &sum);
  if (rc) {
    return rc;
  }
  count_train_set = (int)sum;

  stataVarNum = mani + 4;
  rc = stata_column_and_sum(stataVarNum, &predict_use, &sum);
  if (rc) {
    return rc;
  }
  count_predict_set = (int)sum;

  stataVarNum = mani + 5;
  rc = stata_column_filtered(predict_use, count_predict_set, stataVarNum, &S);
  if (rc) {
    return rc;
  }

  // Allocation of matrix M and vector y.
  ST_double *flat_M, *y;

  stataVarNum = 1;
  rc = stata_columns_filtered(train_use, count_train_set, stataVarNum, mani, &flat_M);
  if (rc) {
    return rc;
  }

  stataVarNum = mani + 1;
  rc = stata_column_filtered(train_use, count_train_set, stataVarNum, &y);
  if (rc) {
    return rc;
  }

  // Allocation of matrices Mp and Bimap, and vector ystar.
  ST_int Mpcol;
  if (pmani_flag) {
    Mpcol = pmani;
    stataVarNum = mani + 6;
  } else {
    Mpcol = mani;
    stataVarNum = 1;
  }

  ST_double* flat_Mp;
  rc = stata_columns_filtered(predict_use, count_predict_set, stataVarNum, Mpcol, &flat_Mp);
  if (rc) {
    return rc;
  }

  ST_double* flat_Bi_map = NULL;
  if (save_mode) {
    flat_Bi_map = (ST_double*)malloc(sizeof(ST_double) * count_predict_set * varssv);
    if (flat_Bi_map == NULL) {
      return MALLOC_ERROR;
    }
  }

  ST_double* ystar = (ST_double*)malloc(sizeof(ST_double) * count_predict_set);
  if (ystar == NULL) {
    return MALLOC_ERROR;
  }

#ifdef DUMP_INPUT
  // Here we want to dump the input so we can use it without stata for
  // debugging and profiling purposes.
  if (argc >= 12) {
    hid_t fid = H5Fcreate(argv[11], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    H5LTset_attribute_int(fid, "/", "count_train_set", &count_train_set, 1);
    H5LTset_attribute_int(fid, "/", "count_predict_set", &count_predict_set, 1);
    H5LTset_attribute_int(fid, "/", "Mpcol", &Mpcol, 1);
    H5LTset_attribute_int(fid, "/", "mani", &mani, 1);

    hsize_t yLen[] = { (hsize_t)count_train_set };
    H5LTmake_dataset_double(fid, "y", 1, yLen, y);

    H5LTset_attribute_int(fid, "/", "l", &l, 1);
    H5LTset_attribute_double(fid, "/", "theta", &theta, 1);

    hsize_t SLen[] = { (hsize_t)count_predict_set };
    H5LTmake_dataset_double(fid, "S", 1, SLen, S);

    H5LTset_attribute_string(fid, "/", "algorithm", algorithm);
    H5LTset_attribute_int(fid, "/", "save_mode", (int*)&save_mode, 1);
    H5LTset_attribute_int(fid, "/", "force_compute", (int*)&force_compute, 1);
    H5LTset_attribute_int(fid, "/", "varssv", &varssv, 1);
    H5LTset_attribute_double(fid, "/", "missingdistance", &missingdistance, 1);

    hsize_t MpLen[] = { (hsize_t)(count_predict_set * Mpcol) };
    H5LTmake_dataset_double(fid, "flat_Mp", 1, MpLen, flat_Mp);
    hsize_t MLen[] = { (hsize_t)(count_train_set * mani) };
    H5LTmake_dataset_double(fid, "flat_M", 1, MLen, flat_M);

    H5Fclose(fid);
  }
#endif

  // Find the number of threads Stata was already using, so we can reset to this later.
  int originalNumThreads;
#pragma omp parallel
  {
    originalNumThreads = omp_get_num_threads();
  }
  omp_set_num_threads(nthreads);

  // Ask OpenMP how many threads it's using, in case it ignored our request in the previous line.
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }

  if (verbosity > 0) {
    print_debug_info(argc, argv, theta, algorithm, force_compute, missingdistance, mani, count_train_set,
                     count_predict_set, pmani_flag, pmani, l, save_mode, varssv, nthreads);
  }

  rc = mf_smap_loop(count_predict_set, count_train_set, mani, Mpcol, flat_M, flat_Mp, y, l, theta, S, algorithm,
                    save_mode, varssv, force_compute, missingdistance, ystar, flat_Bi_map);

  omp_set_num_threads(originalNumThreads);

  // If there are no errors, return the value of ystar (and smap coefficients) to Stata.
  if (rc == SUCCESS) {
    stataVarNum = mani + 2;
    rc = write_stata_column_filtered(predict_use, stataVarNum, ystar);

    if (rc == SUCCESS && save_mode) {
      stataVarNum = mani + 5 + 1 + (int)pmani_flag * pmani;
      rc = write_stata_columns_filtered(predict_use, stataVarNum, varssv, flat_Bi_map);
    }
  }

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

  // Print a Footer message for the plugin.
  if (verbosity > 0) {
    display("\nEnd of the plugin\n");
    display("====================\n\n");
  }

  return rc;
}

STDLL stata_call(int argc, char* argv[])
{
  try {
    // On Mac, it may complain that we use multiple OpenMP
    // runtimes; just power on for now rather than crashing.
#ifndef _MSC_VER
    putenv((char*)"KMP_DUPLICATE_LIB_OK=TRUE");
#endif

    ST_retcode rc = edm(argc, argv);
    print_error(rc);
    return rc;
  } catch (const std::exception& e) {
    error(e.what());
    error("\n");
  } catch (...) {
    error("Unknown error in edm plugin\n");
  }
  return UNKNOWN_ERROR;
}
