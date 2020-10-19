// Suppress Windows problems with sprintf etc. functions.
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include "edm.h"
#include "stplugin.h"
#include <omp.h>
#include <stdio.h>
#include <string.h>

#include <numeric> // for std::accumulate
#include <optional>
#include <stdexcept>
#include <vector>

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

typedef struct
{
  std::vector<bool> useRow;
  int numRows;
} row_filter_t;

/*
 * Read in columns from Stata (i.e. what Stata calls variables).
 * Starting from column number 'j0', read in 'numCols' of columns.
 * If 'filter' is supplied, we consider each row 'i' only if 'filter[i]'
 * evaluates to true. */
template<typename T>
std::vector<T> stata_columns(ST_int j0, int numCols = 1, const std::optional<row_filter_t>& filter = std::nullopt)
{
  // Allocate space for the matrix of data from Stata
  int numRows = filter.has_value() ? filter->numRows : num_if_in_rows();
  std::vector<T> M(numRows * numCols);
  int ind = 0; // Flattened index of M matrix
  int r = 0;   // Count each row that isn't filtered by Stata 'if'
  for (ST_int i = SF_in1(); i <= SF_in2(); i++) {
    if (SF_ifobs(i)) {                                // Skip rows according to Stata's 'if'
      if (!filter.has_value() || filter->useRow[r]) { // Skip rows given our own filter
        for (ST_int j = j0; j < j0 + numCols; j++) {
          ST_double value;
          ST_retcode rc = SF_vdata(j, i, &value);
          if (rc) {
            throw std::runtime_error("Cannot read variables from Stata");
          }
          if (SF_is_missing(value)) {
            if (std::is_floating_point<T>::value) {
              value = MISSING;
            } else {
              value = 0;
            }
          }
          M[ind] = (T)value;
          ind += 1;
        }
      }
      r += 1;
    }
  }

  return M;
}

/*
 * Write data to columns in Stata (i.e. what Stata calls variables).
 *
 * Starting from column number 'j0', write 'numCols' of columns.
 * The data being written is in the 'toSave' parameter, which is a
 * flattened row-major array.
 *
 * If supplied, we consider each row 'i' only if 'filter.hasrow[i]' evaluates to true.
 */
void write_stata_columns(std::vector<ST_double> toSave, ST_int j0, int numCols = 1,
                         const std::optional<row_filter_t>& filter = std::nullopt)
{
  int ind = 0; // Index of y vector
  int r = 0;   // Count each row that isn't filtered by Stata 'if'
  for (ST_int i = SF_in1(); i <= SF_in2(); i++) {
    if (SF_ifobs(i)) {                                // Skip rows according to Stata's 'if'
      if (!filter.has_value() || filter->useRow[r]) { // Skip rows given our own filter
        for (ST_int j = j0; j < j0 + numCols; j++) {
          // Convert MISSING back to Stata's missing value
          ST_double value = (toSave[ind] == MISSING) ? SV_missval : toSave[ind];
          ST_retcode rc = SF_vstore(j, i, value);
          if (rc) {
            throw std::runtime_error("Cannot write variables to Stata");
          }
          ind += 1;
        }
      }
      r += 1;
    }
  }
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
  ST_int stataVarNum = mani + 3;
  std::vector<bool> train_use = stata_columns<bool>(stataVarNum);
  int count_train_set = std::accumulate(train_use.begin(), train_use.end(), 0);
  row_filter_t train_filter = { train_use, count_train_set };

  stataVarNum = mani + 4;
  std::vector<bool> predict_use = stata_columns<bool>(stataVarNum);
  int count_predict_set = std::accumulate(predict_use.begin(), predict_use.end(), 0);
  row_filter_t predict_filter = { predict_use, count_predict_set };

  stataVarNum = mani + 5;
  std::vector<ST_double> S = stata_columns<ST_double>(stataVarNum, 1, predict_filter);

  // Allocation of matrix M and vector y.
  stataVarNum = 1;
  std::vector<ST_double> flat_M = stata_columns<ST_double>(stataVarNum, mani, train_filter);

  stataVarNum = mani + 1;
  std::vector<ST_double> y = stata_columns<ST_double>(stataVarNum, 1, train_filter);

  // Allocation of matrices Mp and Bimap, and vector ystar.
  ST_int Mpcol;
  if (pmani_flag) {
    Mpcol = pmani;
    stataVarNum = mani + 6;
  } else {
    Mpcol = mani;
    stataVarNum = 1;
  }

  std::vector<ST_double> flat_Mp = stata_columns<ST_double>(stataVarNum, Mpcol, predict_filter);

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
    H5LTmake_dataset_double(fid, "y", 1, yLen, y.data());

    H5LTset_attribute_int(fid, "/", "l", &l, 1);
    H5LTset_attribute_double(fid, "/", "theta", &theta, 1);

    hsize_t SLen[] = { (hsize_t)count_predict_set };
    H5LTmake_dataset_double(fid, "S", 1, SLen, S.data());

    H5LTset_attribute_string(fid, "/", "algorithm", algorithm);
    char bool_var = (char)save_mode;
    H5LTset_attribute_char(fid, "/", "save_mode", &bool_var, 1);
    bool_var = (char)force_compute;
    H5LTset_attribute_char(fid, "/", "force_compute", &bool_var, 1);
    H5LTset_attribute_int(fid, "/", "varssv", &varssv, 1);
    H5LTset_attribute_double(fid, "/", "missingdistance", &missingdistance, 1);

    hsize_t MpLen[] = { (hsize_t)(count_predict_set * Mpcol) };
    H5LTmake_dataset_double(fid, "flat_Mp", 1, MpLen, flat_Mp.data());
    hsize_t MLen[] = { (hsize_t)(count_train_set * mani) };
    H5LTmake_dataset_double(fid, "flat_M", 1, MLen, flat_M.data());

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

  smap_res_t smap_res = mf_smap_loop(count_predict_set, count_train_set, mani, Mpcol, l, theta, algorithm, save_mode,
                                     varssv, force_compute, missingdistance, y, S, flat_M, flat_Mp);

  omp_set_num_threads(originalNumThreads);

  // If there are no errors, return the value of ystar (and smap coefficients) to Stata.
  if (smap_res.rc == SUCCESS) {
    stataVarNum = mani + 2;
    write_stata_columns(smap_res.ystar, stataVarNum, 1, predict_filter);

    if (save_mode) {
      stataVarNum = mani + 5 + 1 + (int)pmani_flag * pmani;
      write_stata_columns(*(smap_res.flat_Bi_map), stataVarNum, varssv, predict_filter);
    }
  }

  // Print a Footer message for the plugin.
  if (verbosity > 0) {
    display("\nEnd of the plugin\n");
    display("====================\n\n");
  }

  return smap_res.rc;
}

STDLL stata_call(int argc, char* argv[])
{
  try {
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
