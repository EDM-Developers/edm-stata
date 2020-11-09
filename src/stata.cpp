// Suppress Windows problems with sprintf etc. functions.
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include "edm.h"
#include "stplugin.h"

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include <future>
#include <numeric> // for std::accumulate
#include <optional>
#include <stdexcept>
#include <thread>
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

void display(std::string s)
{
  SF_display((char*)s.c_str());
}

void error(const char* s)
{
  SF_error((char*)s);
}

void flush()
{
  _stata_->spoutflush();
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

struct RowFilter
{
  std::vector<bool> useRow;
  int numRows;
};

/*
 * Read in columns from Stata (i.e. what Stata calls variables).
 * Starting from column number 'j0', read in 'numCols' of columns.
 * If 'filter' is supplied, we consider each row 'i' only if 'filter[i]'
 * evaluates to true. */
template<typename T>
std::vector<T> stata_columns(ST_int j0, int numCols = 1, const std::optional<RowFilter>& filter = std::nullopt)
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
                         const std::optional<RowFilter>& filter = std::nullopt)
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
void print_debug_info(int argc, char* argv[], EdmOptions opts, const Manifold& M, const Manifold& Mp, bool pmani_flag,
                      ST_int pmani, ST_int nthreads)
{
  // Header of the plugin
  display("\n====================\n");
  display("Start of the plugin\n\n");

  // Overview of variables and arguments passed and observations in sample
  display(fmt::format("number of vars & obs = {}, {}\n", SF_nvars(), SF_nobs()));
  display(fmt::format("first and last obs in sample = {}, {}\n\n", SF_in1(), SF_in2()));

  for (int i = 0; i < argc; i++) {
    display(fmt::format("arg {}: {}\n", i, argv[i]));
  }
  display("\n");

  display(fmt::format("theta = {:6.4f}\n\n", opts.theta));
  display(fmt::format("algorithm = {}\n\n", opts.algorithm.c_str()));
  display(fmt::format("force compute = {}\n\n", opts.force_compute));
  display(fmt::format("missing distance = {:.06f}\n\n", opts.missingdistance));
  display(fmt::format("number of variables in manifold = {}\n\n", M.cols()));
  display(fmt::format("train set obs: {}\n", M.rows()));
  display(fmt::format("predict set obs: {}\n\n", Mp.rows()));
  display(fmt::format("p_manifold flag = {}\n", pmani_flag));

  if (pmani_flag) {
    display(fmt::format("number of variables in p_manifold = {}\n", pmani));
  }
  display("\n");

  display(fmt::format("l = {}\n\n", opts.l));

  if (opts.save_mode) {
    display(fmt::format("columns in smap coefficients = {}\n", opts.varssv));
  }

  display(fmt::format("save_mode = {}\n\n", opts.save_mode));

  display(fmt::format("Requested {} threads\n", argv[9]));
  display(fmt::format("Using {} threads\n\n", nthreads));

  flush();
}

ST_int mani;
std::future<EdmResult> future;
ST_int smap_coeffs_column;
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
  std::string algorithm(argv[2]);
  bool force_compute = (strcmp(argv[3], "force") == 0);
  ST_double missingdistance = atof(argv[4]);
  mani = atoi(argv[5]);            // number of columns in the manifold
  bool pmani_flag = atoi(argv[6]); // contains the flag for p_manifold
  bool save_mode = atoi(argv[7]);
  ST_int pmani = atoi(argv[8]);  // contains the number of columns in p_manifold
  ST_int varssv = atoi(argv[8]); // number of columns in smap coefficients
  ST_int nthreads = atoi(argv[9]);
  ST_int verbosity = atoi(argv[10]);

  // Default number of neighbours is E + 1
  if (l <= 0) {
    l = mani + 1;
  }

  EdmOptions opts = { force_compute, save_mode, l, varssv, verbosity, theta, missingdistance, algorithm };

  // Default number of threads is the number of cores available
  if (nthreads <= 0) {
    nthreads = std::thread::hardware_concurrency();
  }

  // Find which Stata rows contain the main manifold & for the y vector
  ST_int stataVarNum = mani + 3;
  auto train_use = stata_columns<bool>(stataVarNum);
  int count_train_set = std::accumulate(train_use.begin(), train_use.end(), 0);
  RowFilter train_filter = { train_use, count_train_set };

  // Read in the y vector from Stata
  stataVarNum = mani + 1;
  std::vector<double> y = stata_columns<ST_double>(stataVarNum, 1, train_filter);

  // Read in the main manifold from Stata
  stataVarNum = 1;
  Manifold M = { stata_columns<ST_double>(stataVarNum, mani, train_filter), (size_t)count_train_set, (size_t)mani };

  // Find which Stata rows contain the second manifold
  stataVarNum = mani + 4;
  auto predict_use = stata_columns<bool>(stataVarNum);
  int count_predict_set = std::accumulate(predict_use.begin(), predict_use.end(), 0);
  RowFilter predict_filter = { predict_use, count_predict_set };

  // Find which Stata columns contain the second manifold
  ST_int Mpcol;
  if (pmani_flag) {
    Mpcol = pmani;
    stataVarNum = mani + 6;
  } else {
    Mpcol = mani;
    stataVarNum = 1;
  }

  // Read in the second manifold from Stata
  Manifold Mp = { stata_columns<ST_double>(stataVarNum, Mpcol, predict_filter), (size_t)count_predict_set,
                  (size_t)Mpcol };

#ifdef DUMP_INPUT
  // Here we want to dump the input so we can use it without stata for
  // debugging and profiling purposes.
  if (argc >= 12) {
    hid_t fid = H5Fcreate(argv[11], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    H5LTset_attribute_int(fid, "/", "count_train_set", &count_train_set, 1);
    H5LTset_attribute_int(fid, "/", "count_predict_set", &count_predict_set, 1);
    H5LTset_attribute_int(fid, "/", "Mpcol", &Mpcol, 1);
    H5LTset_attribute_int(fid, "/", "mani", &mani, 1);

    hsize_t yLen = y.size();
    H5LTmake_dataset_double(fid, "y", 1, &yLen, y.data());

    H5LTset_attribute_int(fid, "/", "l", &l, 1);
    H5LTset_attribute_double(fid, "/", "theta", &theta, 1);

    H5LTset_attribute_string(fid, "/", "algorithm", algorithm.c_str());
    char bool_var = (char)save_mode;
    H5LTset_attribute_char(fid, "/", "save_mode", &bool_var, 1);
    bool_var = (char)force_compute;
    H5LTset_attribute_char(fid, "/", "force_compute", &bool_var, 1);
    H5LTset_attribute_int(fid, "/", "varssv", &varssv, 1);
    H5LTset_attribute_double(fid, "/", "missingdistance", &missingdistance, 1);

    hsize_t MLen = M.flat.size();
    H5LTmake_dataset_double(fid, "flat_M", 1, &MLen, M.flat.data());
    hsize_t MpLen = Mp.flat.size();
    H5LTmake_dataset_double(fid, "flat_Mp", 1, &MpLen, Mp.flat.data());

    H5LTset_attribute_int(fid, "/", "nthreads", &nthreads, 1);

    H5Fclose(fid);
  }
#endif

  if (verbosity > 3) {
    print_debug_info(argc, argv, opts, M, Mp, pmani_flag, pmani, nthreads);
  }

  smap_coeffs_column = mani + 5 + 1 + (int)pmani_flag * pmani;

  IO io = { display, error, flush };
  future = std::async(mf_smap_loop, opts, y, M, Mp, nthreads, io);
  return SUCCESS;
}

template<typename R>
bool is_ready(std::future<R> const& f)
{
  return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

ST_retcode save_results()
{
  if (is_ready(future)) {

    EdmResult res = future.get();

    ST_retcode rc = SF_scal_save("edm_mae", res.mae);
    if (rc) {
      throw std::runtime_error("Cannot read 'edm_save_prediction' scalar from Stata");
    }

    rc = SF_scal_save("edm_rho", res.rho);
    if (rc) {
      throw std::runtime_error("Cannot read 'edm_save_prediction' scalar from Stata");
    }

    // If there are no errors, return the value of ystar (and smap coefficients) to Stata.
    if (res.rc == SUCCESS) {

      ST_int stataVarNum = mani + 2;
      write_stata_columns(res.ystar, stataVarNum);

      if (res.opts.save_mode) {
        write_stata_columns(res.flat_Bi_map, smap_coeffs_column, res.opts.varssv);
      }
    }

    // Print a Footer message for the plugin.
    if (res.opts.verbosity > 0) {
      display("\nEnd of the plugin\n");
      display("====================\n\n");
    }

    SF_scal_save("edm_running", 0.0);

    return res.rc;

  } else {
    return SUCCESS;
  }
}

STDLL stata_call(int argc, char* argv[])
{
  try {
    if (argc > 0) {
      ST_retcode rc = edm(argc, argv);
      print_error(rc);
      return rc;

    } else {
      ST_retcode rc = save_results();
      print_error(rc);
      return rc;
    }

  } catch (const std::exception& e) {
    error(e.what());
    error("\n");
  } catch (...) {
    error("Unknown error in edm plugin\n");
  }

  return UNKNOWN_ERROR;
}
