// Suppress Windows problems with sprintf etc. functions.
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include "cpu.h"
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
#include <string>
#include <thread>
#include <vector>

#ifdef DUMP_INPUT
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

class StataIO : public IO
{
public:
  virtual void out(const char* s) const { SF_display((char*)s); }
  virtual void error(const char* s) const { SF_error((char*)s); }
  virtual void flush() const { _stata_->spoutflush(); }

  virtual void print(std::string s) const { IO::print(replace_newline(s)); }

  virtual void print_async(std::string s) const { IO::print_async(replace_newline(s)); }

  virtual void out_async(const char* s) const { SF_macro_save("_edm_print", (char*)s); }

private:
  std::string replace_newline(std::string s) const
  {
    size_t ind;
    while ((ind = s.find("\n")) != std::string::npos) {
      s = s.replace(ind, ind + 1, "{break}");
    }
    return s;
  }
};

StataIO io;

bool keep_going()
{
  double edm_running;
  SF_scal_use("edm_running", &edm_running);
  return (bool)edm_running;
}

void finished()
{
  SF_scal_save("edm_running", 0.0);
}

void print_error(ST_retcode rc)
{
  switch (rc) {
    case TOO_FEW_VARIABLES:
      io.error("edm plugin call requires 11 or 12 arguments\n");
      break;
    case TOO_MANY_VARIABLES:
      io.error("edm plugin call requires 11 or 12 arguments\n");
      break;
    case NOT_IMPLEMENTED:
      io.error("Method is not yet implemented\n");
      break;
    case INSUFFICIENT_UNIQUE:
      io.error("Insufficient number of unique observations, consider "
               "tweaking the values of E, k or use -force- option\n");
      break;
    case INVALID_ALGORITHM:
      io.error("Invalid algorithm argument\n");
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
void print_debug_info(int argc, char* argv[], smap_opts_t opts, const manifold_t& M, const manifold_t& Mp,
                      bool pmani_flag, ST_int pmani, ST_int nthreads)
{
  if (io.verbosity > 1) {
    // Header of the plugin
    io.print("\n====================\n");
    io.print("Start of the plugin\n\n");

    // Overview of variables and arguments passed and observations in sample
    io.print(fmt::format("number of vars & obs = {}, {}\n", SF_nvars(), SF_nobs()));
    io.print(fmt::format("first and last obs in sample = {}, {}\n\n", SF_in1(), SF_in2()));

    for (int i = 0; i < argc; i++) {
      io.print(fmt::format("arg {}: {}\n", i, argv[i]));
    }
    io.print("\n");

    io.print(fmt::format("theta = {:6.4f}\n\n", opts.theta));
    io.print(fmt::format("algorithm = {}\n\n", opts.algorithm));
    io.print(fmt::format("force compute = {}\n\n", opts.force_compute));
    io.print(fmt::format("missing distance = {:.06f}\n\n", opts.missingdistance));
    io.print(fmt::format("number of variables in manifold = {}\n\n", M.cols));
    io.print(fmt::format("train set obs: {}\n", M.rows));
    io.print(fmt::format("predict set obs: {}\n\n", Mp.rows));
    io.print(fmt::format("p_manifold flag = {}\n", pmani_flag));

    if (pmani_flag) {
      io.print(fmt::format("number of variables in p_manifold = {}\n", pmani));
    }
    io.print("\n");

    io.print(fmt::format("l = {}\n\n", opts.l));

    if (opts.save_mode) {
      io.print(fmt::format("columns in smap coefficients = {}\n", opts.varssv));
    }

    io.print(fmt::format("save_mode = {}\n\n", opts.save_mode));

    io.print(fmt::format("Requested {} threads\n", argv[9]));
    io.print(fmt::format("Using {} threads\n\n", nthreads));

    io.flush();
  }
}

std::future<smap_res_t> predictions;
ST_int predCol, coeffsCol, coeffsWidth;

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

  smap_opts_t opts;

  opts.theta = atof(argv[0]);
  opts.l = atoi(argv[1]);
  opts.algorithm = std::string(argv[2]);
  opts.force_compute = (strcmp(argv[3], "force") == 0);
  opts.missingdistance = atof(argv[4]);
  ST_int mani = atoi(argv[5]);     // number of columns in the manifold
  bool pmani_flag = atoi(argv[6]); // contains the flag for p_manifold
  opts.save_mode = atoi(argv[7]);
  ST_int pmani = atoi(argv[8]); // contains the number of columns in p_manifold
  opts.varssv = atoi(argv[8]);  // number of columns in smap coefficients
  ST_int nthreads = atoi(argv[9]);
  io.verbosity = atoi(argv[10]);

  // Default number of neighbours is E + 1
  if (opts.l <= 0) {
    opts.l = mani + 1;
  }

  // Default number of threads is the number of physical cores available
  ST_int npcores = (ST_int)num_physical_cores();
  if (nthreads <= 0) {
    nthreads = npcores;
  }

  // Restrict going over the number of logical cores available
  ST_int nlcores = (ST_int)num_logical_cores();
  if (nthreads > nlcores) {
    io.print(fmt::format("Restricting to {} threads (recommend {} threads)\n", nlcores, npcores));
    nthreads = nlcores;
  }

  // Find which Stata rows contain the main manifold & for the y vector
  ST_int stataVarNum = mani + 3;
  auto train_use = stata_columns<bool>(stataVarNum);
  int count_train_set = std::accumulate(train_use.begin(), train_use.end(), 0);
  row_filter_t train_filter = { train_use, count_train_set };

  // Read in the y vector from Stata
  stataVarNum = mani + 1;
  auto y = stata_columns<ST_double>(stataVarNum, 1, train_filter);

  // Read in the main manifold from Stata
  stataVarNum = 1;
  auto _flat_M = stata_columns<ST_double>(stataVarNum, mani, train_filter);
  manifold_t M = { std::move(_flat_M), count_train_set, mani };

  // Find which Stata rows contain the second manifold
  stataVarNum = mani + 4;
  auto predict_use = stata_columns<bool>(stataVarNum);
  int count_predict_set = std::accumulate(predict_use.begin(), predict_use.end(), 0);
  row_filter_t predict_filter = { predict_use, count_predict_set };

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
  auto _flat_Mp = stata_columns<ST_double>(stataVarNum, Mpcol, predict_filter);
  manifold_t Mp{ std::move(_flat_Mp), count_predict_set, Mpcol };

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

  print_debug_info(argc, argv, opts, M, Mp, pmani_flag, pmani, nthreads);

  predCol = mani + 2;
  coeffsCol = mani + 5 + 1 + (int)pmani_flag * pmani;
  coeffsWidth = opts.varssv;

  std::packaged_task<smap_res_t()> task(std::bind(mf_smap_loop, opts, y, M, Mp, nthreads, io, keep_going, finished));
  predictions = task.get_future();

  std::thread master(std::move(task));
  master.detach();

  return SUCCESS;
}

ST_retcode save_results()
{
  smap_res_t res = predictions.get();

  // If there are no errors, return the value of ystar (and smap coefficients) to Stata.
  if (res.rc == SUCCESS) {
    write_stata_columns(res.ystar, predCol);

    if (res.flat_Bi_map.has_value()) {
      write_stata_columns(*res.flat_Bi_map, coeffsCol, coeffsWidth);
    }
  }

  // Print a Footer message for the plugin.
  if (io.verbosity > 1) {
    io.out("\nEnd of the plugin\n");
    io.out("====================\n\n");
  }

  finished();

  return res.rc;
}

STDLL stata_call(int argc, char* argv[])
{
  try {
    if (argc > 0) {
      return edm(argc, argv);
    } else {
      ST_retcode rc = save_results();
      print_error(rc);
      return rc;
    }
  } catch (const std::exception& e) {
    io.error(e.what());
    io.error("\n");
  } catch (...) {
    io.error("Unknown error in edm plugin\n");
  }
  return UNKNOWN_ERROR;
}
