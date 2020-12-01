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
#include "driver.h"
#endif

class StataIO : public IO
{
public:
  virtual void out(const char* s) const { SF_display((char*)s); }
  virtual void error(const char* s) const { SF_error((char*)s); }
  virtual void flush() const { _stata_->spoutflush(); }

  virtual void print(std::string s) const { IO::print(replace_newline(s)); }

  virtual void print_async(std::string s) const { IO::print_async(replace_newline(s)); }

  virtual void out_async(const char* s) const
  {
    SF_macro_use("_edm_print", buffer, BUFFER_SIZE);
    strcat(buffer, s);
    SF_macro_save("_edm_print", buffer);
  }

private:
  static const size_t BUFFER_SIZE = 1000;
  mutable char buffer[BUFFER_SIZE];

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

/*
 * Read in columns from Stata (i.e. what Stata calls variables).
 * Starting from column number 'j0', read in 'numCols' of columns
 */
template<typename T>
std::vector<T> stata_columns(ST_int j0, int numCols = 1)
{
  // Allocate space for the matrix of data from Stata
  int numRows = num_if_in_rows();
  std::vector<T> M(numRows * numCols);
  int ind = 0; // Flattened index of M matrix
  int r = 0;   // Count each row that isn't filtered by Stata 'if'
  for (ST_int i = SF_in1(); i <= SF_in2(); i++) {
    if (SF_ifobs(i)) { // Skip rows according to Stata's 'if'
      for (ST_int j = j0; j < j0 + numCols; j++) {
        ST_double value;
        ST_retcode rc = SF_vdata(j, i, &value);
        if (rc) {
          throw std::runtime_error(fmt::format("Cannot read Stata's variable {}", j));
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
 */
void write_stata_columns(std::vector<ST_double> toSave, ST_int j0, int numCols = 1)
{
  int ind = 0; // Index of y vector
  int r = 0;   // Count each row that isn't filtered by Stata 'if'
  for (ST_int i = SF_in1(); i <= SF_in2(); i++) {
    if (SF_ifobs(i)) { // Skip rows according to Stata's 'if'
      for (ST_int j = j0; j < j0 + numCols; j++) {
        // Convert MISSING back to Stata's missing value
        ST_double value = (toSave[ind] == MISSING) ? SV_missval : toSave[ind];
        ST_retcode rc = SF_vstore(j, i, value);
        if (rc) {
          throw std::runtime_error(fmt::format("Cannot write to Stata's variable {}", j));
        }
        ind += 1;
      }
      r += 1;
    }
  }
}

/* Print to the Stata console the inputs to the plugin  */
void print_debug_info(int argc, char* argv[], Options opts, const Manifold& M, const Manifold& Mp, bool pmani_flag,
                      ST_int pmani, ST_int E, ST_int zcount, ST_double dtweight)
{
  if (io.verbosity > 1) {
    // Header of the plugin
    io.print("\n{hline 20}\n");
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
    io.print(fmt::format("force compute = {}\n\n", opts.forceCompute));
    io.print(fmt::format("missing distance = {:.06f}\n\n", opts.missingdistance));
    io.print(fmt::format("number of variables in manifold = {}\n\n", M.E_actual()));
    io.print(fmt::format("train set obs: {}\n", M.nobs()));
    io.print(fmt::format("predict set obs: {}\n\n", Mp.nobs()));
    io.print(fmt::format("p_manifold flag = {}\n", pmani_flag));

    if (pmani_flag) {
      io.print(fmt::format("number of variables in p_manifold = {}\n", pmani));
    }
    io.print("\n");

    io.print(fmt::format("k = {}\n\n", opts.k));

    if (opts.saveMode) {
      io.print(fmt::format("columns in smap coefficients = {}\n", opts.varssv));
    }

    io.print(fmt::format("save_mode = {}\n\n", opts.saveMode));

    io.print(fmt::format("E is {}\n", E));
    io.print(fmt::format("We have {} 'extra' columns\n", zcount));
    io.print(fmt::format("Adding dt with weight {}\n", dtweight));

    io.print(fmt::format("Requested {} threads\n", argv[9]));
    io.print(fmt::format("Using {} threads\n\n", opts.nthreads));

    io.flush();
  }
}

std::future<Prediction> predictions;
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

  Options opts;

  opts.theta = atof(argv[0]);
  opts.k = atoi(argv[1]);
  opts.algorithm = std::string(argv[2]);
  opts.forceCompute = (strcmp(argv[3], "force") == 0);
  opts.missingdistance = atof(argv[4]);
  ST_int mani = atoi(argv[5]);    // number of columns in the manifold
  bool pmaniFlag = atoi(argv[6]); // contains the flag for p_manifold
  opts.saveMode = atoi(argv[7]);
  ST_int pmani = atoi(argv[8]); // contains the number of columns in p_manifold
  opts.varssv = atoi(argv[8]);  // number of columns in smap coefficients
  opts.nthreads = atoi(argv[9]);
  io.verbosity = atoi(argv[10]);

  // Default number of neighbours k is E + 1
  if (opts.k <= 0) {
    opts.k = mani + 1;
  }

  // Default number of threads is the number of physical cores available
  ST_int npcores = (ST_int)num_physical_cores();
  if (opts.nthreads <= 0) {
    opts.nthreads = npcores;
  }

  // Restrict going over the number of logical cores available
  ST_int nlcores = (ST_int)num_logical_cores();
  if (opts.nthreads > nlcores) {
    io.print(fmt::format("Restricting to {} threads (recommend {} threads)\n", nlcores, npcores));
    opts.nthreads = nlcores;
  }

  // Find which rows are used for training & which for prediction
  std::vector<bool> trainingRows = stata_columns<bool>(mani + 3);
  std::vector<bool> predictionRows = stata_columns<bool>(mani + 4);

  // Read in the main data from Stata
  std::vector<ST_double> x = stata_columns<ST_double>(1);

  // Find the number of lags 'E' for the main data.
  int E;
  char buffer[100];
  if (pmaniFlag) {
    SF_macro_use("_e", buffer, 100);
    std::string Estr(buffer);
    std::size_t found = Estr.find_last_of(" ");
    Estr = Estr.substr(found + 1);
    E = atoi(buffer);
  } else {
    SF_macro_use("_i", buffer, 100);
    E = atoi(buffer);
  }

  // Read in time
  ST_int timeCol = mani + 5 + 1 + (int)pmaniFlag * pmani + opts.saveMode * opts.varssv;
  std::vector<ST_int> t = stata_columns<ST_int>(timeCol);

  // Handle 'dt' flag
  SF_macro_use("_parsed_dt", buffer, 100);
  bool parsed_dt = (bool)atoi(buffer);
  std::vector<ST_double> dt;

  double dtweight = 0;
  if (parsed_dt) {
    SF_macro_use("_parsed_dtw", buffer, 100);
    dtweight = atof(buffer);
  }

  // Read in the extras
  SF_macro_use("_zcount", buffer, 100);
  int zcount = atoi(buffer);

  std::vector<std::vector<ST_double>> extras(zcount);

  for (int z = 0; z < zcount; z++) {
    extras[z] = stata_columns<ST_double>(2 + z);
  }

  Manifold M(x, t, extras, trainingRows, E, dtweight, MISSING);

  // Read in the prediction manifold
  std::vector<ST_double> xPred;
  std::vector<std::vector<ST_double>> extrasPred(zcount);
  if (pmaniFlag) {
    xPred = stata_columns<ST_double>(mani + 6);
    for (int z = 0; z < zcount; z++) {
      extrasPred[z] = stata_columns<ST_double>(mani + 6 + 1 + z);
    }
  } else {
    // In cases like 'edm explore x' (pmani_flag = false), then we
    // share the same data between both manifolds.
    xPred = x;
    extrasPred = extras;
  }

  Manifold Mp(xPred, t, extrasPred, predictionRows, E, dtweight, MISSING);

  // Read in the target vector 'y' from Stata
  std::vector<ST_double> yAll = stata_columns<ST_double>(mani + 1);
  std::vector<ST_double> y;
  for (size_t i = 0; i < trainingRows.size(); i++) {
    if (trainingRows[i]) {
      y.push_back(yAll[i]);
    }
  }

  print_debug_info(argc, argv, opts, M, Mp, pmaniFlag, pmani, E, zcount, dtweight);

#ifdef DUMP_INPUT
  // Here we want to dump the input so we can use it without stata for
  // debugging and profiling purposes.
  if (argc >= 12) {
    hid_t fid = H5Fcreate(argv[11], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    save_options(fid, opts);
    save_manifold(fid, "M", x, t, extras, trainingRows, E, dtweight);
    save_manifold(fid, "Mp", xPred, t, extrasPred, predictionRows, E, dtweight);

    hsize_t yLen = y.size();
    H5LTmake_dataset_double(fid, "y", 1, &yLen, y.data());

    H5Fclose(fid);
  }
#endif

  predCol = mani + 2;
  coeffsCol = mani + 5 + 1 + (int)pmaniFlag * pmani;
  coeffsWidth = opts.varssv;

  std::packaged_task<Prediction()> task(std::bind(mf_smap_loop, opts, y, M, Mp, io, keep_going, finished));
  predictions = task.get_future();

  std::thread master(std::move(task));
  master.detach();

  return SUCCESS;
}

ST_retcode save_results()
{
  Prediction res = predictions.get();

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
    io.out("{hline 20}\n\n");
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
