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
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
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
      s.replace(ind, 1, "{break}");
    }
    return s;
  }
};

// Global state, needed to persist between multiple edm calls
StataIO io;
std::queue<std::future<Prediction>> predictions;

bool keep_going()
{
  double edm_running;
  SF_scal_use("edm_running", &edm_running);
  return (bool)edm_running;
}

void finished(PredictionStats stats)
{
  SF_scal_save("edm_running", 0.0);

  // Save the rho/MAE results if requested (i.e. not for coprediction)
  if (stats.calcRhoMAE) {
    std::string resultMatrix = "r";
    if (stats.xmap) {
      resultMatrix += fmt::format("{}", stats.xmapDirectionNum);
    }

    if (SF_mat_store((char*)resultMatrix.c_str(), stats.taskNum, 3, stats.rho)) {
      io.print(fmt::format("Error: failed to save rho {} to matrix '{}'\n", stats.rho, resultMatrix));
    }
    if (SF_mat_store((char*)resultMatrix.c_str(), stats.taskNum, 4, stats.mae)) {
      io.print(fmt::format("Error: failed to save MAE {} to matrix '{}'\n", stats.mae, resultMatrix));
    }
  }
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
 * Write data to columns in Stata (i.e. what Stata calls variables),
 * starting from column number 'j0'.
 */
void write_stata_columns(span_2d_double matrix, ST_int j0)
{
  int r = 0; // Count each row that isn't filtered by Stata 'if'
  for (ST_int i = SF_in1(); i <= SF_in2(); i++) {
    if (SF_ifobs(i)) { // Skip rows according to Stata's 'if'
      for (ST_int j = j0; j < j0 + matrix.extent(0); j++) {
        // Convert MISSING back to Stata's missing value
        ST_double value = (matrix(j - j0, r) == MISSING) ? SV_missval : matrix(j - j0, r);
        ST_retcode rc = SF_vstore(j, i, value);
        if (rc) {
          throw std::runtime_error(fmt::format("Cannot write to Stata's variable {}", j));
        }
      }
      r += 1;
    }
  }
}

/*
 * Write data to columns in Stata (i.e. what Stata calls variables),
 * starting from column number 'j0'.
 */
void write_stata_columns(span_3d_double matrix, ST_int j0)
{
  for (int t = 0; t < matrix.extent(0); t++) {
    int r = 0; // Count each row that isn't filtered by Stata 'if'
    for (ST_int i = SF_in1(); i <= SF_in2(); i++) {
      if (SF_ifobs(i)) { // Skip rows according to Stata's 'if'
        for (ST_int j = j0; j < j0 + matrix.extent(2); j++) {
          // Convert MISSING back to Stata's missing value
          ST_double value = (matrix(t, r, j - j0) == MISSING) ? SV_missval : matrix(t, r, j - j0);
          ST_retcode rc = SF_vstore(j, i, value);
          if (rc) {
            throw std::runtime_error(fmt::format("Cannot write to Stata's variable {}", j));
          }
        }
        r += 1;
      }
    }

    j0 += (ST_int)matrix.extent(2);
  }
}

std::vector<double> stata_numlist(std::string macro)
{
  std::vector<double> numlist;

  char buffer[1000];
  SF_macro_use((char*)("_" + macro).c_str(), buffer, 1000);

  std::string list(buffer);
  size_t found = list.find(' ');
  while (found != std::string::npos) {
    std::string theta = list.substr(0, found);
    numlist.push_back(atof(theta.c_str()));
    list = list.substr(found + 1);
    found = list.find(' ');
  }
  numlist.push_back(atof(list.c_str()));

  return numlist;
}

/* Print to the Stata console the inputs to the plugin  */
void print_debug_info(int argc, char* argv[], Options opts, ManifoldGenerator generator, std::vector<bool> trainingRows,
                      std::vector<bool> predictionRows, bool pmani_flag, ST_int pmani, ST_int E, ST_int zcount,
                      ST_double dtWeight)
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

    for (int t = 0; t < opts.thetas.size(); t++) {
      io.print(fmt::format("theta = {:6.4f}\n\n", opts.thetas[t]));
    }
    io.print(fmt::format("algorithm = {}\n\n", opts.algorithm));
    io.print(fmt::format("force compute = {}\n\n", opts.forceCompute));
    io.print(fmt::format("missing distance = {:.06f}\n\n", opts.missingdistance));
    io.print(fmt::format("number of variables in manifold = {}\n\n", generator.E_actual()));
    io.print(fmt::format("train set obs: {}\n", std::accumulate(trainingRows.begin(), trainingRows.end(), 0)));
    io.print(fmt::format("predict set obs: {}\n\n", std::accumulate(predictionRows.begin(), predictionRows.end(), 0)));
    io.print(fmt::format("p_manifold flag = {}\n", pmani_flag));

    if (pmani_flag) {
      io.print(fmt::format("number of variables in p_manifold = {}\n", pmani));
    }
    io.print("\n");

    io.print(fmt::format("k = {}\n\n", opts.k));
    io.print(fmt::format("savePrediction = {}\n\n", opts.savePrediction));
    io.print(fmt::format("saveSMAPCoeffs = {}\n\n", opts.saveSMAPCoeffs));
    io.print(fmt::format("columns in smap coefficients = {}\n", opts.varssv));

    io.print(fmt::format("E is {}\n", E));
    io.print(fmt::format("We have {} 'extra' columns\n", zcount));
    io.print(fmt::format("Adding dt with weight {}\n", dtWeight));

    io.print(fmt::format("Requested {} threads\n", argv[9]));
    io.print(fmt::format("Using {} threads\n\n", opts.nthreads));

    io.flush();
  }
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

  Options opts;

  opts.thetas = stata_numlist("theta");
  double theta = atof(argv[0]);
  opts.k = atoi(argv[1]);
  opts.algorithm = std::string(argv[2]);
  opts.forceCompute = (strcmp(argv[3], "force") == 0);
  opts.missingdistance = atof(argv[4]);
  ST_int mani = atoi(argv[5]);    // number of columns in the manifold
  bool copredict = atoi(argv[6]); // contains the flag for p_manifold
  opts.calcRhoMAE = !copredict;
  opts.saveSMAPCoeffs = atoi(argv[7]);
  ST_int pmani = atoi(argv[8]);                  // contains the number of columns in p_manifold
  opts.varssv = opts.saveSMAPCoeffs ? pmani : 0; // number of columns in smap coefficients
  opts.nthreads = atoi(argv[9]);
  io.verbosity = atoi(argv[10]);

  double v;
  SF_scal_use("edm_xmap", &v);
  opts.xmap = (bool)v;
  if (opts.xmap && opts.calcRhoMAE) {
    SF_scal_use("edm_direction_num", &v);
    opts.xmapDirectionNum = (int)v;
  }

  SF_scal_use("store_prediction", &v);
  opts.savePrediction = (bool)v;

  char buffer[1001];

  // For multiple simultaneous edm calls, each is allocated a task number
  SF_macro_use("_task_num", buffer, 1000);
  opts.taskNum = atoi(buffer);

  // Find the number of lags 'E' for the main data.
  int E;
  if (copredict) {
    std::vector<double> E_list = stata_numlist("e");
    E = (int)E_list.back();
  } else {
    SF_macro_use("_i", buffer, 1000);
    E = atoi(buffer);
  }

  SF_macro_use("_parsed_dt", buffer, 1000);
  bool parsed_dt = (bool)atoi(buffer);
  double dtWeight = 0;
  if (parsed_dt) {
    SF_macro_use("_parsed_dtw", buffer, 1000);
    dtWeight = atof(buffer);
  }

  SF_macro_use("_zcount", buffer, 1000);
  int numExtras = atoi(buffer);

  // Default number of neighbours k is E_actual + 1
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

  // Read in the main data from Stata
  std::vector<ST_double> x = stata_columns<ST_double>(1);

  // Read in the target vector 'y' from Stata
  std::vector<ST_double> y = stata_columns<ST_double>(2);

  // Find which rows are used for training & which for prediction
  std::vector<bool> trainingRows = stata_columns<bool>(3);
  std::vector<bool> predictionRows = stata_columns<bool>(4);

  // Read in the prediction manifold
  std::vector<ST_double> co_x;
  if (copredict) {
    co_x = stata_columns<ST_double>(5);
  }

  // Read in the extras
  // TODO: Check that 'dt' isn't thrown in here in the edm.ado script
  std::vector<std::vector<ST_double>> extras(numExtras);

  for (int z = 0; z < numExtras; z++) {
    extras[z] = stata_columns<ST_double>(4 + copredict + 1 + z);
  }

  // Handle 'dt' flag
  // (We only need the time column in the case when 'dt' is set.)
  std::vector<ST_double> t;

  if (dtWeight > 0) {
    t = stata_columns<ST_double>(4 + copredict + numExtras + 1);
  }

  ManifoldGenerator generator(x, y, co_x, extras, t, E, dtWeight, MISSING);

  print_debug_info(argc, argv, opts, generator, trainingRows, predictionRows, copredict, pmani, E, numExtras, dtWeight);

  opts.thetas.clear();
  opts.thetas.push_back(theta);
  io.print(fmt::format("For now just doing theta = {}\n", theta));

#ifdef DUMP_INPUT
  // Here we want to dump the input so we can use it without stata for
  // debugging and profiling purposes.
  if (argc >= 12) {
    write_dumpfile(argv[11], opts, x, y, co_x, extras, t, E, dtWeight, trainingRows, predictionRows);
  }
#endif

  predictions.push(std::async(std::launch::async, mf_smap_loop, opts, generator, trainingRows, predictionRows, io,
                              keep_going, finished));

  return SUCCESS;
}

ST_retcode save_prediction_to_stata_variables()
{
  Prediction pred = predictions.front().get();
  predictions.pop();

  // If there are no errors, return the value of ystar (and smap coefficients) to Stata.
  if (pred.rc == SUCCESS) {
    if (pred.ystar != nullptr) {
      auto ystar = span_2d_double(pred.ystar.get(), (int)pred.numThetas, (int)pred.numPredictions);
      write_stata_columns(ystar, 1);
    }

    if (pred.coeffs != nullptr) {
      auto coeffs =
        span_3d_double(pred.coeffs.get(), (int)pred.numThetas, (int)pred.numPredictions, (int)pred.numCoeffCols);
      write_stata_columns(coeffs, (pred.ystar != nullptr) + 1);
    }
  }

  return pred.rc;
}

STDLL stata_call(int argc, char* argv[])
{
  try {
    if (argc > 0) {
      return edm(argc, argv);
    } else {
      ST_retcode rc = save_prediction_to_stata_variables();
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
