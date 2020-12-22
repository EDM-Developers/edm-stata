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

// These are all the variables in the edm.ado script we modify in the plugin.
// These definitions also suppress the "C++ doesn't permit string literals as char*" warnings.
char* PRINT_MACRO = (char*)"_edm_print";
char* RUNNING_SCALAR = (char*)"edm_running";

class StataIO : public IO
{
public:
  virtual void out(const char* s) const { SF_display((char*)s); }
  virtual void error(const char* s) const { SF_error((char*)s); }
  virtual void flush() const { _stata_->spoutflush(); }
  virtual void out_async(const char* s) const
  {
#ifdef _MSC_VER
    out(s);
  }
#else
    SF_macro_use(PRINT_MACRO, buffer, BUFFER_SIZE);
    strcat(buffer, s);
    SF_macro_save(PRINT_MACRO, buffer);
  }

private:
  static const size_t BUFFER_SIZE = 1000;
  mutable char buffer[BUFFER_SIZE];
#endif
};

// Global state, needed to persist between multiple edm calls
StataIO io;
Options opts;
ManifoldGenerator generator;
std::queue<Prediction> predictions;
std::queue<std::future<void>> futures;

std::mutex rwStataScalar;

bool keep_going()
{
  double edm_running;
  {
    std::lock_guard<std::mutex> guard(rwStataScalar);
    SF_scal_use(RUNNING_SCALAR, &edm_running);
  }
  return (bool)edm_running;
}

void all_tasks_finished()
{
  {
    std::lock_guard<std::mutex> guard(rwStataScalar);
    SF_scal_save(RUNNING_SCALAR, 0.0);
  }
}

void print_error(std::string command, ST_retcode rc)
{
  if (rc != SUCCESS) {
    io.error((char*)fmt::format("Error in edm '{}': ", command).c_str());
  }
  switch (rc) {
    case TOO_FEW_VARIABLES:
      io.error("Too few arguments\n");
      break;
    case TOO_MANY_VARIABLES:
      io.error("Too many arguments\n");
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
    case UNKNOWN_ERROR:
      io.error("Unknown error\n");
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
 * Write data to a column in Stata (i.e. what Stata calls variables),
 * starting from column number 'j'.
 */
void write_stata_column(ST_double* data, size_t len, ST_int j)
{
  int r = 0; // Count each row that isn't filtered by Stata 'if'
  for (ST_int i = SF_in1(); i <= SF_in2(); i++) {
    if (SF_ifobs(i)) { // Skip rows according to Stata's 'if'
      // Convert MISSING back to Stata's missing value
      ST_double value = (data[r] == MISSING) ? SV_missval : data[r];
      ST_retcode rc = SF_vstore(j, i, value);
      if (rc) {
        throw std::runtime_error(fmt::format("Cannot write to Stata's variable {}", j));
      }
      r += 1;
      if (r >= len) {
        break;
      }
    }
  }
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
      for (ST_int j = j0; j < j0 + matrix.extent(1); j++) {
        // Convert MISSING back to Stata's missing value
        ST_double value = (matrix(r, j - j0) == MISSING) ? SV_missval : matrix(r, j - j0);
        ST_retcode rc = SF_vstore(j, i, value);
        if (rc) {
          throw std::runtime_error(fmt::format("Cannot write to Stata's variable {}", j));
        }
      }
      r += 1;
      if (r >= matrix.extent(0)) {
        break;
      }
    }
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
void print_setup_info(int argc, char* argv[], char* reqThreads, ST_int numExtras, ST_double dtWeight)
{
  if (io.verbosity > 1) {
    // Overview of variables and arguments passed and observations in sample
    io.print(fmt::format("number of vars & obs = {}, {}\n", SF_nvars(), SF_nobs()));
    io.print(fmt::format("first and last obs in sample = {}, {}\n\n", SF_in1(), SF_in2()));

    for (int i = 0; i < argc; i++) {
      io.print(fmt::format("arg {}: {}\n", i, argv[i]));
    }
    io.print("\n");

    io.print(fmt::format("algorithm = {}\n\n", opts.algorithm));
    io.print(fmt::format("force compute = {}\n\n", opts.forceCompute));
    io.print(fmt::format("missing distance = {:.06f}\n\n", opts.missingdistance));

    io.print(fmt::format("We have {} 'extra' columns\n", numExtras));
    io.print(fmt::format("Adding dt with weight {}\n", dtWeight));

    io.print(fmt::format("Requested {} threads\n", reqThreads));
    io.print(fmt::format("Using {} threads\n\n", opts.nthreads));

    ST_int npcores = (ST_int)num_physical_cores();
    ST_int nlcores = (ST_int)num_logical_cores();
    io.print(fmt::format("System has {} physical cores <= {} logical cores\n", npcores, nlcores));

    io.flush();
  }
}

/* Print to the Stata console the inputs to the plugin  */
void print_launch_info(int argc, char* argv[], Options taskOpts, std::vector<bool> trainingRows,
                       std::vector<bool> predictionRows, ST_int E)
{
  if (io.verbosity > 1) {

    for (int i = 0; i < argc; i++) {
      io.print(fmt::format("arg {}: {}\n", i, argv[i]));
    }
    io.print("\n");

    for (int t = 0; t < taskOpts.thetas.size(); t++) {
      io.print(fmt::format("theta = {:6.4f}\n\n", taskOpts.thetas[t]));
    }

    // io.print(fmt::format("number of variables in manifold = {}\n\n", generator.E_actual()));
    io.print(fmt::format("train set obs: {}\n", std::accumulate(trainingRows.begin(), trainingRows.end(), 0)));
    io.print(fmt::format("predict set obs: {}\n\n", std::accumulate(predictionRows.begin(), predictionRows.end(), 0)));

    io.print(fmt::format("k = {}\n\n", taskOpts.k));
    io.print(fmt::format("savePrediction = {}\n\n", taskOpts.savePrediction));
    io.print(fmt::format("saveSMAPCoeffs = {}\n\n", taskOpts.saveSMAPCoeffs));

    io.print(fmt::format("E is {}\n", E));
    io.flush();
  }
}

/*
 * Read that information needed for the edm tasks which is doesn't change across
 * the various tasks, and store it in the 'opts' and 'generator' global variables.
 */
ST_retcode read_manifold_data_from_stata(int argc, char* argv[])
{
  if (argc < 8) {
    return TOO_FEW_VARIABLES;
  }
  if (argc > 8) {
    return TOO_MANY_VARIABLES;
  }

  opts.calcRhoMAE = true;

  int numExtras = atoi(argv[0]);
  double dtWeight = atof(argv[1]);
  opts.algorithm = std::string(argv[2]);
  opts.forceCompute = (std::string(argv[3]) == "force");
  opts.missingdistance = atof(argv[4]);
  char* reqThreads = argv[5];
  opts.nthreads = atoi(reqThreads);
  io.verbosity = atoi(argv[6]);
  opts.numTasks = atoi(argv[7]);

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

  // Read in the extras
  // TODO: Check that 'dt' isn't thrown in here in the edm.ado script
  std::vector<std::vector<ST_double>> extras(numExtras);

  for (int z = 0; z < numExtras; z++) {
    extras[z] = stata_columns<ST_double>(3 + z);
  }

  generator = ManifoldGenerator(x, y, extras, MISSING);

  // Handle 'dt' flag
  if (dtWeight > 0) {
    std::vector<ST_double> t = stata_columns<ST_double>(2 + numExtras + 1);
    generator.add_dt_data(t, dtWeight);
  }

  print_setup_info(argc, argv, reqThreads, numExtras, dtWeight);

  return SUCCESS;
}

ST_retcode launch_edm_task(int argc, char* argv[])
{
  if (argc < 5) {
    return TOO_FEW_VARIABLES;
  }
  if (argc > 6) {
    return TOO_MANY_VARIABLES;
  }

  Options taskOpts = opts;

  taskOpts.taskNum = futures.size();

  // Find the number of lags 'E' for the main data.
  int E = atoi(argv[0]);
  int E_actual = (int)generator.E_actual(E);

  taskOpts.thetas.push_back(atof(argv[1]));
  taskOpts.k = atoi(argv[2]);

  taskOpts.savePrediction = atoi(argv[3]);
  taskOpts.saveSMAPCoeffs = atoi(argv[4]);

  // Default number of neighbours k is E_actual + 1
  if (taskOpts.k <= 0) {
    taskOpts.k = E_actual + 1;
  }

  // Find which rows are used for training & which for prediction
  std::vector<bool> trainingRows = stata_columns<bool>(1);
  std::vector<bool> predictionRows = stata_columns<bool>(2);

#ifdef DUMP_INPUT
  if (argc == 6) {
    io.print("Dumping inputs to hdf5 file\n");
    io.flush();
    write_dumpfile(argv[5], taskOpts, generator, E, trainingRows, predictionRows);
  }
#endif

  predictions.push({});
  futures.push(edm_async(taskOpts, &generator, E, trainingRows, predictionRows, &io, &(predictions.back()), keep_going,
                         all_tasks_finished));

  return SUCCESS;
}

ST_retcode launch_coprediction_task(int argc, char* argv[])
{
  if (argc < 3) {
    return TOO_FEW_VARIABLES;
  }
  if (argc > 4) {
    return TOO_MANY_VARIABLES;
  }

  Options taskOpts = opts;

  // Just one task when in coprediction mode.
  taskOpts.taskNum = 0;
  taskOpts.numTasks = 1;

  // Always saving prediction vector in coprediction mode.
  // Never calculating rho & MAE statistics in this mode.
  // Never saving SMAP coefficients in coprediction mode.
  taskOpts.savePrediction = 1;
  taskOpts.calcRhoMAE = 0;
  taskOpts.saveSMAPCoeffs = false;

  int E = atoi(argv[0]);
  int E_actual = (int)generator.E_actual(E);

  taskOpts.thetas.push_back(atof(argv[1]));
  taskOpts.k = atoi(argv[2]);

  // Default number of neighbours k is E_actual + 1
  if (taskOpts.k <= 0) {
    taskOpts.k = E_actual + 1;
  }

  // Add co_x directly to the manifold generator.
  generator.add_coprediction_data(stata_columns<ST_double>(1));

  // Find which rows are used for training & which for prediction
  std::vector<bool> coTrainingRows = stata_columns<bool>(2);
  std::vector<bool> coPredictionRows = stata_columns<bool>(3);

#ifdef DUMP_INPUT
  if (argc == 4) {
    write_dumpfile(argv[3], taskOpts, generator, E, coTrainingRows, coPredictionRows);
  }
#endif

  predictions.push({});
  futures.push(edm_async(taskOpts, &generator, E, coTrainingRows, coPredictionRows, &io, &(predictions.back()),
                         keep_going, all_tasks_finished));

  return SUCCESS;
}

ST_retcode save_all_task_results_to_stata(int argc, char* argv[])
{
  if (argc > 1) {
    return TOO_MANY_VARIABLES;
  }

  char* resultMatrix = nullptr;
  if (argc == 1) {
    resultMatrix = argv[0];
  }

  ST_retcode rc = 0;
  size_t numCoeffColsSaved = 0;

  while (predictions.size() > 0) {
    std::future<void>& fut = futures.front();
    fut.get();
    futures.pop();

    // If there are no errors, store the prediction ystar and smap coefficients to Stata variables.
    const Prediction& pred = predictions.front();
    if (pred.rc == SUCCESS) {
      // Save the rho/MAE results if requested (i.e. not for coprediction)
      if (pred.stats.calcRhoMAE) {
        if (SF_mat_store(resultMatrix, pred.stats.taskNum + 1, 3, pred.stats.rho)) {
          io.print(fmt::format("Error: failed to save rho {} to matrix '{}'\n", pred.stats.rho, resultMatrix));
        }

        if (SF_mat_store(resultMatrix, pred.stats.taskNum + 1, 4, pred.stats.mae)) {
          io.print(fmt::format("Error: failed to save MAE {} to matrix '{}'\n", pred.stats.mae, resultMatrix));
        }
      }

      if (pred.ystar != nullptr) {
        write_stata_column(pred.ystar.get(), pred.numPredictions, 1);
      }

      if (pred.coeffs != nullptr) {
        auto coeffs = span_2d_double(pred.coeffs.get(), (int)pred.numPredictions, (int)pred.numCoeffCols);

        write_stata_columns(coeffs, (pred.ystar != nullptr) + numCoeffColsSaved + 1);
        numCoeffColsSaved += pred.numCoeffCols;
      }
    }

    if (pred.rc > rc) {
      rc = pred.rc;
    }

    predictions.pop();
  }

  return rc;
}

STDLL stata_call(int argc, char* argv[])
{
  try {
    ST_retcode rc;
    std::string command(argv[0]);

    if (command == "transfer_manifold_data") {
      rc = read_manifold_data_from_stata(argc - 1, argv + 1);
    } else if (command == "launch_edm_task") {
      rc = launch_edm_task(argc - 1, argv + 1);
    } else if (command == "collect_results") {
      rc = save_all_task_results_to_stata(argc - 1, argv + 1);
    } else if (command == "launch_coprediction_task") {
      rc = launch_coprediction_task(argc - 1, argv + 1);
    } else {
      rc = UNKNOWN_ERROR;
    }

    print_error(command, rc);
    return rc;
  } catch (const std::exception& e) {
    io.error(e.what());
    io.error("\n");
  } catch (...) {
    io.error("Unknown error in edm plugin\n");
  }
  return UNKNOWN_ERROR;
}
