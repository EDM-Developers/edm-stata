#pragma warning(disable : 4018)

#include "cpu.h"
#include "edm.h"
#include "stplugin.h"

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Core>

#include <algorithm>
#include <future>
#include <numeric> // for std::accumulate
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "cli.h" // to save the inputs to a local file for debugging
#include "stats.h"
#include "train_predict_split.h" // Just for 'TrainPredictSplitter::requiresRandomNumbers', can simplify

const double PI = 3.141592653589793238463;

// These are all the variables in the edm.ado script we modify in the plugin.
// These definitions also suppress the "C++ doesn't permit string literals as char*" warnings.
char* FINISHED_SCALAR = (char*)"plugin_finished";
char* MISSING_DISTANCE_USED = (char*)"_missing_dist_used";
char* RNG_STATE = (char*)"_rngstate";

char* NUM_NEIGHBOURS = (char*)"_k";
char* SAVE_PREDICTION = (char*)"_predict";
char* SAVE_SMAP = (char*)"_savesmap";
char* SAVE_INPUTS = (char*)"_saveinputs";
char* NUM_REPS = (char*)"_round";

class StataIO : public IO
{
public:
  virtual void out(const char* s) const { SF_display((char*)s); }
  virtual void error(const char* s) const { SF_error((char*)s); }
  virtual void flush() const { _stata_->spoutflush(); }
};

// Global state, needed to persist between multiple edm calls
StataIO io;
std::atomic<bool> breakButtonPressed = false;
std::atomic<bool> allTasksFinished = false;

bool keep_going()
{
  return !breakButtonPressed;
}

void all_tasks_finished()
{
  allTasksFinished = true;
}

void print_error(std::string command, ST_retcode rc)
{
  // Don't print header if rc=SUCCESS or rc=1 (when Break button pressed)
  if (rc > 1 && io.verbosity > 1) {
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
    case INVALID_DISTANCE:
      io.error("Invalid distance argument\n");
      break;
    case INVALID_METRICS:
      io.error("Invalid metrics argument\n");
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
 * Write data to a column number 'j' in Stata (i.e. to a Stata 'variable').
 * If supplied, we consider each row 'i' only if filter[i] == true.
 */
void write_stata_column(ST_double* data, int len, ST_int j, const std::vector<bool>& filter = {})
{
  bool useFilter = (filter.size() > 0);
  int obs = 0;
  int r = 0; // Count each row that isn't filtered by Stata 'if'
  for (ST_int i = SF_in1(); i <= SF_in2(); i++) {
    if (SF_ifobs(i)) { // Skip rows according to Stata's 'if'
      if ((useFilter && filter[r]) || !useFilter) {
        // Convert MISSING back to Stata's missing value
        ST_double value = (data[obs] == MISSING) ? SV_missval : data[obs];
        ST_retcode rc = SF_vstore(j, i, value);
        if (rc) {
          throw std::runtime_error(fmt::format("Cannot write to Stata's variable {}", j));
        }
        obs += 1;
      }
      r += 1;
      if (obs >= len) {
        break;
      }
    }
  }
}

/*
 * Write data to columns ('variables') in Stata, starting from column number 'j0'.
 * If supplied, we consider each row 'i' only if filter[i] == true.
 */
void write_stata_columns(double* matrix, int matrixNumRows, int matrixNumCols, ST_int j0,
                         const std::vector<bool>& filter = {})
{
  bool useFilter = (filter.size() > 0);
  int obs = 0;
  int r = 0; // Count each row that isn't filtered by Stata 'if'
  for (ST_int i = SF_in1(); i <= SF_in2(); i++) {
    if (SF_ifobs(i)) { // Skip rows according to Stata's 'if'
      if ((useFilter && filter[r]) || !useFilter) {
        for (ST_int j = j0; j < j0 + matrixNumCols; j++) {
          // Convert MISSING back to Stata's missing value
          ST_double value = matrix[(j - j0) * matrixNumRows + obs];
          if (value == MISSING) {
            value = SV_missval;
          }
          ST_retcode rc = SF_vstore(j, i, value);
          if (rc) {
            throw std::runtime_error(fmt::format("Cannot write to Stata's variable {}", j));
          }
        }
        obs += 1;
      }
      r += 1;
      if (obs >= matrixNumRows) {
        break;
      }
    }
  }
}

std::vector<std::string> split_string(std::string list)
{
  std::vector<std::string> splitList;

  size_t found = list.find(' ');
  while (found != std::string::npos) {
    std::string part = list.substr(0, found);
    splitList.push_back(part);
    list = list.substr(found + 1);
    found = list.find(' ');
  }

  if (!list.empty()) {
    splitList.push_back(list);
  }

  return splitList;
}

template<typename T>
std::vector<T> numlist_to_vector(std::string list)
{
  std::vector<T> numList;

  size_t found = list.find(' ');
  while (found != std::string::npos) {
    std::string theta = list.substr(0, found);
    numList.push_back(atof(theta.c_str()));
    list = list.substr(found + 1);
    found = list.find(' ');
  }

  if (!list.empty()) {
    numList.push_back((T)atof(list.c_str()));
  }

  return numList;
}

template<typename T>
std::vector<T> stata_numlist(std::string macro)
{
  char buffer[1000];
  SF_macro_use((char*)("_" + macro).c_str(), buffer, 1000);

  std::string list(buffer);
  return numlist_to_vector<T>(list);
}

template<typename T>
void print_vector(std::string name, std::vector<T> vec)
{
  if (io.verbosity > 1) {
    io.print(fmt::format("{} [{}]:\n", name, vec.size()));
    for (int i = 0; i < vec.size(); i++) {
      if (i == 10) {
        io.print("... ");
        continue;
      }
      if (i > 10 && i < vec.size() - 10) {
        continue;
      }
      io.print(fmt::format("{} ", vec[i]));
    }
    io.print("\n");
  }
}

/* Print to the Stata console the inputs to the plugin  */
void print_setup_info(int argc, char* argv[], const Options& opts, char* reqThreads, ST_int numExtras, bool dtMode,
                      ST_double dtWeight, const std::vector<Metric>& metrics)
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
    io.print(fmt::format("missing distance = {}\n\n", opts.missingdistance));

    io.print(fmt::format("We have {} 'extra' columns\n", numExtras));
    if (dtMode) {
      io.print(fmt::format("Adding dt with weight {}\n", dtWeight));
    }

    io.print("Metrics:");
    for (const Metric& m : metrics) {
      if (m == Metric::Diff) {
        io.print(" Diff");
      } else {
        io.print(" CheckSame");
      }
    }
    io.print("\n");

    io.print(fmt::format("Requested {} threads\n", reqThreads));
    io.print(fmt::format("Using {} threads\n\n", opts.nthreads));

    ST_int npcores = (ST_int)num_physical_cores();
    ST_int nlcores = (ST_int)num_logical_cores();
    io.print(fmt::format("System has {} physical cores <= {} logical cores\n", npcores, nlcores));

    io.flush();
  }
}

/* Print to the Stata console the inputs to the plugin  */
void print_launch_info(const ManifoldGenerator& generator, Options& taskOpts, std::vector<bool>& trainingRows,
                       std::vector<bool>& predictionRows, ST_int E)
{
  if (io.verbosity > 1) {
    for (int t = 0; t < taskOpts.thetas.size(); t++) {
      io.print(fmt::format("theta = {:6.4f}\n\n", taskOpts.thetas[t]));
    }

    io.print(fmt::format("train set obs: {}\n", std::accumulate(trainingRows.begin(), trainingRows.end(), 0)));
    io.print(fmt::format("predict set obs: {}\n\n", std::accumulate(predictionRows.begin(), predictionRows.end(), 0)));

    io.print(fmt::format("k = {}\n\n", taskOpts.k));
    io.print(fmt::format("savePrediction = {}\n\n", taskOpts.savePrediction));
    io.print(fmt::format("saveSMAPCoeffs = {}\n\n", taskOpts.saveSMAPCoeffs));

    io.print(fmt::format("E is {}\n", E));
    io.flush();

    if (io.verbosity > 2) {
      auto M = generator.create_manifold(E, trainingRows, false);
      auto Mp = generator.create_manifold(E, predictionRows, true);

      print_vector<bool>("training rows", trainingRows);
      print_vector<bool>("prediction rows", predictionRows);

      io.print("dt\n");
      for (int i = 0; i < M.nobs(); i++) {
        io.print(fmt::format("[{}] dt0 = {} dt1 = {}\n", i, M.dt(i, 0), M.dt(i, 1)));
        if (i > 5) {
          break;
        }
      }
      io.print("\n");

      io.print("M Manifold\n");
      for (int i = 0; i < M.nobs(); i++) {
        for (int j = 0; j < M.E_actual(); j++) {
          io.print(fmt::format("{} ", M(i, j)));
        }
        io.print("\n");
      }
      io.print("\n");

      io.print("Mp Manifold\n");
      for (int i = 0; i < Mp.nobs(); i++) {
        for (int j = 0; j < Mp.E_actual(); j++) {
          io.print(fmt::format("{} ", Mp(i, j)));
        }
        io.print("\n");
      }
    }
  }
}

double default_missing_distance(std::vector<double> x, std::vector<bool> usable)
{
  std::vector<double> x_usable;
  for (int i = 0; i < x.size(); i++) {
    if (usable[i] && x[i] != MISSING) {
      x_usable.push_back(x[i]);
    }
  }

  Eigen::Map<const Eigen::ArrayXd> xMap(x_usable.data(), x_usable.size());
  const Eigen::ArrayXd xCent = xMap - xMap.mean();
  double xSD = std::sqrt((xCent * xCent).sum() / (xCent.size() - 1));
  double defaultMissingDist = 2 / sqrt(PI) * xSD;

  return defaultMissingDist;
}

Metric guess_appropriate_metric(std::vector<ST_double> data, int targetSample = 100)
{
  std::unordered_set<double> uniqueValues;

  int sampleSize = 0;
  for (int i = 0; i < data.size() && sampleSize < targetSample; i++) {
    if (data[i] != MISSING) {
      sampleSize += 1;
      uniqueValues.insert(data[i]);
    }
  }

  if (uniqueValues.size() <= 10) {
    // The data is likely binary or categorical, calculate the indicator function for two values being identical
    return Metric::CheckSame;
  } else {
    // The data is likely continuous, just take differences between the values
    return Metric::Diff;
  }
}

// In case we have some remnants of previous runs still
// in the system (e.g. after a 'break'), clear our past results.
void reset_global_state()
{
  io.get_and_clear_async_buffer();
  breakButtonPressed = false;
  allTasksFinished = false;
}

std::vector<bool> generate_usable(std::vector<bool> touse, ManifoldGenerator& generator, int maxE, bool allowMissing)
{
  // Make the largest manifold we'll need in order to find missing values for 'usable'
  std::vector<bool> allTrue(touse.size());
  for (int i = 0; i < allTrue.size(); i++) {
    allTrue[i] = true;
  }

  Manifold M = generator.create_manifold(maxE, allTrue, false);

  // Generate the 'usable' variable
  std::vector<bool> usable(touse.size());
  for (int i = 0; i < usable.size(); i++) {
    if (allowMissing) {
      usable[i] = touse[i] && M.any_not_missing(i) && M.y(i) != MISSING;
    } else {
      usable[i] = touse[i] && !M.any_missing(i) && M.y(i) != MISSING;
    }
  }

  return usable;
}

/*
 * Read that information needed for the edm tasks which is doesn't change across
 * the various tasks, and store it in the 'opts' and 'generator' global variables.
 */
ST_retcode launch_edm_tasks(int argc, char* argv[])
{
  if (argc < 22) {
    return TOO_FEW_VARIABLES;
  }
  if (argc > 22) {
    return TOO_MANY_VARIABLES;
  }

  reset_global_state();

  Options opts;
  opts.copredict = false;

  opts.calcRhoMAE = true;
  int numExtras = atoi(argv[0]);
  bool dtMode = atoi(argv[1]);
  bool dt0 = atoi(argv[2]);
  double dtWeight = atof(argv[3]);
  std::string alg = std::string(argv[4]);
  if (alg.empty() || alg == "simplex") {
    opts.algorithm = Algorithm::Simplex;
  } else if (alg == "smap") {
    opts.algorithm = Algorithm::SMap;
  } else if (alg == "llr") {
    return NOT_IMPLEMENTED;
  } else {
    return INVALID_ALGORITHM;
  }
  opts.forceCompute = (std::string(argv[5]) == "force");
  opts.missingdistance = atof(argv[6]);
  char* reqThreads = argv[7];
  opts.nthreads = atoi(reqThreads);
  io.verbosity = atoi(argv[8]);
  opts.numTasks = atoi(argv[9]);
  bool explore = atoi(argv[10]);
  bool full = atoi(argv[11]);
  int crossfold = atoi(argv[12]);
  int tau = atoi(argv[13]);
  int maxE = atoi(argv[14]);
  bool allowMissing = atoi(argv[15]);
  double nextRV = std::stod(argv[16]);
  opts.thetas = numlist_to_vector<double>(std::string(argv[17]));
  opts.aspectRatio = atof(argv[18]);
  std::string distance(argv[19]);
  std::string requestedMetrics(argv[20]);
  opts.cmdLine = argv[21];

  auto factorVariables = stata_numlist<bool>("factor_var");
  auto extrasEVarying = stata_numlist<bool>("z_e_varying");

  if (distance == "l1" || distance == "L1" || distance == "mae" || distance == "MAE") {
    opts.distance = Distance::MeanAbsoluteError;
  } else if (distance == "l2" || distance == "L2" || distance == "euclidean" || distance == "Euclidean") {
    opts.distance = Distance::Euclidean;
  } else if (distance == "wasserstein" || distance == "Wasserstein") {
    opts.distance = Distance::Wasserstein;
  } else {
    return INVALID_DISTANCE;
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
  std::vector<ST_double> x = stata_columns<ST_double>(2);

  // Read in the target vector 'y' from Stata
  std::vector<ST_double> y = stata_columns<ST_double>(3);

  // Read in the extras
  std::vector<std::vector<ST_double>> extras(numExtras);

  for (int z = 0; z < numExtras; z++) {
    extras[z] = stata_columns<ST_double>(4 + z);
  }

  std::vector<Metric> metrics;
  if (requestedMetrics == "auto" || requestedMetrics.empty()) {
    for (bool isFactorVariable : factorVariables) {
      metrics.push_back(isFactorVariable ? Metric::CheckSame : Metric::Diff);
    }
  } else {
    for (std::string& metric : split_string(requestedMetrics)) {
      if (metric == "same" || metric == "indicator" || metric == "onehot") {
        metrics.push_back(Metric::CheckSame);
      } else {
        metrics.push_back(Metric::Diff);
      }
    }

    // If the user supplied fewer than the required number of metrics,
    // just repeat the last one to pad out the list.
    while (metrics.size() < 1 + numExtras) {
      metrics.push_back(metrics.back());
    }
  }
  opts.metrics = metrics;

  ManifoldGenerator generator = ManifoldGenerator(x, y, extras, extrasEVarying, MISSING, tau);

  // Handle 'dt' flag
  if (dtMode) {
    std::vector<ST_double> t = stata_columns<ST_double>(1);
    print_vector<ST_double>("t", t);
    generator.add_dt_data(t, dtWeight, dt0);
  }

  // The stata variable named `touse'
  std::vector<bool> touse = stata_columns<bool>(3 + numExtras + 1);
  print_vector<bool>("touse", touse);

  std::vector<bool> usable = generate_usable(touse, generator, maxE, allowMissing);
  print_vector<bool>("usable", usable);

  if (allowMissing && opts.missingdistance == 0) {
    opts.missingdistance = default_missing_distance(x, usable);
  }
  SF_macro_save(MISSING_DISTANCE_USED, (char*)fmt::format("{}", opts.missingdistance).c_str());

  // If we need to create a randomised train/predict split, then sync the state of the
  // Mersenne Twister in Stata to that in the splitter instance.
  bool requiresRandomNumbers = TrainPredictSplitter::requiresRandomNumbers(crossfold, full);
  std::string rngState;

  if (requiresRandomNumbers) {
    char buffer[5200]; // Need at least 5011 + 1 bytes.
    if (SF_macro_use(RNG_STATE, buffer, 5200)) {
      io.print("Got an error rc from macro_use!\n");
    }

    rngState = std::string(buffer);

    if (rngState.empty()) {
      io.print("Error: couldn't read the c(rngstate).\n");
    }
  }

  print_setup_info(argc, argv, opts, reqThreads, numExtras, dtMode, dtWeight, opts.metrics);

  auto usableToSave = std::make_unique<double[]>(usable.size());
  for (int i = 0; i < usable.size(); i++) {
    usableToSave[i] = usable[i];
  }
  write_stata_column(usableToSave.get(), (int)usable.size(), 3 + numExtras + 2);

  // Read in some macros from Stata
  char buffer[200];

  // What is k?
  if (SF_macro_use(NUM_NEIGHBOURS, buffer, 200)) {
    io.print("Got an error rc from macro_use!\n");
  }
  int k = atoi(buffer);

  // Are we saving the predictions?
  if (SF_macro_use(SAVE_PREDICTION, buffer, 200)) {
    io.print("Got an error rc from macro_use!\n");
  }
  bool saveFinalPredictions = !(std::string(buffer).empty());

  // Are we saving the S-map coefficients (only in xmap mode)?
  bool saveSMAPCoeffs;
  if (explore) {
    saveSMAPCoeffs = false;
  } else {
    if (SF_macro_use(SAVE_SMAP, buffer, 200)) {
      io.print("Got an error rc from macro_use!\n");
    }
    saveSMAPCoeffs = !(std::string(buffer).empty());
  }

  // Are we saving the inputs to a JSON file?
  if (SF_macro_use(SAVE_INPUTS, buffer, 200)) {
    io.print("Got an error rc from macro_use!\n");
  }
  std::string saveInputsFilename(buffer);

  // Number of replications
  if (SF_macro_use(NUM_REPS, buffer, 200)) {
    io.print("Got an error rc from macro_use!\n");
  }
  int numReps = atoi(buffer);

  std::vector<int> Es = stata_numlist<int>("e");

  std::vector<int> libraries;
  if (!explore) {
    libraries = stata_numlist<int>("l_ori"); // The 'library' macro gets overwritten
    if (libraries.empty()) {
      int numUsable = std::accumulate(usable.begin(), usable.end(), 0);
      libraries.push_back(numUsable);
    }
  }

  // If requested, save the inputs to a local file for testing
  if (!saveInputsFilename.empty()) {
    if (io.verbosity > 1) {
      io.print(fmt::format("Saving inputs to '{}.json'\n", saveInputsFilename));
      io.flush();
    }

    json taskGroup;

    taskGroup["generator"] = generator;
    taskGroup["opts"] = opts;

    taskGroup["Es"] = Es;
    taskGroup["libraries"] = libraries;
    taskGroup["k"] = k;
    taskGroup["numReps"] = numReps;
    taskGroup["crossfold"] = crossfold;
    taskGroup["explore"] = explore;
    taskGroup["full"] = full;
    taskGroup["saveFinalPredictions"] = saveFinalPredictions;
    taskGroup["saveSMAPCoeffs"] = saveSMAPCoeffs;
    taskGroup["usable"] = usable;
    taskGroup["rngState"] = rngState;
    taskGroup["nextRV"] = nextRV;

    append_to_dumpfile(saveInputsFilename + ".json", taskGroup);
  }

  return launch_task_group(generator, opts, Es, libraries, k, numReps, crossfold, explore, full, saveFinalPredictions,
                           saveSMAPCoeffs, usable, rngState, nextRV, &io, keep_going, all_tasks_finished);
}

ST_retcode prepare_coprediction_task(int argc, char* argv[])
{
  if (argc < 3) {
    return TOO_FEW_VARIABLES;
  }
  if (argc > 3) {
    return TOO_MANY_VARIABLES;
  }

  //  reset_global_state();

  int E = atoi(argv[0]);
  int k = atoi(argv[1]);
  std::string saveInputsFilename(argv[2]);

  std::vector<ST_double> co_x = stata_columns<ST_double>(1);
  std::vector<bool> coTrainingRows = stata_columns<bool>(2);
  std::vector<bool> coPredictionRows = stata_columns<bool>(3);

  //  if (!saveInputsFilename.empty()) {
  //    json taskGroup;
  //    //    taskGroup["generator"] = generator;
  //    //    taskGroup["opts"] = taskOpts;
  //    taskGroup["E"] = E;
  //    taskGroup["trainingRows"] = coTrainingRows;
  //    taskGroup["predictionRows"] = coPredictionRows;
  //
  //    append_to_dumpfile(saveInputsFilename + ".json", taskGroup);
  //  }

  //  if (io.verbosity > 2) {
  //    auto M = generator.create_manifold(E, coTrainingRows, false);
  //    auto Mp = generator.create_manifold(E, coPredictionRows, true);
  //    io.print("Coprediction M Manifold\n");
  //    for (int i = 0; i < M.nobs(); i++) {
  //      for (int j = 0; j < M.E_actual(); j++) {
  //        io.print(fmt::format("{} ", M(i, j)));
  //      }
  //      io.print("\n");
  //    }
  //    io.print("\n");
  //
  //    io.print("Coprediction  Mp Manifold\n");
  //    for (int i = 0; i < Mp.nobs(); i++) {
  //      for (int j = 0; j < Mp.E_actual(); j++) {
  //        io.print(fmt::format("{} ", Mp(i, j)));
  //      }
  //      io.print("\n");
  //    }
  //  }

  return launch_coprediction_task(E, k, co_x, coTrainingRows, coPredictionRows, &io, keep_going, all_tasks_finished);
}

ST_retcode save_all_task_results_to_stata(int argc, char* argv[])
{
  if (argc < 3) {
    return TOO_FEW_VARIABLES;
  }
  if (argc > 3) {
    return TOO_MANY_VARIABLES;
  }

  char* resultMatrix = argv[0];
  bool savePredictMode = atoi(argv[1]);
  bool saveCoPredictMode = atoi(argv[2]);

  ST_retcode rc = 0;

  int numCoeffColsSaved = 0;

  std::queue<Prediction>& predictions = get_results();

  while (predictions.size() > 0) {

    // If there are no errors, store the prediction ystar and smap coefficients to Stata variables.
    const Prediction& pred = predictions.front();
    if (pred.rc == SUCCESS) {
      // Save the rho/MAE results if requested (i.e. not for coprediction)
      for (auto& stats : pred.stats) {

        if (SF_mat_store(resultMatrix, stats.taskNum + 1, 3, stats.rho)) {
          io.error(fmt::format("Error: failed to save rho {} to matrix '{}[{},{}]'\n", stats.rho, resultMatrix,
                               stats.taskNum + 1, 3)
                     .c_str());
          rc = CANNOT_SAVE_RESULTS;
        }

        if (SF_mat_store(resultMatrix, stats.taskNum + 1, 4, stats.mae)) {
          io.error(fmt::format("Error: failed to save MAE {} to matrix '{}[{},{}]'\n", stats.mae, resultMatrix,
                               stats.taskNum + 1, 4)
                     .c_str());
          rc = CANNOT_SAVE_RESULTS;
        }
      }

      if (pred.ystar != nullptr) {
        if (!pred.copredict) {
          write_stata_column(pred.ystar.get(), pred.numPredictions, 1, pred.predictionRows);
        } else {
          write_stata_column(pred.ystar.get(), pred.numPredictions, savePredictMode + 1, pred.predictionRows);
        }
      }

      if (pred.coeffs != nullptr) {
        write_stata_columns(pred.coeffs.get(), pred.numPredictions, pred.numCoeffCols,
                            savePredictMode + saveCoPredictMode + numCoeffColsSaved + 1, pred.predictionRows);
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
    ST_retcode rc = UNKNOWN_ERROR + 1;
    std::string command(argv[0]);

    if (command == "launch_edm_tasks") {
      rc = launch_edm_tasks(argc - 1, argv + 1);
    } else if (command == "report_progress") {
      if (!breakButtonPressed) {
        io.print(io.get_and_clear_async_buffer());
      }

      bool breakHit = (argc == 2) && atoi(argv[1]);
      if (breakHit) {
        breakButtonPressed = true;
        io.out("Aborting edm run (this may take a few seconds).\n");
      }

      if (allTasksFinished) {
        SF_scal_save(FINISHED_SCALAR, 1.0);
      }

      rc = SUCCESS;
    } else if (command == "collect_results") {
      io.print(io.get_and_clear_async_buffer());

      rc = save_all_task_results_to_stata(argc - 1, argv + 1);
    } else if (command == "launch_coprediction_task") {
      rc = prepare_coprediction_task(argc - 1, argv + 1);
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
