#include "edm.h"
#include <fstream>
#include <iomanip>
#include <iostream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

class ConsoleIO : public IO
{
public:
  ConsoleIO() { this->verbosity = std::numeric_limits<int>::max(); }
  ConsoleIO(int v) { this->verbosity = v; }
  virtual void out(const char* s) const { std::cout << s; }
  virtual void error(const char* s) const { std::cerr << s; }
  virtual void flush() const { fflush(stdout); }
};

struct Inputs
{
  Options opts;
  ManifoldGenerator generator;
  int E;
  std::vector<bool> trainingRows, predictionRows;
};

/*! \brief Read in a dump file.
 *
 * Read in a dump file created with the Stata `saveinputs' option.
 *
 * \param fname dump filename
 * \param pointer to InputVars struct to store the read
 */
Inputs parse_lowlevel_inputs_file(const json& j)
{
  int E = j["E"];
  Options opts = j["opts"];
  ManifoldGenerator generator = j["generator"];
  std::vector<bool> trainingRows = j["trainingRows"], predictionRows = j["predictionRows"];

  return { opts, generator, E, trainingRows, predictionRows };
}

Inputs read_lowlevel_inputs_file(std::string fName)
{
  std::ifstream i(fName);
  json j;
  i >> j;

  return parse_lowlevel_inputs_file(j);
}

void append_to_dumpfile(std::string fName, const json& taskGroup)
{
  json allTaskGroups;

  std::ifstream i(fName);
  if (i.is_open()) {
    i >> allTaskGroups;
  }

  allTaskGroups.push_back(taskGroup);

  // Add "o << std::setw(4) << allTaskGroups" to pretty-print the saved JSON
  std::ofstream o(fName);
  o << allTaskGroups << std::endl;
}

std::vector<bool> int_to_bool(std::vector<int> iv)
{
  std::vector<bool> bv;
  std::copy(iv.begin(), iv.end(), std::back_inserter(bv));
  return bv;
}

json run_tests(json testInputs, int nthreads, IO* io, bool verb)
{
  int rc = 0;
  json results;

  int numTaskGroups = testInputs.size();

  if (verb) {
    std::cout << "Number of task groups is " << numTaskGroups << "\n";
  }

  for (int taskGroupNum = 0; taskGroupNum < numTaskGroups; taskGroupNum++) {
    if (verb) {
      std::cout << "Starting task group number " << taskGroupNum << "\n";
    }

    json taskGroup = testInputs[taskGroupNum];

    Options opts = taskGroup["opts"];
    opts.nthreads = nthreads;

    if (verb) {
      std::cout << "Loading: " << opts.cmdLine << "\n";
    }

    ManifoldGenerator generator = taskGroup["generator"];

    if (verb) {
      std::cout << "Generator loaded!" << std::endl;
    }

    std::vector<int> Es = taskGroup["Es"];
    std::vector<int> libraries = taskGroup["libraries"];
    int k = taskGroup["k"];
    int numReps = taskGroup["numReps"];
    int crossfold = taskGroup["crossfold"];
    bool explore = taskGroup["explore"];
    bool full = taskGroup["full"];
    bool saveFinalPredictions = taskGroup["saveFinalPredictions"];
    bool saveSMAPCoeffs = taskGroup["saveSMAPCoeffs"];
    bool copredictMode = taskGroup["copredictMode"];
    std::vector<bool> usable = int_to_bool(taskGroup["usable"]);
    std::vector<bool> coTrainingRows = int_to_bool(taskGroup["coTrainingRows"]);
    std::vector<bool> coPredictionRows = int_to_bool(taskGroup["coPredictionRows"]);
    std::string rngState = taskGroup["rngState"];
    double nextRV = taskGroup["nextRV"];

    std::vector<std::future<Prediction>> futures = launch_task_group(
      generator, opts, Es, libraries, k, numReps, crossfold, explore, full, saveFinalPredictions, saveSMAPCoeffs,
      copredictMode, usable, coTrainingRows, coPredictionRows, rngState, nextRV, io, nullptr, nullptr);

    // Collect the results of this task group before moving on to the next task group
    if (verb) {
      std::cout << "Waiting for results...\n";
    }

    for (int f = 0; f < futures.size(); f++) {
      const Prediction pred = futures[f].get();

      results.push_back(pred);
      if (pred.rc > rc) {
        rc = pred.rc;
      }
    }
  }

  if (verb) {
    std::cout << "rc is " << rc;
  }

  return results;
}
