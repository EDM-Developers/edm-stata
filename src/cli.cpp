#include "cli.h"
#include <iostream>
#include <queue>

std::atomic<bool> going = true;

bool keep_going()
{
  return going;
}

std::vector<bool> int_to_bool(std::vector<int> iv)
{
  std::vector<bool> bv;
  std::copy(iv.begin(), iv.end(), std::back_inserter(bv));
  return bv;
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cerr << "Usage: ./edm_cli filename [numThreads=1]" << std::endl;
    return -1;
  }

  std::string fnameIn(argv[1]);

  int nthreads = 4;
  if (argc > 2) {
    nthreads = atoi(argv[2]);
  }

  std::cerr << "Using nthreads = " << nthreads << "\n";

  ConsoleIO io;

  std::cerr << "Read in the JSON input from " << fnameIn << "\n";
  std::ifstream i(fnameIn);
  json j;
  i >> j;

  size_t ext = fnameIn.find_last_of(".");
  fnameIn = fnameIn.substr(0, ext);
  std::string fnameOut = fnameIn + "-out.json";

  remove(fnameOut.c_str());

  int rc = 0;
  json results;

  bool verb = true;

  int numTaskGroups = j.size();
  std::cerr << "Number of task groups is " << numTaskGroups << "\n";
  for (int taskGroupNum = 0; taskGroupNum < numTaskGroups; taskGroupNum++) {
    if (verb) {
      std::cerr << "Starting task group number " << taskGroupNum << "\n";
    }

    json taskGroup = j[taskGroupNum];

    ManifoldGenerator generator = taskGroup["generator"];
    Options opts = taskGroup["opts"];

    std::cerr << "Loading: " << opts.cmdLine << "\n";

    if (opts.copredict) {
      std::cerr << "Actually, skipping the coprediction for not (not yet implemented)\n";
      continue;
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
    std::vector<double> co_x = taskGroup["co_x"];
    std::vector<bool> coTrainingRows = int_to_bool(taskGroup["coTrainingRows"]);
    std::vector<bool> coPredictionRows = int_to_bool(taskGroup["coPredictionRows"]);
    std::string rngState = taskGroup["rngState"];
    double nextRV = taskGroup["nextRV"];

    std::cerr << "Loaded this part of the JSON\n";

    std::vector<std::future<Prediction>> futures = launch_task_group(
      generator, opts, Es, libraries, k, numReps, crossfold, explore, full, saveFinalPredictions, saveSMAPCoeffs,
      copredictMode, usable, co_x, coTrainingRows, coPredictionRows, rngState, nextRV, &io, nullptr, nullptr);

    // Collect the results of this task group before moving on to the next task group
    if (verb) {
      std::cerr << "Waiting for results...\n";
    }

    if (verb) {
      std::cerr << "Storing results...\n";
    }

    for (int i = 0; i < futures.size(); i++) {
      const Prediction pred = futures[i].get();

      results.push_back(pred);
      if (pred.rc > rc) {
        rc = pred.rc;
      }
    }
  }

  std::ofstream o(fnameOut);
  o << std::setw(4) << results << std::endl;

  return rc;
}
