#include "cli.h"
#include <iostream>
#include <queue>

std::atomic<bool> going = true;

bool keep_going()
{
  return going;
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
  std::queue<Prediction> predictions;
  std::queue<std::future<void>> futures;
  std::queue<Inputs> inputs;

  std::cerr << "Read in the JSON input from " << fnameIn << "\n";
  std::ifstream i(fnameIn);
  json j;
  i >> j;

  bool verb = false;

  int numTaskGroups = j.size();
  std::cerr << "Number of task groups is " << numTaskGroups << "\n";
  for (int taskGroupNum = 0; taskGroupNum < numTaskGroups; taskGroupNum++) {
    if (verb) {
      std::cout << "Starting task group number " << taskGroupNum << "\n";
    }

    json taskGroup = j[taskGroupNum];

    ManifoldGenerator generator = taskGroup["generator"];
    Options opts = taskGroup["opts"];

    std::vector<int> Es = taskGroup["Es"];
    std::vector<int> libraries = taskGroup["libraries"];
    int k = taskGroup["k"];
    int numReps = taskGroup["numReps"];
    int crossfold = taskGroup["crossfold"];
    bool explore = taskGroup["explore"];
    bool full = taskGroup["full"];
    bool saveFinalPredictions = taskGroup["saveFinalPredictions"];
    bool saveSMAPCoeffs = taskGroup["saveSMAPCoeffs"];
    std::vector<bool> usable = taskGroup["usable"];
    std::string rngState = taskGroup["rngState"];
    double nextRV = taskGroup["nextRV"];

//    launch_task_group(generator, opts, Es, libraries, k, numReps, crossfold, explore, full, saveFinalPredictions,
//                      saveSMAPCoeffs, usable, rngState, nextRV);

    // Collect the results of this task group before moving on to the next task group
    int numTasks = futures.size();
    for (int taskNum = 0; taskNum < numTasks; taskNum++) {
      if (verb) {
        std::cout << "Getting results of task number " << taskNum << "\n";
      }
      futures.front().get();
      futures.pop();
    }
  }

  size_t ext = fnameIn.find_last_of(".");
  fnameIn = fnameIn.substr(0, ext);
  std::string fnameOut = fnameIn + "-out.json";

  remove(fnameOut.c_str());

  int rc = 0;
  json results;

  std::cerr << "Saving " << predictions.size() << " predictions\n";
  while (!predictions.empty()) {
    const Prediction& pred = predictions.front();

    results.push_back(pred);
    if (pred.rc > rc) {
      rc = pred.rc;
    }

    predictions.pop();
  }

  std::ofstream o(fnameOut);
  o << std::setw(4) << results << std::endl;

  return rc;
}
