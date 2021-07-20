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
    ManifoldGenerator generator = j[taskGroupNum]["generator"];
    std::vector<bool> trainingRows, predictionRows;

    int numTasks = j[taskGroupNum]["tasks"].size();
    std::cerr << "Number of tasks is " << numTasks << "\n";

    for (int taskNum = 0; taskNum < numTasks; taskNum++) {
      Options opts = j[taskGroupNum]["tasks"][taskNum]["opts"];
      int E = j[taskGroupNum]["tasks"][taskNum]["E"];

      if (j[taskGroupNum]["tasks"][taskNum].contains("trainingRows")) {
        // Saving the vector of bools as a vector of ints (as it's much smaller in JSON format),
        // so have to convert it back.
        trainingRows.clear();
        predictionRows.clear();

        std::vector<int> trainingRowsInt = j[taskGroupNum]["tasks"][taskNum]["trainingRows"];
        std::vector<int> predictionRowsInt = j[taskGroupNum]["tasks"][taskNum]["predictionRows"];

        std::copy(trainingRowsInt.begin(), trainingRowsInt.end(), std::back_inserter(trainingRows));

        std::copy(predictionRowsInt.begin(), predictionRowsInt.end(), std::back_inserter(predictionRows));
      }

      opts.nthreads = nthreads;

      predictions.push({});

      futures.emplace(
        edm_task_async(opts, &generator, E, trainingRows, predictionRows, &io, &(predictions.back()), keep_going));
    }

    // Collect the results of this task group before moving on to the next task group
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
