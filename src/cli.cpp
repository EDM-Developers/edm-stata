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

  bool verb = true;

  std::cerr << "Input is of size " << j.size() << "\n";
  for (int taskNum = 0; taskNum < j.size(); taskNum++) {
    if (verb) {
      std::cout << "Starting task number " << taskNum;
    }

    inputs.emplace(parse_dumpfile(j[taskNum]));
    Inputs* vars = &(inputs.back());

    if (verb) {
      std::cerr << " (" << vars->opts.taskNum + 1 << " of " << vars->opts.numTasks << ")\n";
    }

    vars->opts.nthreads = nthreads;

    predictions.push({});

    futures.emplace(edm_task_async(vars->opts, &(vars->generator), vars->E, vars->trainingRows, vars->predictionRows,
                                   &io, &(predictions.back()), keep_going));
  }

  for (int taskNum = 0; taskNum < j.size(); taskNum++) {
    if (verb) {
      std::cout << "Getting results of task number " << taskNum << "\n";
    }
    futures.front().get();
    futures.pop();
  }

  size_t ext = fnameIn.find_last_of(".");
  fnameIn = fnameIn.substr(0, ext);
  std::string fnameOut = fnameIn + "-out.json";

  remove(fnameOut.c_str());

  int rc = 0;
  json results;

  for (int taskNum = 0; taskNum < j.size(); taskNum++) {
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
