#include "cli.h"
#include <iostream>

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

  std::vector<Prediction*> predictions;
  std::vector<std::future<void>> futures;
  std::vector<ManifoldGenerator*> generators;

  std::cerr << "Read in the JSON input from " << fnameIn << "\n";
  std::ifstream i(fnameIn);
  json j;
  i >> j;

  bool verb = false;

  std::cerr << "Input is of size " << j.size() << "\n";
  for (int taskNum = 0; taskNum < j.size(); taskNum++) {
    if (verb)
      std::cout << "Starting task number " << taskNum;
    Inputs vars = parse_dumpfile(j[taskNum]);

    if (verb)
      std::cerr << " (" << vars.opts.taskNum << " of " << vars.opts.numTasks << ")\n";

    vars.opts.nthreads = nthreads;

    Prediction* pred = new Prediction;
    predictions.push_back(pred);

    ManifoldGenerator* gen = new ManifoldGenerator(vars.generator);
    generators.push_back(gen);

    futures.emplace_back(
      edm_task_async(vars.opts, gen, vars.E, vars.trainingRows, vars.predictionRows, &io, pred, keep_going));
  }

  for (int taskNum = 0; taskNum < j.size(); taskNum++) {
    if (verb)
      std::cout << "Getting results of task number " << taskNum << "\n";
    futures[taskNum].get();
  }

  size_t ext = fnameIn.find_last_of(".");
  fnameIn = fnameIn.substr(0, ext);
  std::string fnameOut = fnameIn + "-out.json";

  remove(fnameOut.c_str());

  int rc = 0;
  json results;

  for (int taskNum = 0; taskNum < j.size(); taskNum++) {
    Prediction* pred = predictions[taskNum];
    results.push_back(*pred);
    if (pred->rc > rc) {
      rc = pred->rc;
    }

    delete pred;
    delete generators[taskNum];
  }

  std::ofstream o(fnameOut);
  o << std::setw(4) << results << std::endl;

  return rc;
}
