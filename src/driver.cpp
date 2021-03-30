#include "driver.h"

std::atomic<bool> going = true;

bool keep_going()
{
  return going;
}

int main(int argc, char* argv[])
{
  if (argc < 2) {
    std::cerr << "Usage: ./driver <fname>" << std::endl;
    return -1;
  }

  std::string fnameIn(argv[1]);

  Inputs vars = read_dumpfile(fnameIn);

  if (argc > 2) {
    vars.opts.nthreads = atoi(argv[2]);
  } else {
    vars.opts.nthreads = 4;
  }
  vars.opts.numTasks = 1;
  vars.opts.taskNum = 1;

  ConsoleIO io;
  Prediction pred;

  std::future<void> fut =
    edm_async(vars.opts, &vars.generator, vars.E, vars.trainingRows, vars.predictionRows, &io, &pred, keep_going);
  fut.get();

  std::size_t ext = fnameIn.find_last_of(".");
  fnameIn = fnameIn.substr(0, ext);
  std::string fnameOut = fnameIn + "-out.json";

  write_results(fnameOut, pred);

  return pred.rc;
}