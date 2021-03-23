#include "driver.h"

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
    vars.opts.nthreads = 1;
  }
  vars.opts.numTasks = 1;
  vars.opts.taskNum = 1;

  ConsoleIO io;
  Prediction pred;

  std::future<void> fut =
    edm_async(vars.opts, &vars.generator, vars.E, vars.trainingRows, vars.predictionRows, &io, &pred);
  fut.get();

  std::size_t ext = fnameIn.find_last_of(".");
  fnameIn = fnameIn.substr(0, ext);
  std::string fnameOut = fnameIn + "-out.h5";

  write_results(fnameOut, pred);

  return pred.rc;
}