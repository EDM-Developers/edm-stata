#define DRIVER_MODE 1

#include "common.h"
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
  virtual void out_async(const char* s) const { out(s); }
  virtual void error(const char* s) const { std::cerr << s; }
  virtual void flush() const { fflush(stdout); }
};

struct Inputs
{
  Options opts;
  ManifoldGenerator generator;
  size_t E;
  std::vector<bool> trainingRows, predictionRows;
};

/*! \brief Read in a dump file.
 *
 * Read in a dump file created with compile flag DUMP_INPUT.
 *
 * \param fname dump filename
 * \param pointer to InputVars struct to store the read
 */
Inputs read_dumpfile(std::string fname_in)
{
  std::ifstream i(fname_in);
  json j;
  i >> j;

  size_t E = j["E"];
  Options opts = j["opts"];
  ManifoldGenerator generator = j["generator"];

  std::vector<bool> trainingRows = j["trainingRows"], predictionRows = j["predictionRows"];

  return { opts, generator, E, trainingRows, predictionRows };
}

void write_dumpfile(const char* fname, const Options& opts, const ManifoldGenerator& generator, int E,
                    const std::vector<bool>& trainingRows, const std::vector<bool>& predictionRows)
{
  json j;
  j["opts"] = opts;
  j["generator"] = generator;
  j["E"] = E;
  j["trainingRows"] = trainingRows;
  j["predictionRows"] = predictionRows;

  std::ofstream o(fname);
  o << std::setw(4) << j << std::endl;
}

void write_results(std::string fname_out, const Prediction& pred)
{
  json j = pred;
  std::ofstream o(fname_out);
  o << std::setw(4) << j << std::endl;
}