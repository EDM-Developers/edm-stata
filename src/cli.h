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
Inputs parse_dumpfile(const json& j)
{
  int E = j["E"];
  Options opts = j["opts"];
  ManifoldGenerator generator = j["generator"];

  std::vector<bool> trainingRows = j["trainingRows"], predictionRows = j["predictionRows"];

  return { opts, generator, E, trainingRows, predictionRows };
}

Inputs read_dumpfile(std::string fName)
{
  std::ifstream i(fName);
  json j;
  i >> j;

  return parse_dumpfile(j[0]);
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