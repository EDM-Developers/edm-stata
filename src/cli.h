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

void to_json(json& j, const ManifoldGenerator& g)
{
  j = json{ { "_use_dt", g._use_dt },
            { "_add_dt0", g._add_dt0 },
            { "_tau", g._tau },
            { "_missing", g._missing },
            { "_num_extras", g._num_extras },
            { "_num_extras_lagged", g._num_extras_lagged },
            { "_dtWeight", g._dtWeight },
            { "_x", g._x },
            { "_y", g._y },
            { "_co_x", g._co_x },
            { "_t", g._t },
            { "_extras", g._extras } };
}

void from_json(const json& j, ManifoldGenerator& g)
{
  j.at("_use_dt").get_to(g._use_dt);
  j.at("_add_dt0").get_to(g._add_dt0);
  j.at("_tau").get_to(g._tau);
  j.at("_missing").get_to(g._missing);
  j.at("_num_extras").get_to(g._num_extras);
  j.at("_num_extras_lagged").get_to(g._num_extras_lagged);
  j.at("_dtWeight").get_to(g._dtWeight);
  j.at("_x").get_to(g._x);
  j.at("_y").get_to(g._y);
  j.at("_co_x").get_to(g._co_x);
  j.at("_t").get_to(g._t);
  j.at("_extras").get_to(g._extras);
}

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