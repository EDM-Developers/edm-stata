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
  int E;
  std::vector<bool> trainingRows, predictionRows;
};

void to_json(json& j, const ManifoldGenerator& g)
{
  j = json{ { "_copredict", g._copredict },
            { "_use_dt", g._use_dt },
            { "_add_dt0", g._add_dt0 },
            { "_tau", g._tau },
            { "_missing", g._missing },
            { "_num_extras", g._num_extras },
            { "_num_extras_varying", g._num_extras_varying },
            { "_dtWeight", g._dtWeight },
            { "_x", g._x },
            { "_y", g._y },
            { "_co_x", g._co_x },
            { "_t", g._t },
            { "_extras", g._extras },
            { "_extrasEVarying", g._extrasEVarying } };
}

void from_json(const json& j, ManifoldGenerator& g)
{
  j.at("_copredict").get_to(g._copredict);
  j.at("_use_dt").get_to(g._use_dt);
  j.at("_add_dt0").get_to(g._add_dt0);
  j.at("_tau").get_to(g._tau);
  j.at("_missing").get_to(g._missing);
  j.at("_num_extras").get_to(g._num_extras);
  j.at("_num_extras_varying").get_to(g._num_extras_varying);
  j.at("_dtWeight").get_to(g._dtWeight);
  j.at("_x").get_to(g._x);
  j.at("_y").get_to(g._y);
  j.at("_co_x").get_to(g._co_x);
  j.at("_t").get_to(g._t);
  j.at("_extras").get_to(g._extras);
  j.at("_extrasEVarying").get_to(g._extrasEVarying);
}

/*! \brief Read in a dump file.
 *
 * Read in a dump file created with the Stata `saveinputs' option.
 *
 * \param fname dump filename
 * \param pointer to InputVars struct to store the read
 */
Inputs parse_dumpfile(json j)
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

void write_dumpfile(const char* fname, const Options& opts, const ManifoldGenerator& generator, int E,
                    const std::vector<bool>& trainingRows, const std::vector<bool>& predictionRows)
{
  json task;
  task["opts"] = opts;
  task["generator"] = generator;
  task["E"] = E;
  task["trainingRows"] = trainingRows;
  task["predictionRows"] = predictionRows;

  json tasks;

  std::ifstream i(fname);
  if (i.is_open()) {
    i >> tasks;
  }

  tasks.push_back(task);

  std::ofstream o(fname);
  o << std::setw(4) << tasks << std::endl;
}

void write_results(std::string fname_out, const Prediction& pred)
{
  json results;

  std::ifstream i(fname_out);
  if (i.is_open()) {
    i >> results;
  }

  json result = pred;
  results.push_back(result);

  std::ofstream o(fname_out);
  o << std::setw(4) << results << std::endl;
}