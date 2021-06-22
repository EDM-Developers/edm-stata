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

void to_json(json& j, const Options& o)
{
  j = json{ { "copredict", o.copredict },
            { "forceCompute", o.forceCompute },
            { "savePrediction", o.savePrediction },
            { "saveSMAPCoeffs", o.saveSMAPCoeffs },
            { "distributeThreads", o.distributeThreads },
            { "k", o.k },
            { "nthreads", o.nthreads },
            { "missingdistance", o.missingdistance },
            { "thetas", o.thetas },
            { "algorithm", o.algorithm },
            { "taskNum", o.taskNum },
            { "numTasks", o.numTasks },
            { "calcRhoMAE", o.calcRhoMAE },
            { "parMode", o.parMode } };
}

void from_json(const json& j, Options& o)
{
  j.at("copredict").get_to(o.copredict);
  j.at("forceCompute").get_to(o.forceCompute);
  j.at("savePrediction").get_to(o.savePrediction);
  j.at("saveSMAPCoeffs").get_to(o.saveSMAPCoeffs);
  j.at("distributeThreads").get_to(o.distributeThreads);
  j.at("k").get_to(o.k);
  j.at("nthreads").get_to(o.nthreads);
  j.at("missingdistance").get_to(o.missingdistance);
  j.at("thetas").get_to(o.thetas);
  j.at("algorithm").get_to(o.algorithm);
  j.at("taskNum").get_to(o.taskNum);
  j.at("numTasks").get_to(o.numTasks);
  j.at("calcRhoMAE").get_to(o.calcRhoMAE);
  j.at("parMode").get_to(o.parMode);
}

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

void to_json(json& j, const PredictionStats& s)
{
  j = json{ { "mae", s.mae }, { "rho", s.rho }, { "taskNum", s.taskNum }, { "calcRhoMAE", s.calcRhoMAE } };
}

void from_json(const json& j, PredictionStats& s)
{
  j.at("mae").get_to(s.mae);
  j.at("rho").get_to(s.rho);
  j.at("taskNum").get_to(s.taskNum);
  j.at("calcRhoMAE").get_to(s.calcRhoMAE);
}

void to_json(json& j, const Prediction& p)
{
  std::vector<double> yStarVec, coeffsVec;
  if (p.ystar != nullptr) {
    yStarVec = std::vector<double>(p.ystar.get(), p.ystar.get() + p.numThetas * p.numPredictions);
  }
  if (p.coeffs != nullptr) {
    coeffsVec = std::vector<double>(p.coeffs.get(), p.coeffs.get() + p.numPredictions * p.numCoeffCols);
  }

  j = json{ { "rc", p.rc },
            { "numThetas", p.numThetas },
            { "numPredictions", p.numPredictions },
            { "numCoeffCols", p.numCoeffCols },
            { "ystar", yStarVec },
            { "coeffs", coeffsVec },
            { "stats", p.stats },
            { "predictionRows", p.predictionRows } };
}

// Eventually, we'll need a function like the following.
// Currently it won't compile as the unique_ptrs aren't cooperating.
void from_json(const json& j, Prediction& p)
{
  j.at("rc").get_to(p.rc);
  j.at("numThetas").get_to(p.numThetas);
  j.at("numPredictions").get_to(p.numPredictions);
  j.at("numCoeffCols").get_to(p.numCoeffCols);
  j.at("predictionRows").get_to(p.predictionRows);
  j.at("stats").get_to(p.stats);
  // j.at("ystar").get_to(p.ystar);
  // j.at("coeffs").get_to(p.coeffs);
}

/*! \brief Read in a dump file.
 *
 * Read in a dump file created with the Stata `saveinputs' option.
 *
 * \param fname dump filename
 * \param pointer to InputVars struct to store the read
 */
Inputs read_dumpfile(std::string fname_in)
{
  std::ifstream i(fname_in);
  json j;
  i >> j;

  int E = j["E"];
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