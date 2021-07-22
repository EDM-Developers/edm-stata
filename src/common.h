#pragma once

#define SUCCESS 0
#define TOO_FEW_VARIABLES 102
#define TOO_MANY_VARIABLES 103
#define INVALID_ALGORITHM 400
#define INVALID_DISTANCE 401
#define INVALID_METRICS 402
#define INSUFFICIENT_UNIQUE 503
#define NOT_IMPLEMENTED 908
#define CANNOT_SAVE_RESULTS 1000
#define UNKNOWN_ERROR 8000

/* global variable placeholder for missing values */
#define MISSING 1.0e+100

typedef int retcode;

#include <future>
#include <memory> // For unique_ptr
#include <queue>
#include <string>
#include <vector>

#include "manifold.h"

enum class Algorithm
{
  Simplex,
  SMap,
};

enum class Distance
{
  MeanAbsoluteError,
  Euclidean,
  Wasserstein
};

enum class Metric
{
  Diff,
  CheckSame
};

// Store these enum classes to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM(Algorithm, { { Algorithm::Simplex, "Simplex" }, { Algorithm::SMap, "SMap" } })

NLOHMANN_JSON_SERIALIZE_ENUM(Distance, { { Distance::MeanAbsoluteError, "MeanAbsoluteError" },
                                         { Distance::Euclidean, "Euclidean" },
                                         { Distance::Wasserstein, "Wasserstein" } })

NLOHMANN_JSON_SERIALIZE_ENUM(Metric, {
                                       { Metric::Diff, "Diff" },
                                       { Metric::CheckSame, "CheckSame" },
                                     })

struct Options
{
  bool copredict, forceCompute, savePrediction, saveSMAPCoeffs;
  bool distributeThreads = false;
  int k, nthreads;
  double missingdistance;
  std::vector<double> thetas;
  Algorithm algorithm;
  size_t taskNum = 1, numTasks = 1;
  bool calcRhoMAE = false;
  double aspectRatio;
  Distance distance;
  std::vector<Metric> metrics;
  std::string cmdLine;
  bool saveKUsed = false;
};

void to_json(json& j, const Options& o);
void from_json(const json& j, Options& o);

struct PredictionStats
{
  double mae, rho;
  int taskNum;
  bool calcRhoMAE;
};

void to_json(json& j, const PredictionStats& s);
void from_json(const json& j, PredictionStats& s);

struct Prediction
{
  retcode rc;
  size_t numThetas, numPredictions, numCoeffCols;
  std::unique_ptr<double[]> ystar;
  std::unique_ptr<double[]> coeffs;
  std::vector<PredictionStats> stats;
  std::vector<bool> predictionRows;
  std::vector<int> kUsed;
  std::string cmdLine;
  bool copredict;
};

void to_json(json& j, const Prediction& p);
void from_json(const json& j, Prediction& p);

class IO
{
public:
  int verbosity = 0;

  virtual void print(std::string s)
  {
    if (verbosity > 0) {
      out(s.c_str());
      flush();
    }
  }

  virtual void print_async(std::string s)
  {
    if (verbosity > 0) {
      std::lock_guard<std::mutex> guard(bufferMutex);
      buffer += s;
    }
  }

  virtual std::string get_and_clear_async_buffer()
  {
    std::lock_guard<std::mutex> guard(bufferMutex);
    std::string ret = buffer;
    buffer.clear();
    return ret;
  }

  virtual void progress_bar(double progress)
  {
    std::lock_guard<std::mutex> guard(bufferMutex);

    if (progress == 0.0) {
      buffer += "Percent complete: 0";
      nextMessage = 1.0 / 40;
      dots = 0;
      tens = 0;
      return;
    }

    while (progress >= nextMessage && nextMessage < 1.0) {
      if (dots < 3) {
        buffer += ".";
        dots += 1;
      } else {
        tens += 1;
        buffer += std::to_string(tens * 10);
        dots = 0;
      }
      nextMessage += 1.0 / 40;
    }

    if (progress >= 1.0) {
      buffer += "\n";
    }
  }

  // Actual implementation of IO functions are in the subclasses
  virtual void out(const char*) const = 0;
  virtual void error(const char*) const = 0;
  virtual void flush() const = 0;

private:
  std::string buffer = "";
  std::mutex bufferMutex;

  int dots, tens;
  double nextMessage;
};
