#pragma once

#ifdef _MSC_VER
#define DLL extern __declspec(dllexport)
#else
#define DLL
#endif

#define SUCCESS 0
#define TOO_FEW_VARIABLES 102
#define TOO_MANY_VARIABLES 103
#define INVALID_ALGORITHM 400
#define INSUFFICIENT_UNIQUE 503
#define NOT_IMPLEMENTED 908
#define UNKNOWN_ERROR 8000

/* global variable placeholder for missing values */
#define MISSING 1.0e+100

typedef int retcode;

#include <future>
#include <memory> // For unique_ptr
#include <string>
#include <vector>

#include <experimental/mdspan>

namespace stdex = std::experimental;

using span_2d_double = stdex::mdspan<double, stdex::dynamic_extent, stdex::dynamic_extent>;
using span_2d_retcode = stdex::mdspan<retcode, stdex::dynamic_extent, stdex::dynamic_extent>;

#include "manifold.h"

struct Options
{
  bool copredict, forceCompute, savePrediction, saveSMAPCoeffs;
  bool distributeThreads = false;
  int k, nthreads;
  double missingdistance;
  std::vector<double> thetas;
  std::string algorithm;
  size_t taskNum = 1, numTasks = 1;
  bool calcRhoMAE = false;
};

class IO
{
public:
  int verbosity = 0;

  virtual void print(std::string s) const
  {
    if (verbosity > 0) {
      std::lock_guard<std::mutex> guard(printMutex);
      out(s.c_str());
      flush();
    }
  }

  virtual void progress_bar(double progress) const
  {
    if (verbosity < 1) {
      return;
    }

    if (progress == 0.0) {
      print("Percent complete: 0");
      nextMessage = 1.0 / 40;
      dots = 0;
      tens = 0;
      return;
    }

    while (progress >= nextMessage) {
      if (dots < 3) {
        print(".");
        dots += 1;
      } else {
        tens += 1;
        print(std::to_string(tens * 10));
        dots = 0;
      }
      nextMessage += 1.0 / 40;
    }

    if (progress >= 1.0) {
      print("\n");
    }
  }
  mutable int dots, tens;
  mutable double nextMessage;
  mutable std::mutex printMutex;

  // Actual implementation of IO functions are in the subclasses
  virtual void out(const char*) const = 0;
  virtual void error(const char*) const = 0;
  virtual void flush() const = 0;
};

struct PredictionStats
{
  double mae, rho;
  int taskNum;
  bool calcRhoMAE;
};

struct Prediction
{
  retcode rc;
  size_t numThetas, numPredictions, numCoeffCols;
  std::unique_ptr<double[]> ystar;
  std::unique_ptr<double[]> coeffs;
  PredictionStats stats;
};

std::future<void> edm_async(Options opts, const ManifoldGenerator* generator, size_t E, std::vector<bool> trainingRows,
                            std::vector<bool> predictionRows, IO* io, Prediction* pred, bool keep_going() = nullptr,
                            void all_tasks_finished(void) = nullptr);

void edm_task(Options opts, const ManifoldGenerator* generator, size_t E, std::vector<bool> trainingRows,
              std::vector<bool> predictionRows, IO* io, Prediction* pred, bool keep_going() = nullptr,
              void all_tasks_finished(void) = nullptr, bool serial = false);