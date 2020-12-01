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

#include <optional>
#include <string>
#include <vector>

#include "manifold.h"

typedef int retcode;

struct Options
{
  bool forceCompute, saveMode, distributeThreads = false;
  int k, varssv, nthreads;
  double theta, missingdistance;
  std::string algorithm;
};

class IO
{
public:
  int verbosity = 0;

  virtual void print(std::string s) const
  {
    if (verbosity > 0) {
      out(s.c_str());
    }
  }

  virtual void print_async(std::string s) const
  {
    if (verbosity > 0) {
      out_async(s.c_str());
    }
  }

  virtual void progress_bar(double progress) const
  {
    if (progress == 0.0) {
      print_async("Percent complete: 0");
      nextMessage = 1.0 / 40;
      dots = 0;
      tens = 0;
      return;
    }

    while (progress >= nextMessage) {
      if (dots < 3) {
        print_async(".");
        dots += 1;
      } else {
        tens += 1;
        print_async(std::to_string(tens * 10));
        dots = 0;
      }
      nextMessage += 1.0 / 40;
    }
  }
  mutable int dots, tens;
  mutable double nextMessage;

  // Actual implementation of IO functions are in the subclasses
  virtual void out(const char*) const = 0;
  virtual void out_async(const char*) const = 0;
  virtual void error(const char*) const = 0;
  virtual void flush() const = 0;
};

struct Prediction
{
  retcode rc;
  std::vector<double> ystar;
  std::optional<std::vector<double>> flat_Bi_map;
};

DLL Prediction mf_smap_loop(Options opts, const std::vector<double>& y, const Manifold& M, const Manifold& Mp,
                            const IO& io, bool keep_going() = nullptr, void finished() = nullptr);
