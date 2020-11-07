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

typedef int retcode;

struct Observation
{
  const double* data;
  size_t E;

  double operator()(size_t j) const { return data[j]; }

  bool any_missing() const
  {
    bool missing = false;
    for (size_t i = 0; i < E; i++) {
      if (data[i] == MISSING) {
        missing = true;
        break;
      }
    }
    return missing;
  }
};

struct Manifold
{
  std::vector<double> flat;
  size_t _rows, _cols;

  double operator()(size_t i, size_t j) const { return flat[i * _cols + j]; }

  size_t rows() const { return _rows; }
  size_t cols() const { return _cols; }

  Observation get_observation(size_t i) const
  {
    Observation obs{ &(flat[i * _cols]), _cols };
    return obs;
  }
};

typedef struct
{
  bool force_compute, save_mode;
  int l, varssv;
  double theta, missingdistance;
  std::string algorithm;
} smap_opts_t;

typedef struct
{
  retcode rc;
  std::vector<double> ystar;
  std::optional<std::vector<double>> flat_Bi_map;
} smap_res_t;

DLL smap_res_t mf_smap_loop(smap_opts_t opts, const std::vector<double>& y, const Manifold& M, const Manifold& Mp,
                            int nthreads, void display(char*), void flush(), int verbosity);