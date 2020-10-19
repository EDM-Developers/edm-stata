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

#include <stdbool.h>

#include <optional>
#include <string>
#include <vector>

typedef int retcode;

typedef struct
{
  std::vector<double> flat;
  int rows, cols;
} manifold_t;

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

DLL smap_res_t mf_smap_loop(smap_opts_t opts, const std::vector<double>& y, const manifold_t& M, const manifold_t& Mp);