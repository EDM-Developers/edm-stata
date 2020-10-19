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
#define MALLOC_ERROR 909
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
  retcode rc;
  std::vector<double> ystar;
  std::optional<std::vector<double>> flat_Bi_map;
} smap_res_t;

DLL smap_res_t mf_smap_loop(int count_predict_set, int count_train_set, int mani, int Mpcol, int l, double theta,
                            std::string algorithm, bool save_mode, int varssv, bool force_compute,
                            double missingdistance, const std::vector<double>& y, const std::vector<double>& S,
                            const std::vector<double>& flat_M, const std::vector<double>& flat_Mp);