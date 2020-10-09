#ifndef EDM_H
#define EDM_H

#ifdef _MSC_VER
#define DLL extern __declspec(dllexport)
#else
#define DLL
#endif

#define SUCCESS 0
#define INVALID_ALGORITHM 400
#define INSUFFICIENT_UNIQUE 503
#define NOT_IMPLEMENTED 908
#define MALLOC_ERROR 909

/* global variable placeholder for missing values */
#define MISSING 1.0e+100

#include <stdbool.h>

typedef int retcode;

DLL retcode mf_smap_loop(int count_predict_set, int count_train_set, int mani, int Mpcol, double* flat_M,
                         double* flat_Mp, double* y, int l, double theta, double* S, char* algorithm, bool save_mode,
                         int varssv, bool force_compute, double missingdistance, double* ystar, double* flat_Bi_map);

#endif
