#ifndef SMAP_BLOCK_MDAP_H
#define SMAP_BLOCK_MDAP_H

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

#include "stplugin.h"
#include <stdbool.h>

DLL ST_retcode mf_smap_loop(ST_int count_predict_set, ST_int count_train_set, ST_int mani, ST_int Mpcol,
                            ST_double* flat_M, ST_double* flat_Mp, ST_double* y, ST_int l, ST_double theta,
                            ST_double* S, char* algorithm, bool save_mode, ST_int varssv, bool force_compute,
                            ST_double missingdistance, ST_double* ystar, ST_double* flat_Bi_map);

#endif
