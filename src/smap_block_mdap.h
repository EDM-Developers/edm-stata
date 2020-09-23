#ifndef SMAP_BLOCK_MDAP_H
#define SMAP_BLOCK_MDAP_H

#ifdef _MSC_VER
#define DLL extern __declspec(dllexport)
#else
#define DLL
#endif

#include "stplugin.h"

DLL ST_double** alloc_matrix(ST_int nrow, ST_int ncol);
DLL void free_matrix(ST_double** M, ST_int nrow);

DLL ST_retcode mf_smap_loop(ST_int count_predict_set, ST_int count_train_set, ST_double** Bi_map, ST_int mani,
                            ST_double** M, ST_double** Mp, ST_double* y, ST_int l, ST_double theta, ST_double* S,
                            char* algorithm, ST_int save_mode, ST_int varssv, ST_int force_compute,
                            ST_double missingdistance, ST_double* ystar);

#endif
