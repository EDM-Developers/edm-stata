#ifndef SMAP_BLOCK_MDAP_H
#define SMAP_BLOCK_MDAP_H

#include "stplugin.h"

ST_retcode mf_smap_loop(ST_int count_predict_set, ST_int count_train_set, ST_double** Bi_map, ST_int mani,
                        ST_double** M, ST_double** Mp, ST_double* y, ST_int l, ST_double theta, ST_double* S,
                        char* algorithm, ST_int save_mode, ST_int varssv, ST_int force_compute, ST_int missingdistance,
                        ST_double* ystar);

#endif
