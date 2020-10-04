/* Suppress Windows problems with sprintf etc. functions. */
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include "edm.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DUMP_INPUT
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

ST_retcode print_error(ST_retcode rc)
{
  char temps[500];

  switch (rc) {
    case MALLOC_ERROR:
      sprintf(temps, "Insufficient memory\n");
      break;
    case NOT_IMPLEMENTED:
      sprintf(temps, "Method is not yet implemented\n");
      break;
    case INSUFFICIENT_UNIQUE:
      sprintf(temps, "Insufficient number of unique observations, consider "
                     "tweaking the values of E, k or use -force- option\n");
      break;
    case INVALID_ALGORITHM:
      sprintf(temps, "Invalid algorithm argument\n");
      break;
  }

  if (rc != SUCCESS) {
    SF_error(temps);
  }

  return rc;
}

/*
Example call to the plugin:

local myvars ``manifold'' `co_mapping' `x_f' `x_p' `train_set' `predict_set' `overlap' `vars_save'

unab vars : ``manifold''
local mani `: word count `vars''

local pmani_flag = 0

local vsave_flag = 0

plugin call smap_block_mdap `myvars', `j' `lib_size' "`algorithm'" "`force'" `missingdistance' `mani' `pmani_flag'
`vsave_flag'
*/

DLL ST_retcode stata_call(int argc, char* argv[])
{
  ST_int nvars, nobs, first, last, mani, pmani_flag, pmani, smaploc;
  ST_int Mpcol, l, vsave_flag, save_mode, varssv;

  ST_double value, theta, missingdistance, *train_use, *predict_use, *skip_obs;
  gsl_matrix *M, *Mp, *Bi_map;
  ST_double *y, *S, *ystar;
  ST_int i, j, h, force_compute, nthreads;
  ST_int count_train_set, count_predict_set, obsi, obsj;
  ST_retcode rc;

  char temps[500], algorithm[500];

  /* header of the plugin */
  SF_display("\n");
  SF_display("====================\n");
  SF_display("Start of the plugin\n");
  SF_display("\n");

  /* overview of variables and arguments passed and observations in sample */
  nvars = SF_nvars();
  nobs = SF_nobs();
  first = SF_in1();
  last = SF_in2();
  sprintf(temps, "number of vars & obs = %i, %i\n", nvars, nobs);
  SF_display(temps);
  sprintf(temps, "first and last obs in sample = %i, %i\n", first, last);
  SF_display(temps);
  SF_display("\n");

  for (i = 0; i < argc; i++) {
    sprintf(temps, "arg %i: ", i);
    SF_display(temps);
    SF_display(argv[i]);
    SF_display("\n");
  }
  SF_display("\n");

  theta = atof(argv[0]); /* contains value of theta = first argument */
  sprintf(temps, "theta = %6.4f\n", theta);
  SF_display(temps);
  SF_display("\n");

  /* allocation of string variable algorithm based on third argument */
  sprintf(algorithm, "%s", argv[2]);
  sprintf(temps, "algorithm = %s\n", algorithm);
  SF_display(temps);
  SF_display("\n");

  /* allocation of variable force_compute based on fourth argument */
  if (strcmp(argv[3], "force") == 0)
    force_compute = 1;
  else
    force_compute = 0;
  sprintf(temps, "force compute = %i\n", force_compute);
  SF_display(temps);
  SF_display("\n");

  /* allocation of variable missingdistance based on fifth argument */
  missingdistance = atof(argv[4]);
  sprintf(temps, "missing distance = %f\n", missingdistance);
  SF_display(temps);
  SF_display("\n");

  /* allocation of number of columns in manifold */
  mani = atoi(argv[5]);
  sprintf(temps, "number of variables in manifold = %i \n", mani);
  SF_display(temps);
  SF_display("\n");

  /* allocation of train_use, predict_use and skip_obs variables */
  train_use = (ST_double*)malloc(sizeof(ST_double) * nobs);
  predict_use = (ST_double*)malloc(sizeof(ST_double) * nobs);
  skip_obs = (ST_double*)malloc(sizeof(ST_double) * nobs);
  if ((train_use == NULL) || (predict_use == NULL) || (skip_obs == NULL)) {
    return print_error(MALLOC_ERROR);
  }

  count_train_set = 0;
  count_predict_set = 0;
  for (i = 1; i <= (last - first + 1); i++) {
    SF_vdata(mani + 3, i, &value);
    train_use[i - 1] = value;
    if (value == 1.)
      count_train_set++;
    if (SF_is_missing(value)) {
      /* missing value */
      train_use[i - 1] = MISSING;
    }
    SF_vdata(mani + 4, i, &value);
    predict_use[i - 1] = value;
    if (value == 1.)
      count_predict_set++;
    if (SF_is_missing(value)) {
      /* missing value */
      predict_use[i - 1] = MISSING;
    }
    SF_vdata(mani + 5, i, &value);
    skip_obs[i - 1] = value;
    if (SF_is_missing(value)) {
      /* missing value */
      skip_obs[i - 1] = MISSING;
    }
  }
  sprintf(temps, "train set obs: %i\n", count_train_set);
  SF_display(temps);
  sprintf(temps, "predict set obs: %i\n", count_predict_set);
  SF_display(temps);
  SF_display("\n");

  /* allocation of matrices M and y */
  ST_double* flat_M = malloc(sizeof(ST_double) * count_train_set * mani);
  gsl_matrix_view M_view = gsl_matrix_view_array(flat_M, count_train_set, mani);
  M = &(M_view.matrix);

  y = (ST_double*)malloc(sizeof(ST_double) * count_train_set);

  obsi = 0;
  for (i = 0; i < nobs; i++) {
    if (train_use[i] == 1.) {
      for (j = 0; j < mani; j++) {
        SF_vdata(j + 1, i + 1, &value);
        gsl_matrix_set(M, obsi, j, value);
        if (SF_is_missing(value)) {
          /* missing value */
          gsl_matrix_set(M, obsi, j, MISSING);
        }
      }
      SF_vdata(j + 1, i + 1, &value);
      y[obsi] = value;
      if (SF_is_missing(value)) {
        /* missing value */
        y[obsi] = MISSING;
      }
      obsi++;
    }
  }

  /* allocation of matrices Mp, S, ystar */
  pmani_flag = atoi(argv[6]); /* contains the flag for p_manifold */
  sprintf(temps, "p_manifold flag = %i \n", pmani_flag);
  SF_display(temps);

  if (pmani_flag == 1) {
    pmani = atoi(argv[8]); /* contains the number of columns in p_manifold */
    sprintf(temps, "number of variables in p_manifold = %i \n", pmani);
    SF_display(temps);
    Mpcol = pmani;
  } else {
    Mpcol = mani;
  }
  SF_display("\n");

  double* flat_Mp = NULL;
  gsl_matrix_view Mp_view;
  S = (ST_double*)malloc(sizeof(ST_double) * count_predict_set);
  if (S == NULL) {
    return print_error(MALLOC_ERROR);
  }

  if (pmani_flag == 1) {
    flat_Mp = malloc(sizeof(ST_double) * count_predict_set * pmani);
    Mp_view = gsl_matrix_view_array(flat_Mp, count_predict_set, pmani);
    Mp = &(Mp_view.matrix);

    smaploc = mani + 5 + pmani + 1;
    obsi = 0;

    for (i = 0; i < nobs; i++) {
      if (predict_use[i] == 1.) {
        obsj = 0;
        for (j = mani + 5; j < mani + 5 + pmani; j++) {
          SF_vdata(j + 1, i + 1, &value);
          gsl_matrix_set(Mp, obsi, obsj, value);
          if (SF_is_missing(value)) {
            /* missing value */
            gsl_matrix_set(Mp, obsi, obsj, MISSING);
          }
          obsj++;
        }
        S[obsi] = skip_obs[i];
        obsi++;
      }
    }

  } else {
    flat_Mp = malloc(sizeof(ST_double) * count_predict_set * mani);
    Mp_view = gsl_matrix_view_array(flat_Mp, count_predict_set, mani);
    Mp = &(Mp_view.matrix);

    smaploc = mani + 5 + 1;
    obsi = 0;
    for (i = 0; i < nobs; i++) {
      if (predict_use[i] == 1.) {
        for (j = 0; j < mani; j++) {
          SF_vdata(j + 1, i + 1, &value);
          gsl_matrix_set(Mp, obsi, j, value);
          if (SF_is_missing(value)) {
            /* missing value */
            gsl_matrix_set(Mp, obsi, j, MISSING);
          }
        }
        S[obsi] = skip_obs[i];
        obsi++;
      }
    }
  }

  l = atoi(argv[1]); /* contains l */
  if (l <= 0) {
    l = mani + 1;
  }
  sprintf(temps, "l = %i \n", l);
  SF_display(temps);
  SF_display("\n");

  vsave_flag = atoi(argv[7]); /* contains the flag for vars_save */

  double* flat_Bi_map = NULL;
  gsl_matrix_view Bi_map_view;
  Bi_map = NULL;

  if (vsave_flag == 1) { /* flag savesmap is ON */
    save_mode = 1;
    varssv = atoi(argv[8]); /* contains the number of columns
                               in smap coefficents */
    flat_Bi_map = malloc(sizeof(ST_double) * count_predict_set * varssv);
    if (flat_Bi_map == NULL) {
      return print_error(MALLOC_ERROR);
    }
    Bi_map_view = gsl_matrix_view_array(flat_Bi_map, count_predict_set, varssv);
    Bi_map = &Bi_map_view.matrix;

    sprintf(temps, "columns in smap coefficents = %i \n", varssv);
    SF_display(temps);

  } else { /* flag savesmap is OFF */
    save_mode = 0;
    Bi_map = NULL;
    varssv = 0;
  }

  sprintf(temps, "save_mode = %i \n", save_mode);
  SF_display(temps);
  SF_display("\n");

  ystar = (ST_double*)malloc(sizeof(ST_double) * count_predict_set);
  if (ystar == NULL) {
    return print_error(MALLOC_ERROR);
  }

  /* setting the number of OpenMP threads */
  nthreads = atoi(argv[9]);
  sprintf(temps, "Requested %i OpenMP threads \n", nthreads);
  SF_display(temps);
  nthreads = nthreads <= 0 ? omp_get_num_procs() : nthreads;
  sprintf(temps, "Using %i OpenMP threads \n", nthreads);
  SF_display(temps);
  SF_display("\n");
  omp_set_num_threads(nthreads);

#ifdef DUMP_INPUT
  // Here we want to dump the input so we can use it without stata for
  // debugging and profiling purposes.
  if (argc >= 11) {
    hid_t fid = H5Fcreate(argv[10], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    H5LTset_attribute_int(fid, "/", "count_train_set", &count_train_set, 1);
    H5LTset_attribute_int(fid, "/", "count_predict_set", &count_predict_set, 1);
    H5LTset_attribute_int(fid, "/", "Mpcol", &Mpcol, 1);
    H5LTset_attribute_int(fid, "/", "mani", &mani, 1);

    H5LTmake_dataset_double(fid, "y", 1, (hsize_t[]){ count_train_set }, y);

    H5LTset_attribute_int(fid, "/", "l", &l, 1);
    H5LTset_attribute_double(fid, "/", "theta", &theta, 1);

    H5LTmake_dataset_double(fid, "S", 1, (hsize_t[]){ count_predict_set }, S);

    H5LTset_attribute_string(fid, "/", "algorithm", algorithm);
    H5LTset_attribute_int(fid, "/", "save_mode", &save_mode, 1);
    H5LTset_attribute_int(fid, "/", "varssv", &varssv, 1);
    H5LTset_attribute_int(fid, "/", "force_compute", &force_compute, 1);
    H5LTset_attribute_double(fid, "/", "missingdistance", &missingdistance, 1);

    H5LTmake_dataset_double(fid, "flat_Mp", 1, (hsize_t[]){ count_predict_set * Mpcol }, flat_Mp);
    H5LTmake_dataset_double(fid, "flat_M", 1, (hsize_t[]){ count_train_set * mani }, flat_M);

    H5Fclose(fid);
  }
#endif

  rc = mf_smap_loop(count_predict_set, count_train_set, mani, M, Mp, y, l, theta, S, algorithm, save_mode, varssv,
                    force_compute, missingdistance, ystar, Bi_map);

  /* If there are no errors, return the value of ystar (and smap coefficients) to Stata */
  if (rc == SUCCESS) {
    j = 0;
    for (i = 0; i < nobs; i++) {
      if (predict_use[i] == 1) {
        if (ystar[j] != MISSING) {
          SF_vstore(mani + 2, i + 1, ystar[j]);
        } else {
          /* returning a missing value */
          SF_vstore(mani + 2, i + 1, SV_missval);
        }
        if (save_mode) {
          for (h = 0; h < varssv; h++) {
            if (gsl_matrix_get(Bi_map, j, h) != MISSING) {
              SF_vstore(smaploc + h, i + 1, gsl_matrix_get(Bi_map, j, h));
            } else {
              SF_vstore(smaploc + h, i + 1, SV_missval);
            }
          }
        }
        j++;
      }
    }
  } else {
    print_error(rc);
  }

  /* deallocation of matrices and arrays before exiting the plugin */
  free(train_use);
  free(predict_use);
  free(S);
  free(flat_M);
  free(y);
  free(flat_Mp);
  if (save_mode) {
    free(flat_Bi_map);
  }
  free(ystar);

  /* footer of the plugin */
  SF_display("\n");
  SF_display("End of the plugin\n");
  SF_display("====================\n");
  SF_display("\n");

  return rc;
}
