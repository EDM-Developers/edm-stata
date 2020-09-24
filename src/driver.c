#include "smap_block_mdap.h"
#include <hdf5.h>
#include <hdf5_hl.h>
#include <stdlib.h>

#define EMPTY_INT -999
#define EMPTY_DOUBLE -999.99

/*! \struct Input
 *  \brief The input variables for an mf_smap_loop call.
 */
typedef struct InputVars
{
  char algorithm[500];
  ST_double* y;
  ST_double* S;
  ST_double* flat_Mp;
  ST_double* flat_M;
  ST_double theta;
  ST_double missingdistance;
  ST_int count_train_set;
  ST_int count_predict_set;
  ST_int Mpcol;
  ST_int mani;
  ST_int l;
  ST_int save_mode;
  ST_int varssv;
  ST_int force_compute;
} InputVars;

InputVars new_InputVars()
{
  InputVars vars;

  vars.y = NULL;
  vars.S = NULL;
  vars.flat_Mp = NULL;
  vars.flat_M = NULL;
  vars.theta = EMPTY_DOUBLE;
  vars.missingdistance = EMPTY_DOUBLE;

  vars.count_train_set = EMPTY_INT;
  vars.count_predict_set = EMPTY_INT;
  vars.Mpcol = EMPTY_INT;
  vars.mani = EMPTY_INT;
  vars.l = EMPTY_INT;
  vars.save_mode = EMPTY_INT;
  vars.varssv = EMPTY_INT;
  vars.force_compute = EMPTY_INT;

  return vars;
}

void free_InputVars(InputVars* vars)
{
  if (vars->flat_M != NULL)
    free(vars->flat_M);
  if (vars->flat_Mp != NULL)
    free(vars->flat_Mp);
  if (vars->S != NULL)
    free(vars->S);
  if (vars->y != NULL)
    free(vars->y);

  vars->count_train_set = EMPTY_INT;
  vars->count_predict_set = EMPTY_INT;
  vars->Mpcol = EMPTY_INT;
  vars->mani = EMPTY_INT;
  vars->l = EMPTY_INT;
  vars->missingdistance = EMPTY_DOUBLE;
  vars->theta = EMPTY_DOUBLE;
  vars->save_mode = EMPTY_INT;
  vars->varssv = EMPTY_INT;
  vars->force_compute = EMPTY_INT;
}

/*! \brief Read in a dump file.
 *
 * Read in a dump file created with compile flag DUMP_INPUT.
 *
 * \param fname dump filename
 * \param pointer to InputVars struct to store the read
 */
static void read_dumpfile(const char* fname, InputVars* vars)
{
  hid_t fid = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

  H5LTget_attribute_int(fid, "/", "count_train_set", &(vars->count_train_set));
  H5LTget_attribute_int(fid, "/", "count_predict_set", &(vars->count_predict_set));
  H5LTget_attribute_int(fid, "/", "Mpcol", &(vars->Mpcol));
  H5LTget_attribute_int(fid, "/", "mani", &(vars->mani));

  vars->y = malloc(vars->count_train_set * sizeof(ST_double));
  H5LTread_dataset_double(fid, "y", vars->y);

  H5LTget_attribute_int(fid, "/", "l", &(vars->l));
  H5LTget_attribute_double(fid, "/", "theta", &(vars->theta));

  vars->S = malloc(vars->count_predict_set * sizeof(ST_double));
  H5LTread_dataset_double(fid, "S", vars->S);

  H5LTget_attribute_string(fid, "/", "algorithm", vars->algorithm);
  H5LTget_attribute_int(fid, "/", "save_mode", &(vars->save_mode));
  H5LTget_attribute_int(fid, "/", "varssv", &(vars->varssv));

  H5LTget_attribute_int(fid, "/", "force_compute", &(vars->force_compute));
  H5LTget_attribute_double(fid, "/", "missingdistance", &(vars->missingdistance));

  vars->flat_Mp = malloc(vars->count_predict_set * vars->Mpcol * sizeof(ST_double));
  H5LTread_dataset_double(fid, "flat_Mp", vars->flat_Mp);
  vars->flat_M = malloc(vars->count_train_set * vars->mani * sizeof(ST_double));
  H5LTread_dataset_double(fid, "flat_M", vars->flat_M);

  H5Fclose(fid);
}

static void call_mf_smap_loop(const InputVars* vars)
{
  ST_double** M = alloc_matrix(vars->count_train_set, vars->mani);
  for (int i = 0; i < vars->count_train_set; ++i)
    for (int j = 0; j < vars->mani; ++j)
      M[i][j] = vars->flat_M[i * vars->mani + j];

  ST_double** Mp = alloc_matrix(vars->count_predict_set, vars->Mpcol);
  for (int i = 0; i < vars->count_predict_set; ++i)
    for (int j = 0; j < vars->Mpcol; ++j)
      Mp[i][j] = vars->flat_Mp[i * vars->Mpcol + j];

  ST_double* ystar = malloc(sizeof(ST_double) * vars->count_predict_set);
  ST_double** Bi_map = alloc_matrix(vars->count_predict_set, vars->varssv);

  mf_smap_loop(vars->count_predict_set, vars->count_train_set, Bi_map, vars->mani, M, Mp, vars->y, vars->l, vars->theta,
               vars->S, (char*)(vars->algorithm), vars->save_mode, vars->varssv, vars->force_compute,
               vars->missingdistance, ystar);

  free_matrix(Bi_map, vars->count_predict_set);
  free(ystar);
  free_matrix(Mp, vars->count_predict_set);
  free_matrix(M, vars->count_train_set);
}

int main(int argc, char* argv[])
{

  if (argc != 2) {
    fprintf(stderr, "Usage: ./driver <fname>\n");
    return -1;
  }

  InputVars vars = new_InputVars();
  read_dumpfile(argv[1], &vars);

  call_mf_smap_loop(&vars);

  // TODO(smutch): dump output

  free_InputVars(&vars);
  return 0;
}
