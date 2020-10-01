#include "edm.h"
#include <hdf5.h>
#include <hdf5_hl.h>
#include <stdlib.h>
#include <string.h>

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

static void write_results(const char* fname_out, ST_double* ystar, ST_double* flat_Bi_map, const InputVars* vars)
{
  hid_t fid = H5Fcreate(fname_out, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  H5LTmake_dataset_double(fid, "ystar", 1, (hsize_t[]){ vars->count_predict_set }, ystar);
  H5LTmake_dataset_double(fid, "Bi_map", 2, (hsize_t[]){ vars->count_predict_set, vars->varssv }, flat_Bi_map);

  H5Fclose(fid);
}

static void call_mf_smap_loop(const InputVars* vars, const char* fname_out)
{
  gsl_matrix_view M_view = gsl_matrix_view_array(vars->flat_M, vars->count_train_set, vars->mani);
  gsl_matrix* M = &(M_view.matrix);

  gsl_matrix_view Mp_view = gsl_matrix_view_array(vars->flat_Mp, vars->count_predict_set, vars->Mpcol);
  gsl_matrix* Mp = &(Mp_view.matrix);

  ST_double* ystar = malloc(sizeof(ST_double) * vars->count_predict_set);
  ST_double* flat_Bi_map = malloc(sizeof(ST_double) * vars->count_predict_set * vars->varssv);
  gsl_matrix_view Bi_map_view = gsl_matrix_view_array(flat_Bi_map, vars->count_predict_set, vars->varssv);
  gsl_matrix* Bi_map = &Bi_map_view.matrix;

  mf_smap_loop(vars->count_predict_set, vars->count_train_set, vars->mani, M, Mp, vars->y, vars->l, vars->theta,
               vars->S, (char*)(vars->algorithm), vars->save_mode, vars->varssv, vars->force_compute,
               vars->missingdistance, ystar, Bi_map);

  write_results(fname_out, ystar, flat_Bi_map, vars);

  free(ystar);
  free(flat_Bi_map);
}

static char* generate_output_fname(const char* fname_in)
{
  char* ext = strrchr(fname_in, '.');

  if (!ext || (ext == fname_in))
    ext = strchr(fname_in, '\0') - 1;

  printf("ext = %s\n", ext);

  const size_t fname_out_size = ext - fname_in + 8;
  char* fname_out = malloc(sizeof(char) * fname_out_size);

  strncpy(fname_out, fname_in, ext - fname_in);

  printf("before sprintf = %s\n", fname_out);

  sprintf(fname_out + (ext - fname_in), "-out.h5");

  return fname_out;
}

int main(int argc, char* argv[])
{

  if (argc != 2) {
    fprintf(stderr, "Usage: ./driver <fname>\n");
    return -1;
  }

  InputVars vars = new_InputVars();
  read_dumpfile(argv[1], &vars);

  char* fname_out = generate_output_fname(argv[1]); // WATCH OUT: fname_out is malloc'd in this function!

  call_mf_smap_loop(&vars, fname_out);

  free(fname_out);
  free_InputVars(&vars);
  return 0;
}
