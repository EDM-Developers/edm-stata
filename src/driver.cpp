/* Suppress Windows problems with sprintf etc. functions. */
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

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
  std::vector<double> y;
  std::vector<double> S;
  std::vector<double> flat_Mp;
  std::vector<double> flat_M;
  double theta;
  double missingdistance;
  int count_train_set;
  int count_predict_set;
  int Mpcol;
  int mani;
  int l;
  bool save_mode;
  int varssv;
  bool force_compute;
} InputVars;

/*! \brief Read in a dump file.
 *
 * Read in a dump file created with compile flag DUMP_INPUT.
 *
 * \param fname dump filename
 * \param pointer to InputVars struct to store the read
 */
InputVars read_dumpfile(const char* fname)
{
  InputVars vars;

  hid_t fid = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);

  H5LTget_attribute_int(fid, "/", "count_train_set", &(vars.count_train_set));
  H5LTget_attribute_int(fid, "/", "count_predict_set", &(vars.count_predict_set));
  H5LTget_attribute_int(fid, "/", "Mpcol", &(vars.Mpcol));
  H5LTget_attribute_int(fid, "/", "mani", &(vars.mani));

  vars.y = std::vector<double>(vars.count_train_set);
  H5LTread_dataset_double(fid, "y", vars.y.data());

  H5LTget_attribute_int(fid, "/", "l", &(vars.l));
  H5LTget_attribute_double(fid, "/", "theta", &(vars.theta));

  vars.S = std::vector<double>(vars.count_predict_set);
  H5LTread_dataset_double(fid, "S", vars.S.data());

  H5LTget_attribute_string(fid, "/", "algorithm", vars.algorithm);

  char bool_var;
  H5LTget_attribute_char(fid, "/", "save_mode", &bool_var);
  vars.save_mode = (bool)bool_var;
  H5LTget_attribute_char(fid, "/", "force_compute", &bool_var);
  vars.force_compute = (bool)bool_var;

  H5LTget_attribute_int(fid, "/", "varssv", &(vars.varssv));

  H5LTget_attribute_double(fid, "/", "missingdistance", &(vars.missingdistance));

  vars.flat_Mp = std::vector<double>(vars.count_predict_set * vars.Mpcol);
  H5LTread_dataset_double(fid, "flat_Mp", vars.flat_Mp.data());

  vars.flat_M = std::vector<double>(vars.count_train_set * vars.mani);
  H5LTread_dataset_double(fid, "flat_M", vars.flat_M.data());

  H5Fclose(fid);

  return vars;
}

void write_results(const char* fname_out, const smap_res_t& smap_res, const InputVars& vars)
{
  hid_t fid = H5Fcreate(fname_out, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  H5LTset_attribute_int(fid, "/", "rc", &(smap_res.rc), 1);

  hsize_t ystarLen[] = { (hsize_t)vars.count_predict_set };
  H5LTmake_dataset_double(fid, "ystar", 1, ystarLen, smap_res.ystar.data());

  if (smap_res.flat_Bi_map.has_value()) {
    hsize_t Bi_mapLen[] = { (hsize_t)vars.count_predict_set, (hsize_t)vars.varssv };
    H5LTmake_dataset_double(fid, "flat_Bi_map", 2, Bi_mapLen, smap_res.flat_Bi_map->data());
  }

  H5Fclose(fid);
}

char* generate_output_fname(const char* fname_in)
{
  const char* ext = strrchr(fname_in, '.');

  if (!ext || (ext == fname_in))
    ext = strchr(fname_in, '\0') - 1;

  const size_t fname_out_size = ext - fname_in + 8;
  char* fname_out = (char*)malloc(sizeof(char) * fname_out_size);

  strncpy(fname_out, fname_in, ext - fname_in);

  sprintf(fname_out + (ext - fname_in), "-out.h5");

  return fname_out;
}

int main(int argc, char* argv[])
{

  if (argc != 2) {
    fprintf(stderr, "Usage: ./driver <fname>\n");
    return -1;
  }

  InputVars vars = read_dumpfile(argv[1]);

  char* fname_out = generate_output_fname(argv[1]); // WATCH OUT: fname_out is malloc'd in this function!

  smap_res_t smap_res = mf_smap_loop(vars.count_predict_set, vars.count_train_set, vars.mani, vars.Mpcol, vars.l,
                                     vars.theta, vars.algorithm, vars.save_mode, vars.varssv, vars.force_compute,
                                     vars.missingdistance, vars.y, vars.S, vars.flat_M, vars.flat_Mp);

  write_results(fname_out, smap_res, vars);

  free(fname_out);

  return 0;
}