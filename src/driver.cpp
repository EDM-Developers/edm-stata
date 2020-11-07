/* Suppress Windows problems with sprintf etc. functions. */
#ifdef _MSC_VER
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#endif

#include "edm.h"
#include <hdf5.h>
#include <hdf5_hl.h>
#include <iostream>
#include <limits>

/*! \struct Input
 *  \brief The input variables for an mf_smap_loop call.
 */
typedef struct
{
  smap_opts_t opts;
  std::vector<double> y;
  Manifold M, Mp;
  int nthreads;
} edm_inputs_t;

void display(char* s)
{
  std::cout << s;
}

void flush()
{
  fflush(stdout);
}

/*! \brief Read in a dump file.
 *
 * Read in a dump file created with compile flag DUMP_INPUT.
 *
 * \param fname dump filename
 * \param pointer to InputVars struct to store the read
 */
edm_inputs_t read_dumpfile(std::string fname_in)
{
  hid_t fid = H5Fopen(fname_in.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  smap_opts_t opts;

  char bool_var;
  H5LTget_attribute_char(fid, "/", "save_mode", &bool_var);
  opts.save_mode = (bool)bool_var;
  H5LTget_attribute_char(fid, "/", "force_compute", &bool_var);
  opts.force_compute = (bool)bool_var;

  H5LTget_attribute_int(fid, "/", "l", &(opts.l));
  H5LTget_attribute_int(fid, "/", "varssv", &(opts.varssv));

  H5LTget_attribute_double(fid, "/", "theta", &(opts.theta));
  H5LTget_attribute_double(fid, "/", "missingdistance", &(opts.missingdistance));

  char temps[100];
  H5LTget_attribute_string(fid, "/", "algorithm", temps);
  opts.algorithm = std::string(temps);

  Manifold M, Mp;

  int dim;
  H5LTget_attribute_int(fid, "/", "count_train_set", &dim);
  M._rows = dim;
  H5LTget_attribute_int(fid, "/", "mani", &dim);
  M._cols = dim;

  M.flat = std::vector<double>(M.rows() * M.cols());
  H5LTread_dataset_double(fid, "flat_M", M.flat.data());

  H5LTget_attribute_int(fid, "/", "count_predict_set", &dim);
  Mp._rows = dim;
  H5LTget_attribute_int(fid, "/", "Mpcol", &dim);
  Mp._cols = dim;

  Mp.flat = std::vector<double>(Mp.rows() * Mp.cols());
  H5LTread_dataset_double(fid, "flat_Mp", Mp.flat.data());

  std::vector<double> y(M.rows());
  H5LTread_dataset_double(fid, "y", y.data());

  int nthreads;
  H5LTget_attribute_int(fid, "/", "nthreads", &nthreads);

  H5Fclose(fid);

  return { opts, y, M, Mp, nthreads };
}

void write_results(std::string fname_out, const smap_res_t& smap_res, int varssv)
{
  hid_t fid = H5Fcreate(fname_out.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  H5LTset_attribute_int(fid, "/", "rc", &(smap_res.rc), 1);

  hsize_t ystarLen[] = { (hsize_t)smap_res.ystar.size() };
  H5LTmake_dataset_double(fid, "ystar", 1, ystarLen, smap_res.ystar.data());

  if (smap_res.flat_Bi_map.has_value()) {
    hsize_t Bi_mapLen[] = { (hsize_t)(smap_res.flat_Bi_map->size() / varssv), (hsize_t)varssv };
    H5LTmake_dataset_double(fid, "flat_Bi_map", 2, Bi_mapLen, smap_res.flat_Bi_map->data());
  }

  H5Fclose(fid);
}

int main(int argc, char* argv[])
{

  if (argc != 2) {
    fprintf(stderr, "Usage: ./driver <fname>\n");
    return -1;
  }

  std::string fname_in(argv[1]);

  edm_inputs_t vars = read_dumpfile(fname_in);

  int VERBOSITY = std::numeric_limits<int>::max();
  smap_res_t smap_res = mf_smap_loop(vars.opts, vars.y, vars.M, vars.Mp, vars.nthreads, display, flush, VERBOSITY);

  std::size_t ext = fname_in.find_last_of(".");
  fname_in = fname_in.substr(0, ext);
  std::string fname_out = fname_in + "-out.h5";

  write_results(fname_out, smap_res, vars.opts.varssv);

  return smap_res.rc;
}