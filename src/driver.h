#include "edm.h"
#include <hdf5.h>
#include <hdf5_hl.h>
#include <iostream>

typedef struct
{
  std::vector<double> flat;
  int rows, cols;
} manifold_t;

/*! \struct Input
 *  \brief The input variables for an mf_smap_loop call.
 */
typedef struct
{
  smap_opts_t opts;
  std::vector<double> y;
  manifold_t M, Mp;
  int nthreads;
} edm_inputs_t;

class ConsoleIO : public IO
{
public:
  ConsoleIO() { this->verbosity = std::numeric_limits<int>::max(); }
  ConsoleIO(int v) { this->verbosity = v; }
  virtual void out(const char* s) const { std::cout << s; }
  virtual void out_async(const char* s) const { out(s); }
  virtual void error(const char* s) const { std::cerr << s; }
  virtual void flush() const { fflush(stdout); }
};

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

  manifold_t M, Mp;

  H5LTget_attribute_int(fid, "/", "count_train_set", &(M.rows));
  H5LTget_attribute_int(fid, "/", "mani", &(M.cols));

  M.flat = std::vector<double>(M.rows * M.cols);
  H5LTread_dataset_double(fid, "flat_M", M.flat.data());

  H5LTget_attribute_int(fid, "/", "count_predict_set", &(Mp.rows));
  H5LTget_attribute_int(fid, "/", "Mpcol", &(Mp.cols));

  Mp.flat = std::vector<double>(Mp.rows * Mp.cols);
  H5LTread_dataset_double(fid, "flat_Mp", Mp.flat.data());

  std::vector<double> y(M.rows);
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