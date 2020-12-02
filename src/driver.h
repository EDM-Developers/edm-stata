#include "edm.h"
#include <hdf5.h>
#include <hdf5_hl.h>
#include <iostream>

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

struct Inputs
{
  Options opts;
  Manifold M, Mp;
  std::vector<double> y;
};

void save_options(hid_t fid, Options opts)
{
  char boolVar = (char)opts.forceCompute;
  H5LTset_attribute_char(fid, "/", "forceCompute", &boolVar, 1);
  boolVar = (char)opts.saveMode;
  H5LTset_attribute_char(fid, "/", "saveMode", &boolVar, 1);
  boolVar = (char)opts.distributeThreads;
  H5LTset_attribute_char(fid, "/", "distributeThreads", &boolVar, 1);

  H5LTset_attribute_int(fid, "/", "k", &opts.k, 1);
  H5LTset_attribute_int(fid, "/", "varssv", &opts.varssv, 1);

  H5LTset_attribute_double(fid, "/", "theta", &opts.thetas[0], 1);
  H5LTset_attribute_double(fid, "/", "missingdistance", &opts.missingdistance, 1);

  H5LTset_attribute_string(fid, "/", "algorithm", opts.algorithm.c_str());

  H5LTset_attribute_int(fid, "/", "nthreads", &opts.nthreads, 1);
}

Options read_options(hid_t fid)
{
  Options opts;

  char boolVar;
  H5LTget_attribute_char(fid, "/", "forceCompute", &boolVar);
  opts.forceCompute = (bool)boolVar;
  H5LTget_attribute_char(fid, "/", "saveMode", &boolVar);
  opts.saveMode = (bool)boolVar;
  H5LTget_attribute_char(fid, "/", "distributeThreads", &boolVar);
  opts.distributeThreads = (bool)boolVar;

  H5LTget_attribute_int(fid, "/", "k", &(opts.k));
  H5LTget_attribute_int(fid, "/", "varssv", &(opts.varssv));

  double theta;
  H5LTget_attribute_double(fid, "/", "theta", &theta);
  opts.thetas.push_back(theta);

  H5LTget_attribute_double(fid, "/", "missingdistance", &(opts.missingdistance));

  char temps[100];
  H5LTget_attribute_string(fid, "/", "algorithm", temps);
  opts.algorithm = std::string(temps);

  H5LTget_attribute_int(fid, "/", "nthreads", &opts.nthreads);

  return opts;
}

void save_manifold(const hid_t& fid, std::string name, std::vector<double> x, std::vector<std::vector<double>> extras,
                   std::vector<bool> filter)
{
  hsize_t size = x.size();
  H5LTmake_dataset_double(fid, (name + ".x").c_str(), 1, &size, x.data());

  std::vector<char> filterChar(size);
  for (int i = 0; i < size; i++) {
    filterChar[i] = (char)filter[i];
  }
  H5LTmake_dataset_char(fid, (name + ".filter").c_str(), 1, &size, filterChar.data());

  hsize_t extrasSize = extras.size();
  if (extrasSize > 0 && extras[0].size() > 0) {
    extrasSize *= extras[0].size();

    std::vector<double> extrasFlat(extrasSize);
    for (int i = 0; i < extras[0].size(); i++) {
      for (int j = 0; j < extras.size(); j++) {
        extrasFlat[i * extras[0].size() + j] = extras[j][i];
      }
    }

    H5LTmake_dataset_double(fid, (name + ".extras").c_str(), 1, &extrasSize, extrasFlat.data());
  }
}

Manifold read_manifold(hid_t fid, std::string name, std::vector<double> t, size_t E, double dtWeight)
{
  hsize_t size;
  H5LTget_dataset_info(fid, (name + ".x").c_str(), &size, NULL, NULL);

  std::vector<double> x(size);
  std::vector<char> filterChar(size);

  H5LTread_dataset_double(fid, (name + ".x").c_str(), x.data());
  H5LTread_dataset_char(fid, (name + ".filter").c_str(), filterChar.data());

  std::vector<bool> filter(size);
  for (int i = 0; i < size; i++) {
    filter[i] = (bool)filterChar[i];
  }

  std::vector<std::vector<double>> extras;

  if (H5LTfind_dataset(fid, (name + ".extras").c_str())) {
    hsize_t extrasSize[2];

    H5LTget_dataset_info(fid, (name + ".extras").c_str(), extrasSize, NULL, NULL);

    std::vector<double> extrasFlat(extrasSize[0] * extrasSize[1]);

    H5LTread_dataset_double(fid, (name + ".extras").c_str(), extrasFlat.data());

    extras = std::vector<std::vector<double>>(extrasSize[1]);
    for (int j = 0; j < extras.size(); j++) {
      extras[j] = std::vector<double>(extrasSize[0]);

      for (int i = 0; i < extras[0].size(); i++) {
        extras[j][i] = extrasFlat[i * extras[0].size() + j];
      }
    }
  }

  return Manifold(x, t, extras, filter, E, dtWeight, MISSING);
}

/*! \brief Read in a dump file.
 *
 * Read in a dump file created with compile flag DUMP_INPUT.
 *
 * \param fname dump filename
 * \param pointer to InputVars struct to store the read
 */
Inputs read_dumpfile(std::string fname_in)
{
  hid_t fid = H5Fopen(fname_in.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  Options opts = read_options(fid);

  unsigned Eint;
  H5LTget_attribute_uint(fid, "/", "E", &Eint);
  size_t E = (size_t)Eint;

  double dtWeight;
  H5LTget_attribute_double(fid, "/", "dtWeight", &dtWeight);

  std::vector<double> t;
  if (H5LTfind_dataset(fid, "t")) {
    hsize_t size;
    H5LTget_dataset_info(fid, "t", &size, NULL, NULL);
    t = std::vector<double>(size);
    H5LTread_dataset_double(fid, "t", t.data());
  }

  Manifold M = read_manifold(fid, "M", t, E, dtWeight);
  Manifold Mp = read_manifold(fid, "Mp", t, E, dtWeight);

  hsize_t size;
  H5LTget_dataset_info(fid, "y", &size, NULL, NULL);
  std::vector<double> y(size);
  H5LTread_dataset_double(fid, "y", y.data());

  H5Fclose(fid);

  return { opts, M, Mp, y };
}

void write_dumpfile(const char* fname, const Options& opts, const std::vector<double>& t, const std::vector<double>& x,
                    const std::vector<double>& xPred, const std::vector<std::vector<double>>& extras,
                    const std::vector<std::vector<double>>& extrasPred, const std::vector<bool>& trainingRows,
                    const std::vector<bool>& predictionRows, const std::vector<double>& y, int E, double dtWeight)
{
  hid_t fid = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  save_options(fid, opts);

  unsigned Eint = (unsigned)E;
  H5LTset_attribute_uint(fid, "/", "E", &Eint, 1);
  H5LTset_attribute_double(fid, "/", "dtWeight", &dtWeight, 1);

  hsize_t tSize = t.size();
  if (tSize > 0) {
    H5LTmake_dataset_double(fid, "t", 1, &tSize, t.data());
  }

  save_manifold(fid, "M", x, extras, trainingRows);
  save_manifold(fid, "Mp", xPred, extrasPred, predictionRows);

  hsize_t yLen = y.size();
  H5LTmake_dataset_double(fid, "y", 1, &yLen, y.data());

  H5Fclose(fid);
}

void write_results(std::string fname_out, const Prediction& smap_res, int varssv)
{
  // hid_t fid = H5Fcreate(fname_out.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // H5LTset_attribute_int(fid, "/", "rc", &(smap_res.rc), 1);

  // hsize_t ystarLen[] = { (hsize_t)smap_res.ystar.size() };
  // H5LTmake_dataset_double(fid, "ystar", 1, ystarLen, smap_res.ystar.data());

  // if (smap_res.flat_Bi_map.has_value()) {
  //   hsize_t Bi_mapLen[] = { (hsize_t)(smap_res.flat_Bi_map->size() / varssv), (hsize_t)varssv };
  //   H5LTmake_dataset_double(fid, "flat_Bi_map", 2, Bi_mapLen, smap_res.flat_Bi_map->data());
  // }

  // H5Fclose(fid);
}