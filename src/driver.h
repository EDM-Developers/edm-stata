#define DRIVER_MODE 1

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
  ManifoldGenerator generator;
  size_t E;
  std::vector<bool> trainingRows, predictionRows;
};

void save_options(hid_t fid, Options opts)
{
  char boolVar = (char)opts.forceCompute;
  H5LTset_attribute_char(fid, "/", "forceCompute", &boolVar, 1);
  boolVar = (char)opts.saveSMAPCoeffs;
  H5LTset_attribute_char(fid, "/", "saveSMAPCoeffs", &boolVar, 1);
  boolVar = (char)opts.distributeThreads;
  H5LTset_attribute_char(fid, "/", "distributeThreads", &boolVar, 1);

  H5LTset_attribute_int(fid, "/", "k", &opts.k, 1);

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
  H5LTget_attribute_char(fid, "/", "saveSMAPCoeffs", &boolVar);
  opts.saveSMAPCoeffs = (bool)boolVar;
  H5LTget_attribute_char(fid, "/", "distributeThreads", &boolVar);
  opts.distributeThreads = (bool)boolVar;

  H5LTget_attribute_int(fid, "/", "k", &(opts.k));

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

void save_manifold_generator(const hid_t& fid, const ManifoldGenerator& generator)
{
  hsize_t size = generator._x.size();
  H5LTmake_dataset_double(fid, "x", 1, &size, generator._x.data());

  H5LTmake_dataset_double(fid, "y", 1, &size, generator._y.data());

  if (generator._co_x.size() > 0) {
    H5LTmake_dataset_double(fid, "co_x", 1, &size, generator._co_x.data());
  }

  hsize_t numExtras = generator._extras.size();
  if (numExtras > 0 && generator._extras[0].size() > 0) {
    hsize_t numObs = generator._extras[0].size();
    hsize_t extrasSize = numExtras * numObs;

    std::vector<double> extrasFlat(extrasSize);
    for (int i = 0; i < numObs; i++) {
      for (int j = 0; j < numExtras; j++) {
        extrasFlat[i * numObs + j] = generator._extras[j][i];
      }
    }

    H5LTmake_dataset_double(fid, "extras", 1, &extrasSize, extrasFlat.data());
  }

  if (generator._t.size() > 0) {
    H5LTmake_dataset_double(fid, "time", 1, &size, generator._t.data());
  }

  H5LTset_attribute_double(fid, "/", "dtWeight", &generator._dtWeight, 1);
}

ManifoldGenerator read_manifold_generator(hid_t fid)
{
  hsize_t size;
  H5LTget_dataset_info(fid, "x", &size, NULL, NULL);

  std::vector<double> x(size);
  H5LTread_dataset_double(fid, "x", x.data());

  std::vector<double> y(size);
  H5LTread_dataset_double(fid, "y", y.data());

  std::vector<std::vector<double>> extras;
  if (H5LTfind_dataset(fid, "extras")) {
    hsize_t extrasSize[2];
    H5LTget_dataset_info(fid, "extras", extrasSize, NULL, NULL);
    std::vector<double> extrasFlat(extrasSize[0] * extrasSize[1]);
    H5LTread_dataset_double(fid, "extras", extrasFlat.data());
    extras = std::vector<std::vector<double>>(extrasSize[1]);
    for (int j = 0; j < extras.size(); j++) {
      extras[j] = std::vector<double>(extrasSize[0]);
      for (int i = 0; i < extras[0].size(); i++) {
        extras[j][i] = extrasFlat[i * extras[0].size() + j];
      }
    }
  }

  ManifoldGenerator generator(x, y, extras, MISSING);

  if (H5LTfind_dataset(fid, "co_x")) {
    std::vector<double> co_x = std::vector<double>(size);
    H5LTread_dataset_double(fid, "co_x", co_x.data());
    generator.add_coprediction_data(co_x);
  }

  // Bug in "H5LTfind_dataset" which return true for the column "t"
  // even when that dataset doesn't exist in the HDF5 file.
  if (H5LTfind_dataset(fid, "time")) {
    std::vector<double> t = std::vector<double>(size);
    H5LTread_dataset_double(fid, "time", t.data());

    double dtWeight;
    H5LTget_attribute_double(fid, "/", "dtWeight", &dtWeight);

    generator.add_dt_data(t, dtWeight);
  }

  return generator;
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

  ManifoldGenerator generator = read_manifold_generator(fid);

  unsigned Eint;
  H5LTget_attribute_uint(fid, "/", "E", &Eint);
  size_t E = (size_t)Eint;

  // Read in the training/prediction filters
  hsize_t size;
  H5LTget_dataset_info(fid, "trainingRows", &size, NULL, NULL);

  std::vector<char> filterChar(size);
  H5LTread_dataset_char(fid, "trainingRows", filterChar.data());

  std::vector<bool> trainingRows(size);
  for (int i = 0; i < size; i++) {
    trainingRows[i] = (bool)filterChar[i];
  }

  H5LTread_dataset_char(fid, "predictionRows", filterChar.data());
  std::vector<bool> predictionRows(size);
  for (int i = 0; i < size; i++) {
    predictionRows[i] = (bool)filterChar[i];
  }

  H5Fclose(fid);

  return { opts, generator, E, trainingRows, predictionRows };
}

void write_dumpfile(const char* fname, const Options& opts, const ManifoldGenerator& generator, int E,
                    const std::vector<bool>& trainingRows, const std::vector<bool>& predictionRows)
{
  hid_t fid = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  save_options(fid, opts);

  save_manifold_generator(fid, generator);

  unsigned Eint = (unsigned)E;
  H5LTset_attribute_uint(fid, "/", "E", &Eint, 1);

  hsize_t size = (hsize_t)trainingRows.size();
  std::vector<char> filterChar(size);
  for (int i = 0; i < size; i++) {
    filterChar[i] = (char)trainingRows[i];
  }
  H5LTmake_dataset_char(fid, "trainingRows", 1, &size, filterChar.data());
  for (int i = 0; i < size; i++) {
    filterChar[i] = (char)predictionRows[i];
  }
  H5LTmake_dataset_char(fid, "predictionRows", 1, &size, filterChar.data());

  H5Fclose(fid);
}

void write_results(std::string fname_out, const Prediction& pred)
{
  hid_t fid = H5Fcreate(fname_out.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  H5LTset_attribute_int(fid, "/", "rc", &(pred.rc), 1);

  hsize_t ystarLen[] = { (hsize_t)pred.numPredictions };
  H5LTmake_dataset_double(fid, "ystar", 1, ystarLen, pred.ystar.get());

  if (pred.numCoeffCols > 0) {
    hsize_t Bi_mapLen[] = { (hsize_t)pred.numPredictions, (hsize_t)pred.numCoeffCols };
    H5LTmake_dataset_double(fid, "coeffs", 2, Bi_mapLen, pred.coeffs.get());
  }

  H5LTset_attribute_double(fid, "/", "mae", &pred.stats.mae, 1);
  H5LTset_attribute_double(fid, "/", "rho", &pred.stats.rho, 1);

  H5Fclose(fid);
}