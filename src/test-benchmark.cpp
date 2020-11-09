#include <benchmark/benchmark.h>

#include "driver.h"
#include "edm.h"

void no_display(const char* s) {}

void no_flush() {}

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Core>

using MatrixView = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using MatrixRowView = Eigen::Block<const MatrixView, 1, -1, true>;

#include <Eigen/SVD>
#include <optional>

typedef struct
{
  std::vector<double> flat;
  int rows, cols;
} manifold_t;

typedef struct
{
  bool force_compute, save_mode;
  int l, varssv;
  double theta, missingdistance;
  std::string algorithm;
} smap_opts_t;

retcode fast_mf_smap_single(int Mp_i, smap_opts_t opts, const std::vector<double>& y, const MatrixView& M,
                            const MatrixView& Mp, std::vector<double>& ystar, std::optional<MatrixView>& Bi_map)
{
  int validDistances = 0;
  std::vector<double> d(M.rows());
  auto b = Mp.row(Mp_i);

  for (int i = 0; i < M.rows(); i++) {
    double dist = 0.;
    bool missing = false;
    int numMissingDims = 0;
    for (int j = 0; j < M.cols(); j++) {
      if ((M(i, j) == MISSING) || (b(j) == MISSING)) {
        if (opts.missingdistance == 0) {
          missing = true;
          break;
        }
        numMissingDims += 1;
      } else {
        dist += (M(i, j) - b(j)) * (M(i, j) - b(j));
      }
    }
    // If the distance between M_i and b is 0 before handling missing values,
    // then keep it at 0. Otherwise, add in the correct number of missingdistance's.
    if (dist != 0) {
      dist += numMissingDims * opts.missingdistance * opts.missingdistance;
    }

    if (missing || dist == 0.) {
      d[i] = MISSING;
    } else {
      d[i] = dist;
      validDistances += 1;
    }
  }

  // If we only look at distances which are non-zero and non-missing,
  // do we have enough of them to find 'l' neighbours?
  int l = opts.l;
  if (l > validDistances) {
    if (opts.force_compute) {
      l = validDistances;
      if (l == 0) {
        return INSUFFICIENT_UNIQUE;
      }
    } else {
      return INSUFFICIENT_UNIQUE;
    }
  }

  std::vector<size_t> ind = minindex(d, l);

  double d_base = d[ind[0]];
  std::vector<double> w(l);

  double sumw = 0., r = 0.;
  if (opts.algorithm == "" || opts.algorithm == "simplex") {
    for (int j = 0; j < l; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = exp(-theta*pow((d[ind[j]] / d_base),(0.5))); */
      w[j] = exp(-opts.theta * sqrt(d[ind[j]] / d_base));
      sumw = sumw + w[j];
    }
    for (int j = 0; j < l; j++) {
      r = r + y[ind[j]] * (w[j] / sumw);
    }

    ystar[Mp_i] = r;
    return SUCCESS;

  } else if (opts.algorithm == "smap" || opts.algorithm == "llr") {

    Eigen::MatrixXd X_ls(l, M.cols());
    std::vector<double> y_ls(l), w_ls(l);

    double mean_w = 0.;
    for (int j = 0; j < l; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = pow(d[ind[j]],0.5); */
      w[j] = sqrt(d[ind[j]]);
      mean_w = mean_w + w[j];
    }
    mean_w = mean_w / (double)l;
    for (int j = 0; j < l; j++) {
      w[j] = exp(-opts.theta * (w[j] / mean_w));
    }

    int rowc = -1;
    for (int j = 0; j < l; j++) {
      if (y[ind[j]] == MISSING) {
        continue;
      }
      bool anyMissing = false;
      for (int i = 0; i < M.cols(); i++) {
        if (M(ind[j], i) == MISSING) {
          anyMissing = true;
          break;
        }
      }
      if (anyMissing) {
        continue;
      }
      rowc++;
      if (opts.algorithm == "llr") {
        // llr algorithm is not needed at this stage
        return NOT_IMPLEMENTED;

      } else if (opts.algorithm == "smap") {
        y_ls[rowc] = y[ind[j]] * w[j];
        w_ls[rowc] = w[j];
        for (int i = 0; i < M.cols(); i++) {
          X_ls(rowc, i) = M(ind[j], i) * w[j];
        }
      }
    }
    if (rowc == -1) {
      ystar[Mp_i] = MISSING;
      return SUCCESS;
    }

    // Pull out the first 'rowc+1' elements of the y_ls vector and
    // concatenate the column vector 'w' with 'X_ls', keeping only
    // the first 'rowc+1' rows.
    Eigen::VectorXd y_ls_cj(rowc + 1);
    Eigen::MatrixXd X_ls_cj(rowc + 1, M.cols() + 1);

    for (int i = 0; i < rowc + 1; i++) {
      y_ls_cj(i) = y_ls[i];
      X_ls_cj(i, 0) = w_ls[i];
      for (int j = 1; j < X_ls.cols() + 1; j++) {
        X_ls_cj(i, j) = X_ls(i, j - 1);
      }
    }

    if (opts.algorithm == "llr") {
      // llr algorithm is not needed at this stage
      return NOT_IMPLEMENTED;
    } else {
      Eigen::BDCSVD<Eigen::MatrixXd> svd(X_ls_cj, Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::VectorXd ics = svd.solve(y_ls_cj);

      r = ics(0);
      for (int j = 1; j < M.cols() + 1; j++) {
        if (b(j - 1) != MISSING) {
          r += b(j - 1) * ics(j);
        }
      }

      // saving ics coefficients if savesmap option enabled
      if (opts.save_mode) {
        for (int j = 0; j < opts.varssv; j++) {
          if (ics(j) == 0.) {
            (*Bi_map)(Mp_i, j) = MISSING;
          } else {
            (*Bi_map)(Mp_i, j) = ics(j);
          }
        }
      }

      ystar[Mp_i] = r;
      return SUCCESS;
    }
  }

  return INVALID_ALGORITHM;
}

static void BM_Single_Fast(benchmark::State& state)
{
  EdmInputs vars = read_dumpfile("perfinput.h5");

  // Create Eigen matrixes which are views of the supplied flattened matrices
  MatrixView M_mat((double*)vars.M.flat.data(), vars.M.rows(), vars.M.cols());     //  count_train_set, mani
  MatrixView Mp_mat((double*)vars.Mp.flat.data(), vars.Mp.rows(), vars.Mp.cols()); // count_predict_set, mani

  MatrixRowView b = ((const MatrixView)Mp_mat).row(0);

  int ind = 0;

  std::vector<double> ystar(vars.Mp.rows());

  smap_opts_t opts = { vars.opts.force_compute, vars.opts.save_mode,       vars.opts.l,        vars.opts.varssv,
                       vars.opts.theta,         vars.opts.missingdistance, vars.opts.algorithm };

  std::optional<MatrixView> bout = std::nullopt;

  for (auto _ : state) {
    auto res = fast_mf_smap_single(ind, opts, vars.y, M_mat, Mp_mat, ystar, bout);
    ind = (ind + 1) % vars.Mp.rows();
  }
}

BENCHMARK(BM_Single_Fast);

static void BM_Single(benchmark::State& state)
{
  EdmInputs vars = read_dumpfile("perfinput.h5");

  int ind = 0;

  for (auto _ : state) {
    auto res = mf_smap_single(ind, vars.opts, vars.y, vars.M, vars.Mp);
    ind = (ind + 1) % vars.Mp.rows();
  }
}

BENCHMARK(BM_Single);

std::pair<std::vector<double>, int> get_distances_eigen(const MatrixView& M, const MatrixRowView& b,
                                                        double missingdistance)
{
  int validDistances = 0;
  std::vector<double> d(M.rows());

  for (int i = 0; i < M.rows(); i++) {
    double dist = 0.;
    bool missing = false;
    int numMissingDims = 0;
    for (int j = 0; j < M.cols(); j++) {
      if ((M(i, j) == MISSING) || (b(j) == MISSING)) {
        if (missingdistance == 0) {
          missing = true;
          break;
        }
        numMissingDims += 1;
      } else {
        dist += (M(i, j) - b(j)) * (M(i, j) - b(j));
      }
    }
    // If the distance between M_i and b is 0 before handling missing values,
    // then keep it at 0. Otherwise, add in the correct number of missingdistance's.
    if (dist != 0) {
      dist += numMissingDims * missingdistance * missingdistance;
    }

    if (missing || dist == 0.) {
      d[i] = MISSING;
    } else {
      d[i] = dist;
      validDistances += 1;
    }
  }
  return { d, validDistances };
}

static void BM_Distances_Eigen(benchmark::State& state)
{
  EdmInputs vars = read_dumpfile("test1.h5");

  // Create Eigen matrixes which are views of the supplied flattened matrices
  MatrixView M_mat((double*)vars.M.flat.data(), vars.M.rows(), vars.M.cols());     //  count_train_set, mani
  MatrixView Mp_mat((double*)vars.Mp.flat.data(), vars.Mp.rows(), vars.Mp.cols()); // count_predict_set, mani

  MatrixRowView b = ((const MatrixView)Mp_mat).row(0);

  for (auto _ : state)
    auto res = get_distances_eigen(M_mat, b, vars.opts.missingdistance);
}

BENCHMARK(BM_Distances_Eigen);

static void BM_Distances(benchmark::State& state)
{
  EdmInputs vars = read_dumpfile("test1.h5");

  for (auto _ : state)
    auto res = get_distances(vars.M, vars.Mp.get_observation(0), vars.opts.missingdistance);
}

BENCHMARK(BM_Distances);

double simplex_simple(double theta, const std::vector<double>& y, const std::vector<double>& d,
                      const std::vector<size_t>& ind, int l)
{
  std::vector<double> w(l);
  double d_base = d[ind[0]];
  double sumw = 0., r = 0.;

  for (int j = 0; j < l; j++) {
    /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
    /* w[j] = exp(-theta*pow((d[ind[j]] / d_base),(0.5))); */
    w[j] = exp(-theta * sqrt(d[ind[j]] / d_base));
    sumw = sumw + w[j];
  }
  for (int j = 0; j < l; j++) {
    r = r + y[ind[j]] * (w[j] / sumw);
  }

  return r;
}

static void BM_Simplex_Simple(benchmark::State& state)
{
  EdmInputs vars = read_dumpfile("perfinput.h5");

  int ind = 0;
  for (auto _ : state) {
    auto [d, validDistances] = get_distances(vars.M, vars.Mp.get_observation(ind), vars.opts.missingdistance);
    ind = (ind + 1) % vars.Mp.rows();

    // If we only look at distances which are non-zero and non-missing,
    // do we have enough of them to find 'l' neighbours?
    int l = vars.opts.l;
    if (l > validDistances) {
      if (vars.opts.force_compute && validDistances > 0) {
        l = validDistances;
      }
    }

    std::vector<size_t> ind = minindex(d, l);

    auto res = simplex_simple(vars.opts.theta, vars.y, d, ind, l);
  }
}

BENCHMARK(BM_Simplex_Simple);

static void BM_Simplex(benchmark::State& state)
{
  EdmInputs vars = read_dumpfile("perfinput.h5");

  int ind = 0;
  for (auto _ : state) {
    auto [d, validDistances] = get_distances(vars.M, vars.Mp.get_observation(ind), vars.opts.missingdistance);
    ind = (ind + 1) % vars.Mp.rows();

    // If we only look at distances which are non-zero and non-missing,
    // do we have enough of them to find 'l' neighbours?
    int l = vars.opts.l;
    if (l > validDistances) {
      if (vars.opts.force_compute && validDistances > 0) {
        l = validDistances;
      }
    }

    std::vector<size_t> ind = minindex(d, l);

    Eigen::ArrayXd dNear(l), yNear(l);
    for (int i = 0; i < l; i++) {
      dNear[i] = d[ind[i]];
      yNear[i] = vars.y[ind[i]];
    }

    auto res = simplex(vars.opts.theta, yNear, dNear);
  }
}

BENCHMARK(BM_Simplex);

IO io = { no_display, no_display, no_flush };
// IO io = { display, error, flush };

static void BM_EDM(benchmark::State& state)
{
  EdmInputs vars = read_dumpfile("test1.h5");

  for (auto _ : state)
    EdmResult res = mf_smap_loop(vars.opts, vars.y, vars.M, vars.Mp, vars.nthreads, io);
}

BENCHMARK(BM_EDM);

static void BM_EDM_onethread(benchmark::State& state)
{
  EdmInputs vars = read_dumpfile("test1.h5");
  vars.nthreads = 1;

  for (auto _ : state)
    EdmResult res = mf_smap_loop(vars.opts, vars.y, vars.M, vars.Mp, vars.nthreads, io);
}

BENCHMARK(BM_EDM_onethread);

static void BM_PERF_EDM(benchmark::State& state)
{
  EdmInputs vars = read_dumpfile("perfinput.h5");

  for (auto _ : state)
    EdmResult res = mf_smap_loop(vars.opts, vars.y, vars.M, vars.Mp, vars.nthreads, io);
}
BENCHMARK(BM_PERF_EDM);

static void BM_PERF_EDM_threads(benchmark::State& state)
{
  EdmInputs vars = read_dumpfile("perfinput.h5");
  vars.nthreads = state.range(0);

  for (auto _ : state)
    EdmResult res = mf_smap_loop(vars.opts, vars.y, vars.M, vars.Mp, vars.nthreads, io);
}

BENCHMARK(BM_PERF_EDM_threads)->DenseRange(1, 8);

// // Define another benchmark
// static void BM_StringCopy(benchmark::State& state) {
//   std::string x = "hello";
//   for (auto _ : state)
//     std::string copy(x);
// }
// BENCHMARK(BM_StringCopy);

BENCHMARK_MAIN();