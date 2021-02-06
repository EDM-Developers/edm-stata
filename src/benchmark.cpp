#include <benchmark/benchmark.h>

#ifdef _WIN32
#pragma comment(lib, "Shlwapi.lib")
#ifdef _DEBUG
#pragma comment(lib, "benchmarkd.lib")
#else
#pragma comment(lib, "benchmark.lib")
#endif
#endif

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include "cpu.h"
#include "driver.h"
#include "edm.h"

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/SVD>
typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixView;

std::vector<size_t> minindex(const std::vector<double>& v, int k);

// Compiler flags tried on Windows: "/GL" and "/GL /LTCG", both slightly worse. "/O2" is the default.

// Inputs generated by 'perf-test.do' script
std::vector<std::string> tests = {
  "logmapsmall.h5", // "edm explore x, e(10)" on 200 obs of logistic map
  "logmaplarge.h5", // "edm xmap x y, theta(0.2) algorithm(smap)" on ~50k obs of logistic map
  "affectsmall.h5", // "edm xmap PA NA, dt e(10) k(-1) force alg(smap)" on ~5k obs of affect data
  "affectbige.h5"   // "edm xmap PA NA, dt e(150) k(20) force alg(smap)" on ~5k obs of affect data
};

ConsoleIO io(0);

static void bm_sqrt(benchmark::State& state)
{
  state.SetLabel("'sqrt' function");

  double i = 0.0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::sqrt(i));
    i += 1.0;
  }
}

BENCHMARK(bm_sqrt);

static void bm_pow_half(benchmark::State& state)
{
  state.SetLabel("'pow(., 0.5)' function");

  double i = 0.0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(std::pow(i, 0.5));
    i += 1.0;
  }
}

BENCHMARK(bm_pow_half);

void get_distances(int Mp_i, Options opts, const Manifold& M, const Manifold& Mp)
{
  int validDistances = 0;
  std::vector<double> d(M.nobs());

  for (int i = 0; i < M.nobs(); i++) {
    double dist = 0.;
    bool missing = false;
    int numMissingDims = 0;
    for (int j = 0; j < M.E_actual(); j++) {
      if ((M(i, j) == MISSING) || (Mp(Mp_i, j) == MISSING)) {
        if (opts.missingdistance == 0) {
          missing = true;
          break;
        }
        numMissingDims += 1;
      } else {
        dist += (M(i, j) - Mp(Mp_i, j)) * (M(i, j) - Mp(Mp_i, j));
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
}

static void bm_get_distances(benchmark::State& state)
{
  std::string input = tests[state.range(0)];
  state.SetLabel(input);

  Inputs vars = read_dumpfile(input);

  Manifold M = vars.generator.create_manifold(vars.E, vars.trainingRows, false);
  Manifold Mp = vars.generator.create_manifold(vars.E, vars.predictionRows, true);

  int Mp_i = 0;
  for (auto _ : state) {
    get_distances(Mp_i, vars.opts, M, Mp);
    Mp_i = (Mp_i + 1) % Mp.nobs();
  }
}

BENCHMARK(bm_get_distances)->DenseRange(0, tests.size() - 1)->Unit(benchmark::kMicrosecond);

static void bm_nearest_neighbours(benchmark::State& state)
{
  std::string input = tests[state.range(0)];
  state.SetLabel(input);

  Inputs vars = read_dumpfile(input);

  Manifold M = vars.generator.create_manifold(vars.E, vars.trainingRows, false);
  Manifold Mp = vars.generator.create_manifold(vars.E, vars.predictionRows, true);

  int Mp_i = 0;
  Options opts = vars.opts;

  int validDistances = 0;
  std::vector<double> d(M.nobs());

  for (int i = 0; i < M.nobs(); i++) {
    double dist = 0.;
    bool missing = false;
    int numMissingDims = 0;
    for (int j = 0; j < M.E_actual(); j++) {
      if ((M(i, j) == MISSING) || (Mp(Mp_i, j) == MISSING)) {
        if (opts.missingdistance == 0) {
          missing = true;
          break;
        }
        numMissingDims += 1;
      } else {
        dist += (M(i, j) - Mp(Mp_i, j)) * (M(i, j) - Mp(Mp_i, j));
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

  int k = opts.k;

  for (auto _ : state) {
    std::vector<size_t> ind = minindex(d, k);
  }
}

BENCHMARK(bm_nearest_neighbours)->DenseRange(0, tests.size() - 1)->Unit(benchmark::kMicrosecond);

static void bm_simplex(benchmark::State& state)
{
  std::string input = tests[state.range(0)];
  state.SetLabel(input);

  Inputs vars = read_dumpfile(input);

  Manifold M = vars.generator.create_manifold(vars.E, vars.trainingRows, false);
  Manifold Mp = vars.generator.create_manifold(vars.E, vars.predictionRows, true);

  int Mp_i = 0;
  Options opts = vars.opts;

  int validDistances = 0;
  std::vector<double> d(M.nobs());

  for (int i = 0; i < M.nobs(); i++) {
    double dist = 0.;
    bool missing = false;
    int numMissingDims = 0;
    for (int j = 0; j < M.E_actual(); j++) {
      if ((M(i, j) == MISSING) || (Mp(Mp_i, j) == MISSING)) {
        if (opts.missingdistance == 0) {
          missing = true;
          break;
        }
        numMissingDims += 1;
      } else {
        dist += (M(i, j) - Mp(Mp_i, j)) * (M(i, j) - Mp(Mp_i, j));
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

  int k = opts.k;

  std::vector<size_t> ind = minindex(d, k);

  double d_base = d[ind[0]];
  std::vector<double> w(k);

  double sumw = 0., r = 0.;

  for (auto _ : state) {
    for (int j = 0; j < k; j++) {
      w[j] = exp(-opts.thetas[0] * sqrt(d[ind[j]] / d_base));
      sumw = sumw + w[j];
    }
    for (int j = 0; j < k; j++) {
      r = r + M.y(ind[j]) * (w[j] / sumw);
    }
  }
}

BENCHMARK(bm_simplex)->DenseRange(0, tests.size() - 1);

static void bm_smap(benchmark::State& state)
{
  std::string input = tests[state.range(0)];
  state.SetLabel(input);

  Inputs vars = read_dumpfile(input);

  Manifold M = vars.generator.create_manifold(vars.E, vars.trainingRows, false);
  Manifold Mp = vars.generator.create_manifold(vars.E, vars.predictionRows, true);

  int Mp_i = 0;
  Options opts = vars.opts;

  int validDistances = 0;
  std::vector<double> d(M.nobs());

  for (int i = 0; i < M.nobs(); i++) {
    double dist = 0.;
    bool missing = false;
    int numMissingDims = 0;
    for (int j = 0; j < M.E_actual(); j++) {
      if ((M(i, j) == MISSING) || (Mp(Mp_i, j) == MISSING)) {
        if (opts.missingdistance == 0) {
          missing = true;
          break;
        }
        numMissingDims += 1;
      } else {
        dist += (M(i, j) - Mp(Mp_i, j)) * (M(i, j) - Mp(Mp_i, j));
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

  int k = opts.k;

  std::vector<size_t> ind = minindex(d, k);

  double d_base = d[ind[0]];
  std::vector<double> w(k);
  double sumw = 0., r = 0.;

  for (auto _ : state) {
    Eigen::MatrixXd X_ls(k, M.E_actual());
    std::vector<double> y_ls(k), w_ls(k);

    double mean_w = 0.;
    for (int j = 0; j < k; j++) {
      w[j] = sqrt(d[ind[j]]);
      mean_w = mean_w + w[j];
    }
    mean_w = mean_w / (double)k;
    for (int j = 0; j < k; j++) {
      w[j] = exp(-opts.thetas[0] * (w[j] / mean_w));
    }

    int rowc = -1;
    for (int j = 0; j < k; j++) {
      if (M.y(ind[j]) == MISSING) {
        continue;
      }
      bool anyMissing = false;
      for (int i = 0; i < M.E_actual(); i++) {
        if (M(ind[j], i) == MISSING) {
          anyMissing = true;
          break;
        }
      }
      if (anyMissing) {
        continue;
      }
      rowc++;

      y_ls[rowc] = M.y(ind[j]) * w[j];
      w_ls[rowc] = w[j];
      for (int i = 0; i < M.E_actual(); i++) {
        X_ls(rowc, i) = M(ind[j], i) * w[j];
      }
    }
    if (rowc == -1) {
      continue;
    }

    // Pull out the first 'rowc+1' elements of the y_ls vector and
    // concatenate the column vector 'w' with 'X_ls', keeping only
    // the first 'rowc+1' rows.
    Eigen::VectorXd y_ls_cj(rowc + 1);
    Eigen::MatrixXd X_ls_cj(rowc + 1, M.E_actual() + 1);

    for (int i = 0; i < rowc + 1; i++) {
      y_ls_cj(i) = y_ls[i];
      X_ls_cj(i, 0) = w_ls[i];
      for (int j = 1; j < X_ls.cols() + 1; j++) {
        X_ls_cj(i, j) = X_ls(i, j - 1);
      }
    }

    Eigen::BDCSVD<Eigen::MatrixXd> svd(X_ls_cj, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd ics = svd.solve(y_ls_cj);

    r = ics(0);
    for (int j = 1; j < M.E_actual() + 1; j++) {
      if (Mp(Mp_i, j - 1) != MISSING) {
        r += Mp(Mp_i, j - 1) * ics(j);
      }
    }
  }
}

BENCHMARK(bm_smap)->DenseRange(0, tests.size() - 1);

std::vector<int> get_thread_range()
{
  return { 1, 2, 4, 8 };
}

std::vector<int> threadRange = get_thread_range();

static void bm_edm_task(benchmark::State& state)
{
  int testNum = ((int)state.range(0)) / ((int)threadRange.size());
  int threads = threadRange[state.range(0) % threadRange.size()];

  std::string input = tests[testNum];
  state.SetLabel(fmt::format("{} ({} threads)", input, threads));

  Inputs vars = read_dumpfile(input);

  vars.opts.distributeThreads = false;
  vars.opts.nthreads = threads;

  Prediction pred;

  for (auto _ : state) {
    edm_async(vars.opts, &vars.generator, vars.E, vars.trainingRows, vars.predictionRows, &io, &pred).get();
  }
}

BENCHMARK(bm_edm_task)
  ->DenseRange(0, tests.size() * threadRange.size() - 1)
  ->MeasureProcessCPUTime()
  ->Unit(benchmark::kMillisecond);

BENCHMARK(bm_edm_task)
  ->DenseRange(0, tests.size() * threadRange.size() - 1)
  ->MeasureProcessCPUTime()
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();