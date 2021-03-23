/*
 * Implementation of EDM methods, including S-map and cross-mapping
 *
 * - Edoardo Tescari, Melbourne Data Analytics Platform,
 *  The University of Melbourne, e.tescari@unimelb.edu.au
 * - Patrick Laub, Department of Management and Marketing,
 *   The University of Melbourne, patrick.laub@unimelb.edu.au
 */

#include "edm.h"
#include "ThreadPool.h"
#include "cpu.h"

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/SVD>
#include <algorithm> // std::partial_sort
#include <cmath>
#include <numeric> // std::iota

/* minindex(v,k) returns the indices of the k minimums of v.  */
std::vector<size_t> minindex(const std::vector<double>& v, int k)
{
  // Initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  if (k >= (int)(v.size() / 2)) {
    auto comparator = [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; };
    std::stable_sort(idx.begin(), idx.end(), comparator);
  } else {
    auto stableComparator = [&v](size_t i1, size_t i2) {
      if (v[i1] != v[i2])
        return v[i1] < v[i2];
      else
        return i1 < i2;
    };
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), stableComparator);
  }

  return idx;
}

void simplex(int Mp_i, int t, Options opts, const Manifold& M, int k, const std::vector<double>& d,
             const std::vector<size_t>& ind, span_2d_double ystar, span_2d_retcode rc);
void smap(int Mp_i, int t, Options opts, const Manifold& M, const Manifold& Mp, int k, const std::vector<double>& d,
          const std::vector<size_t>& ind, span_2d_double ystar, span_2d_double coeffs, span_2d_retcode rc);

void mf_smap_single(int Mp_i, Options opts, const Manifold& M, const Manifold& Mp, span_2d_double ystar,
                    span_2d_retcode rc, span_2d_double coeffs, int skipRow, bool keep_going() = nullptr)
{
  if (keep_going != nullptr && keep_going() == false) {
    for (int t = 0; t < opts.thetas.size(); t++) {
      rc(t, Mp_i) = UNKNOWN_ERROR;
    }
    return;
  }
  int validDistances = 0;
  std::vector<double> d(M.nobs());
  std::vector<int> numMissing(M.nobs());

  bool allowMissing = (opts.missingdistance > 0);

  const Eigen::Map<const Eigen::ArrayXd> target(Mp.obs(Mp_i), Mp.E_actual());
  const Eigen::Array<bool, Eigen::Dynamic, 1> targetValid = (target != MISSING);

  // Some easily auto-vectorizable code.
  // const auto testSetSize = 50;
  // std::vector<int> increasing(testSetSize);
  // std::iota(increasing.begin(), increasing.end(), 0);

  // int* x = increasing.data();
  // int* y = increasing.data();
  // auto result = new int[testSetSize];
  // for (auto i = 0; i < testSetSize; i++) {
  //   result[i] = x[i] + y[i];
  // }
  // delete[] result;

  for (int i = 0; i < M.nobs(); i++) {
    const Eigen::Map<const Eigen::ArrayXd> proposal(M.obs(i), M.E_actual());
    auto diff = (target - proposal).square();
    const Eigen::Array<bool, Eigen::Dynamic, 1> isValid = targetValid && (proposal != MISSING);
    d[i] = (diff * isValid.cast<double>()).sum();
    numMissing[i] = M.E_actual() - isValid.count();
  }

  if (allowMissing) {
    Eigen::Map<Eigen::ArrayXd> dView(d.data(), d.size());
    Eigen::Map<const Eigen::ArrayXi> numMissingView(numMissing.data(), numMissing.size());
    dView = dView + numMissingView.cast<double>() * opts.missingdistance * opts.missingdistance;
    validDistances = d.size();
  } else {
    for (int i = 0; i < M.nobs(); i++) {
      if (numMissing[i]) {
        d[i] = MISSING;
      } else {
        validDistances += 1;
      }
    }
  }

  // If we only look at distances which are non-zero and non-missing,
  // do we have enough of them to find k neighbours?
  int k = opts.k;
  if (k > validDistances) {
    if (opts.forceCompute) {
      k = validDistances;
      if (k == 0) {
        for (int t = 0; t < opts.thetas.size(); t++) {
          rc(t, Mp_i) = INSUFFICIENT_UNIQUE;
        }
        return;
      }
    } else {
      for (int t = 0; t < opts.thetas.size(); t++) {
        rc(t, Mp_i) = INSUFFICIENT_UNIQUE;
      }
      return;
    }
  }

  double d_min = *std::min_element(d.begin(), d.end());
  bool skipFirst = (d_min > 0) && (skipRow >= 0);

  for (int i = 0; i < d.size(); i++) {
    if (d[i] == 0) {
      d[i] = MISSING;
    }
  }

  std::vector<size_t> ind = minindex(d, k + skipFirst);

  if (skipFirst) {
    ind.erase(ind.begin(), ind.begin() + 1);
  }

  if (opts.algorithm == "" || opts.algorithm == "simplex") {
    for (int t = 0; t < opts.thetas.size(); t++) {
      simplex(Mp_i, t, opts, M, k, d, ind, ystar, rc);
    }
  } else if (opts.algorithm == "smap" || opts.algorithm == "llr") {
    bool saveCoeffsForLargestTheta = opts.saveSMAPCoeffs;
    opts.saveSMAPCoeffs = false;
    for (int t = 0; t < opts.thetas.size(); t++) {
      if (t == opts.thetas.size() - 1) {
        opts.saveSMAPCoeffs = saveCoeffsForLargestTheta;
      }
      smap(Mp_i, t, opts, M, Mp, k, d, ind, ystar, coeffs, rc);
    }
  } else {
    for (int t = 0; t < opts.thetas.size(); t++) {
      rc(t, Mp_i) = INVALID_ALGORITHM;
    }
  }
}

void simplex(int Mp_i, int t, Options opts, const Manifold& M, int k, const std::vector<double>& d,
             const std::vector<size_t>& ind, span_2d_double ystar, span_2d_retcode rc)
{
  double theta = opts.thetas[t];

  double d_base = d[ind[0]];
  std::vector<double> w(k);
  double sumw = 0., r = 0.;

  for (int j = 0; j < k; j++) {
    if (d[ind[j]] != MISSING) {
      w[j] = exp(-theta * (d[ind[j]] / d_base));
    } else {
      w[j] = 0;
    }
    sumw = sumw + w[j];
  }

  for (int j = 0; j < k; j++) {
    r = r + M.y(ind[j]) * (w[j] / sumw);
  }

  ystar(t, Mp_i) = r;
  rc(t, Mp_i) = SUCCESS;
}

void smap(int Mp_i, int t, Options opts, const Manifold& M, const Manifold& Mp, int k, const std::vector<double>& d,
          const std::vector<size_t>& ind, span_2d_double ystar, span_2d_double coeffs, span_2d_retcode rc)
{
  double d_base = d[ind[0]];
  std::vector<double> w(k);

  Eigen::MatrixXd X_ls(k, M.E_actual());
  std::vector<double> y_ls(k), w_ls(k);

  double mean_w = 0.;
  int kValid = 0;
  for (int j = 0; j < k; j++) {
    if (d[ind[j]] != MISSING) {
      mean_w = mean_w + d[ind[j]];
      kValid += 1;
    }
  }
  mean_w = mean_w / (double)kValid;

  double theta = opts.thetas[t];

  // Need to check for missing values because e^(-theta*w[j]) = e^(-0 * MISSING) = 1
  // will still gives weight to missing values.
  for (int j = 0; j < k; j++) {
    if (d[ind[j]] != MISSING) {
      w[j] = exp(-theta * (d[ind[j]] / mean_w));
    } else {
      w[j] = 0;
    }
  }

  int rowc = -1;
  for (int j = 0; j < k; j++) {
    if (M.any_missing(ind[j])) {
      continue;
    }
    rowc++;
    if (opts.algorithm == "llr") {
      // llr algorithm is not needed at this stage
      rc(t, Mp_i) = NOT_IMPLEMENTED;
      return;

    } else if (opts.algorithm == "smap") {
      y_ls[rowc] = M.y(ind[j]) * w[j];
      w_ls[rowc] = w[j];
      for (int i = 0; i < M.E_actual(); i++) {
        X_ls(rowc, i) = M(ind[j], i) * w[j];
      }
    }
  }
  if (rowc == -1) {
    ystar(t, Mp_i) = MISSING;
    rc(t, Mp_i) = SUCCESS;
    return;
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

  if (opts.algorithm == "llr") {
    // llr algorithm is not needed at this stage
    rc(t, Mp_i) = NOT_IMPLEMENTED;
  } else {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(X_ls_cj, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd ics = svd.solve(y_ls_cj);

    double r = ics(0);
    for (int j = 1; j < M.E_actual() + 1; j++) {
      if (Mp(Mp_i, j - 1) != MISSING) {
        r += Mp(Mp_i, j - 1) * ics(j);
      }
    }

    // saving ics coefficients if savesmap option enabled
    if (opts.saveSMAPCoeffs) {
      for (int j = 0; j < M.E_actual() + 1; j++) {
        if (ics(j) == 0.) {
          coeffs(Mp_i, j) = MISSING;
        } else {
          coeffs(Mp_i, j) = ics(j);
        }
      }
    }

    ystar(t, Mp_i) = r;
    rc(t, Mp_i) = SUCCESS;
  }
}

std::atomic<int> numTasksRunning = 0;
ThreadPool workerPool, masterPool;

std::future<void> edm_async(Options opts, const ManifoldGenerator* generator, size_t E, std::vector<bool> trainingRows,
                            std::vector<bool> predictionRows, IO* io, Prediction* pred, bool keep_going(),
                            void all_tasks_finished(void))
{
  bool serial;
  if (opts.parMode == -1) {
    serial = (opts.numTasks > opts.nthreads);
  } else {
    serial = opts.parMode;
  }

  workerPool.set_num_workers(opts.nthreads);
  if (!serial) {
    masterPool.set_num_workers(opts.nthreads);
  }

  if (numTasksRunning == 0) {
    numTasksRunning = (int)opts.numTasks;
  }

  if (serial) {
    return workerPool.enqueue(
      [opts, generator, E, trainingRows, predictionRows, io, pred, keep_going, all_tasks_finished, serial] {
        edm_task(opts, generator, E, trainingRows, predictionRows, io, pred, keep_going, all_tasks_finished, serial);
      });
  } else {
    return masterPool.enqueue(
      [opts, generator, E, trainingRows, predictionRows, io, pred, keep_going, all_tasks_finished, serial] {
        edm_task(opts, generator, E, trainingRows, predictionRows, io, pred, keep_going, all_tasks_finished, serial);
      });
  }
}

// Don't call this directly. The thread pools won't be setup correctly.
void edm_task(Options opts, const ManifoldGenerator* generator, size_t E, std::vector<bool> trainingRows,
              std::vector<bool> predictionRows, IO* io, Prediction* pred, bool keep_going(),
              void all_tasks_finished(void), bool serial)
{
  Manifold M, Mp;
  if (serial) {
    M = generator->create_manifold(E, trainingRows, false);
    Mp = generator->create_manifold(E, predictionRows, true);
  } else {
    std::future<void> f1 = workerPool.enqueue([&] { M = generator->create_manifold(E, trainingRows, false); });
    std::future<void> f2 = workerPool.enqueue([&] { Mp = generator->create_manifold(E, predictionRows, true); });
    f1.get();
    f2.get();
  }

  size_t numThetas = opts.thetas.size();
  size_t numPredictions = Mp.nobs();
  size_t numCoeffCols = M.E_actual() + 1;

  // If the same observation is in the training & prediction sets,
  // then find the row index of the train manifold for a given prediction row.
  std::vector<int> predToTrainSelfMap(numPredictions);
  int M_i = 0, Mp_i = 0;
  int numSelfToSkip = 0;
  for (int r = 0; r < trainingRows.size(); r++) {
    if (predictionRows[r]) {
      if (trainingRows[r] && !opts.copredict) {
        predToTrainSelfMap[Mp_i] = M_i;
        numSelfToSkip += 1;
      } else {
        predToTrainSelfMap[Mp_i] = -1;
      }
    }

    M_i += trainingRows[r];
    Mp_i += predictionRows[r];
  }

  auto ystar = std::make_unique<double[]>(numThetas * numPredictions);
  auto ystarView = span_2d_double(ystar.get(), (int)numThetas, (int)numPredictions);

  // If we're saving the coefficients (i.e. in xmap mode), then we're not running with multiple 'theta' values.
  auto coeffs = std::make_unique<double[]>(numPredictions * numCoeffCols);
  auto coeffsView = span_2d_double(coeffs.get(), (int)numPredictions, (int)numCoeffCols);
  std::fill_n(coeffs.get(), numPredictions * numCoeffCols, MISSING);

  auto rc = std::make_unique<retcode[]>(numThetas * numPredictions);
  auto rcView = span_2d_retcode(rc.get(), (int)numThetas, (int)numPredictions);

  if (opts.numTasks > 1 && opts.taskNum == 0) {
    io->progress_bar(0.0);
  }

  if (serial) {
    if (opts.numTasks == 1) {
      io->progress_bar(0.0);
    }
    for (int i = 0; i < numPredictions; i++) {
      if (keep_going != nullptr && keep_going() == false) {
        *pred = { UNKNOWN_ERROR, {}, {} };
      }
      mf_smap_single(i, opts, M, Mp, ystarView, rcView, coeffsView, predToTrainSelfMap[i], keep_going);
      if (opts.numTasks == 1) {
        io->progress_bar((i + 1) / ((double)numPredictions));
      }
    }
  } else {
    if (opts.distributeThreads) {
      distribute_threads(workerPool.workers);
    }

    std::vector<std::future<void>> results(numPredictions);
    for (int i = 0; i < numPredictions; i++) {
      results[i] = workerPool.enqueue(
        [&, i] { mf_smap_single(i, opts, M, Mp, ystarView, rcView, coeffsView, predToTrainSelfMap[i], keep_going); });
    }

    if (opts.numTasks == 1) {
      io->progress_bar(0.0);
    }
    for (int i = 0; i < numPredictions; i++) {
      results[i].get();
      if (opts.numTasks == 1) {
        io->progress_bar((i + 1) / ((double)numPredictions));
      }
    }

    if (keep_going != nullptr && keep_going() == false) {
      *pred = { UNKNOWN_ERROR, {}, {} };
    }
  }

  // Calculate the MAE & rho of prediction, if requested
  PredictionStats stats;
  stats.mae = MISSING;
  stats.rho = MISSING;

  if (opts.calcRhoMAE) {
    std::vector<double> y1, y2;
    for (int i = 0; i < Mp.ySize(); i++) {
      if (Mp.y(i) != MISSING && ystar[i] != MISSING) {
        y1.push_back(Mp.y(i));
        y2.push_back(ystar[i]);
      }
    }

    Eigen::Map<const Eigen::ArrayXd> y1Map(y1.data(), y1.size());
    Eigen::Map<const Eigen::ArrayXd> y2Map(y2.data(), y2.size());

    const Eigen::ArrayXd y1Cent = y1Map - y1Map.mean();
    const Eigen::ArrayXd y2Cent = y2Map - y2Map.mean();

    stats.mae = (y1Map - y2Map).abs().mean();
    stats.rho = (y1Cent * y2Cent).sum() / (std::sqrt((y1Cent * y1Cent).sum()) * std::sqrt((y2Cent * y2Cent).sum()));
  }

  stats.taskNum = (int)opts.taskNum;
  stats.calcRhoMAE = opts.calcRhoMAE;
  pred->stats = stats;

  // Check if any mf_smap_single call failed, and if so find the most serious error
  pred->rc = *std::max_element(rc.get(), rc.get() + numThetas * numPredictions);

  // If we're storing the prediction and/or the SMAP coefficients, put them
  // into the resulting Prediction struct. Otherwise, let them be deleted.
  if (opts.savePrediction) {
    // Take only the predictions for the largest theta value.
    if (numThetas == 1) {
      pred->ystar = std::move(ystar);
    } else {
      pred->ystar = std::make_unique<double[]>(numPredictions);
      for (int i = 0; i < numPredictions; i++) {
        pred->ystar[i] = ystarView(numThetas - 1, i);
      }
    }
  } else {
    pred->ystar = nullptr;
  }

  if (opts.saveSMAPCoeffs) {
    pred->coeffs = std::move(coeffs);
  } else {
    pred->coeffs = nullptr;
  }

  if (opts.savePrediction || opts.saveSMAPCoeffs) {
    pred->predictionRows = predictionRows;
  }

  pred->numThetas = numThetas;
  pred->numPredictions = numPredictions;
  pred->numCoeffCols = numCoeffCols;

  numTasksRunning -= 1;
  if (opts.numTasks > 1) {
    io->progress_bar((opts.numTasks - numTasksRunning) / ((double)opts.numTasks));
  }

  if (numTasksRunning <= 0) {
    if (all_tasks_finished != nullptr) {
      all_tasks_finished();
    }
    numTasksRunning = 0;
  }
}
