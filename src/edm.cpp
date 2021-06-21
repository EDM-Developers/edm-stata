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

// For a given point, find the k nearest neighbours of this point.
//
// We are given the distances to all potential neighbours as a vector.
// We return the 'k' indices of the neighbours.
//
// The distance vector may contain missing values which are coded as a massive
// double value with the MISSING macro; this should be enough to put those
// neighbours last in after sorting by closeness.
//
// If there are many potential neighbours with the exact same distances, we
// prefer the neighbours with the smallest index value. This corresponds
// to a stable sort in C++ STL terminology.
//
// In typical use-cases of 'edm explore' the value of 'k' is small, like 5-20.
// However for a typical 'edm xmap' the value of 'k' is set as large as possible.
// If 'k' is small, the partial_sort is efficient as it only finds the 'k' smallest
// distances. If 'k' is larger, then it is faster to simply sort the entire distance
// vector.
//
// N.B. The equivalent function in Stata/Mata is called 'minindex'.
std::vector<int> kNearestNeighboursIndices(const std::vector<double>& dists, int k)
{
  // Initialize original index locations
  std::vector<int> idx(dists.size());
  std::iota(idx.begin(), idx.end(), 0);

  if (k >= (int)(dists.size() / 2)) {
    auto comparator = [&dists](int i1, int i2) { return dists[i1] < dists[i2]; };
    std::stable_sort(idx.begin(), idx.end(), comparator);
  } else {
    auto stableComparator = [&dists](int i1, int i2) {
      if (dists[i1] != dists[i2])
        return dists[i1] < dists[i2];
      else
        return i1 < i2;
    };
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), stableComparator);
  }

  return idx;
}

void simplex_prediction(int Mp_i, int t, Options opts, const Manifold& M, int k, const std::vector<double>& d,
                        const std::vector<int>& ind, span_2d_double ystar, span_2d_retcode rc);
void smap_prediction(int Mp_i, int t, Options opts, const Manifold& M, const Manifold& Mp, int k,
                     const std::vector<double>& d, const std::vector<int>& ind, span_2d_double ystar,
                     span_2d_double coeffs, span_2d_retcode rc);

// Use a training manifold 'M' to make a prediction about the prediction manifold 'Mp'.
// Specifically, predict the 'Mp_i'-th value of the prediction manifold 'Mp'.
//
// The predicted value is stored in 'ystar', along with any return codes in 'rc'.
// Optionally, the user may ask to store some S-map intermediate values in 'coeffs'.
//
// The 'opts' value specifies the kind of prediction to make (e.g. S-map, or simplex method).
// This function is usually run in a worker thread, and the 'keep_going' callback is frequently called to
// see whether the user still wants this result, or if they have given up & simply want the execution
// to terminate.
//
// We sometimes let 'M' and 'Mp' be the same manifold, so we train and predict using the same values.
// In this case, the algorithm may cheat by pulling out the identical trajectory from the training manifold
// and using this as the prediction. The 'skipRow' variable, when positive, indicates that we skip the
// closest neighbour as it is probably cheating. 'skipRow' actually stores the specific index
// of the training manifold which should not be used for this prediction, and in the future we will
// use this directly, though some details need to be worked out.
void make_prediction(int Mp_i, Options opts, const Manifold& M, const Manifold& Mp, span_2d_double ystar,
                     span_2d_retcode rc, span_2d_double coeffs, int skipRow, bool keep_going() = nullptr)
{
  if (keep_going != nullptr && keep_going() == false) {
    return;
  }
  int validDistances = 0;
  std::vector<double> dists(M.nobs());

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
    dist += numMissingDims * opts.missingdistance * opts.missingdistance;

    if (!missing) {
      dists[i] = sqrt(dist);
      validDistances += 1;
    } else {
      dists[i] = MISSING;
    }
  }

  if (keep_going != nullptr && keep_going() == false) {
    return;
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

  double minDist = *std::min_element(dists.begin(), dists.end());
  bool skipFirst = (minDist > 0) && (skipRow >= 0);

  for (int i = 0; i < dists.size(); i++) {
    if (dists[i] == 0) {
      dists[i] = MISSING;
    }
  }

  std::vector<int> kNNInds = kNearestNeighboursIndices(dists, k + skipFirst);

  if (skipFirst) {
    kNNInds.erase(kNNInds.begin(), kNNInds.begin() + 1);
  }

  if (keep_going != nullptr && keep_going() == false) {
    return;
  }

  if (opts.algorithm == "" || opts.algorithm == "simplex") {
    for (int t = 0; t < opts.thetas.size(); t++) {
      simplex_prediction(Mp_i, t, opts, M, k, dists, kNNInds, ystar, rc);
    }
  } else if (opts.algorithm == "smap" || opts.algorithm == "llr") {
    bool saveCoeffsForLargestTheta = opts.saveSMAPCoeffs;
    opts.saveSMAPCoeffs = false;
    for (int t = 0; t < opts.thetas.size(); t++) {
      if (t == opts.thetas.size() - 1) {
        opts.saveSMAPCoeffs = saveCoeffsForLargestTheta;
      }
      smap_prediction(Mp_i, t, opts, M, Mp, k, dists, kNNInds, ystar, coeffs, rc);
    }
  } else {
    for (int t = 0; t < opts.thetas.size(); t++) {
      rc(t, Mp_i) = INVALID_ALGORITHM;
    }
  }
}

void simplex_prediction(int Mp_i, int t, Options opts, const Manifold& M, int k, const std::vector<double>& dists,
                        const std::vector<int>& kNNInds, span_2d_double ystar, span_2d_retcode rc)
{
  double theta = opts.thetas[t];

  double d_base = dists[kNNInds[0]];
  std::vector<double> w(k);
  double sumw = 0., r = 0.;

  for (int j = 0; j < k; j++) {
    if (dists[kNNInds[j]] != MISSING) {
      w[j] = exp(-theta * (dists[kNNInds[j]] / d_base));
    } else {
      w[j] = 0;
    }
    sumw = sumw + w[j];
  }

  for (int j = 0; j < k; j++) {
    r = r + M.y(kNNInds[j]) * (w[j] / sumw);
  }

  ystar(t, Mp_i) = r;
  rc(t, Mp_i) = SUCCESS;
}

void smap_prediction(int Mp_i, int t, Options opts, const Manifold& M, const Manifold& Mp, int k,
                     const std::vector<double>& dists, const std::vector<int>& kNNInds, span_2d_double ystar,
                     span_2d_double coeffs, span_2d_retcode rc)
{
  std::vector<double> w(k);
  Eigen::MatrixXd X_ls(k, M.E_actual());
  std::vector<double> y_ls(k), w_ls(k);

  double mean_d = 0.;
  int kValid = 0;
  for (int j = 0; j < k; j++) {
    if (dists[kNNInds[j]] != MISSING) {
      mean_d = mean_d + dists[kNNInds[j]];
      kValid += 1;
    }
  }
  mean_d = mean_d / (double)kValid;

  double theta = opts.thetas[t];

  // Need to check for missing values because e^(-theta*w[j]) = e^(-0 * MISSING) = 1
  // will still gives weight to missing values.
  for (int j = 0; j < k; j++) {
    if (dists[kNNInds[j]] != MISSING) {
      w[j] = exp(-theta * (dists[kNNInds[j]] / mean_d));
    } else {
      w[j] = 0;
    }
  }

  int rowc = -1;
  for (int j = 0; j < k; j++) {
    if (M.any_missing(kNNInds[j])) {
      continue;
    }
    rowc++;
    if (opts.algorithm == "llr") {
      // llr algorithm is not needed at this stage
      rc(t, Mp_i) = NOT_IMPLEMENTED;
      return;

    } else if (opts.algorithm == "smap") {
      y_ls[rowc] = M.y(kNNInds[j]) * w[j];
      w_ls[rowc] = w[j];
      for (int i = 0; i < M.E_actual(); i++) {
        X_ls(rowc, i) = M(kNNInds[j], i) * w[j];
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

    // The pseudo-inverse of X can be calculated as (X^T * X)^(-1) * X^T
    // see https://scicomp.stackexchange.com/a/33375
    // TODO: Test whether this provides a faster and more reliable way to solve this system.
    // Eigen::BDCSVD<Eigen::MatrixXd> svd(X_ls_cj.transpose() * X_ls_cj, Eigen::ComputeThinU | Eigen::ComputeThinV);
    // Eigen::VectorXd ics = svd.solve(X_ls_cj.transpose() * y_ls_cj);

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

// If the same observation is in the training & prediction sets,
// then find the row index of the train manifold for a given prediction row.
std::vector<int> find_overlaps(std::vector<bool>& trainingRows, std::vector<bool>& predictionRows, int numPredictions,
                               bool copredict)
{

  std::vector<int> predToTrainSelfMap(numPredictions);
  int M_i = 0, Mp_i = 0;
  int numSelfToSkip = 0;
  for (int r = 0; r < trainingRows.size(); r++) {
    if (predictionRows[r]) {
      if (trainingRows[r] && !copredict) {
        predToTrainSelfMap[Mp_i] = M_i;
        numSelfToSkip += 1;
      } else {
        predToTrainSelfMap[Mp_i] = -1;
      }
    }

    M_i += trainingRows[r];
    Mp_i += predictionRows[r];
  }

  return predToTrainSelfMap;
}

std::atomic<int> numTasksStarted = 0;
std::atomic<int> numTasksFinished = 0;
ThreadPool workerPool, masterPool;

std::future<void> edm_async(Options opts, const ManifoldGenerator* generator, int E, std::vector<bool> trainingRows,
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

  if (opts.taskNum == 0) {
    numTasksStarted = 0;
    numTasksFinished = 0;
  }

  numTasksStarted += 1;

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
void edm_task(Options opts, const ManifoldGenerator* generator, int E, std::vector<bool> trainingRows,
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

  int numThetas = (int)opts.thetas.size();
  int numPredictions = Mp.nobs();
  int numCoeffCols = M.E_actual() + 1;

  auto predToTrainSelfMap = find_overlaps(trainingRows, predictionRows, numPredictions, opts.copredict);

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
        break;
      }
      make_prediction(i, opts, M, Mp, ystarView, rcView, coeffsView, predToTrainSelfMap[i], keep_going);
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
        [&, i] { make_prediction(i, opts, M, Mp, ystarView, rcView, coeffsView, predToTrainSelfMap[i], keep_going); });
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
  }

  // Store the results, so long as we weren't interrupted by a 'break'.
  if (keep_going != nullptr && keep_going() == true) {

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

    if (opts.numTasks > 1) {
      io->progress_bar((numTasksFinished + 1) / ((double)opts.numTasks));
    }
  }

  numTasksFinished += 1;

  if (numTasksFinished == opts.numTasks) {
    if (all_tasks_finished != nullptr) {
      all_tasks_finished();
    }
  }
}
