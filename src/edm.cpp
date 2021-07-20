/*
 * Implementation of EDM methods, including S-map and cross-mapping
 *
 * - Edoardo Tescari, Melbourne Data Analytics Platform,
 *  The University of Melbourne, e.tescari@unimelb.edu.au
 * - Patrick Laub, Department of Management and Marketing,
 *   The University of Melbourne, patrick.laub@unimelb.edu.au
 */

#pragma warning(disable : 4018)

#include "edm.h"
#include "cpu.h"
#include "distances.h"
#include "stats.h" // for correlation and mean_absolute_error
#include "thread_pool.h"

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

  if (k >= dists.size()) {
    return idx;
  }

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

void simplex_prediction(int Mp_i, int t, const Options& opts, const Manifold& M, int k, const std::vector<double>& d,
                        const std::vector<int>& kNNInds, Eigen::Map<Eigen::MatrixXd> ystar,
                        Eigen::Map<Eigen::MatrixXi> rc, int* kUsed);
void smap_prediction(int Mp_i, int t, const Options& opts, const Manifold& M, const Manifold& Mp, int k,
                     const std::vector<double>& d, std::vector<int>& kNNInds, Eigen::Map<Eigen::MatrixXd> ystar,
                     Eigen::Map<Eigen::MatrixXd> coeffs, Eigen::Map<Eigen::MatrixXi> rc, int* kUsed);

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
void make_prediction(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp,
                     Eigen::Map<Eigen::MatrixXd> ystar, Eigen::Map<Eigen::MatrixXi> rc,
                     Eigen::Map<Eigen::MatrixXd> coeffs, int* kUsed, int skipRow, bool keep_going() = nullptr)
{
  if (keep_going != nullptr && keep_going() == false) {
    return;
  }

  int validDistances;
  std::vector<double> dists;

  if (opts.distance == Distance::Wasserstein) {
    dists = wasserstein_distances(Mp_i, opts, M, Mp, validDistances);
  } else {
    dists = other_distances(Mp_i, opts, M, Mp, validDistances);
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

  for (double& dist : dists) {
    if (dist == 0) {
      dist = MISSING;
    }
  }

  std::vector<int> kNNInds = kNearestNeighboursIndices(dists, k + skipFirst);

  if (skipFirst) {
    kNNInds.erase(kNNInds.begin(), kNNInds.begin() + 1);
  }

  if (keep_going != nullptr && keep_going() == false) {
    return;
  }

  // Remove the indices for the points which are not in the k nearest neighbour set.
  // After this line, the 'assert(kNNInds.size() == k)' should be true.
  kNNInds.erase(kNNInds.begin() + k, kNNInds.end());

  if (opts.algorithm == "" || opts.algorithm == "simplex") {
    for (int t = 0; t < opts.thetas.size(); t++) {
      simplex_prediction(Mp_i, t, opts, M, k, dists, kNNInds, ystar, rc, kUsed);
    }
  } else if (opts.algorithm == "smap" || opts.algorithm == "llr") {
    for (int t = 0; t < opts.thetas.size(); t++) {
      smap_prediction(Mp_i, t, opts, M, Mp, k, dists, kNNInds, ystar, coeffs, rc, kUsed);
    }
  } else {
    for (int t = 0; t < opts.thetas.size(); t++) {
      rc(t, Mp_i) = INVALID_ALGORITHM;
    }
  }
}

void simplex_prediction(int Mp_i, int t, const Options& opts, const Manifold& M, int k,
                        const std::vector<double>& dists, const std::vector<int>& kNNInds,
                        Eigen::Map<Eigen::MatrixXd> ystar, Eigen::Map<Eigen::MatrixXi> rc, int* kUsed)
{
  *kUsed = k;

  // Find the smallest distance (closest neighbour) among the supplied neighbours.
  double d_base = MISSING;
  for (int j = 0; j < k; j++) {
    if (dists[kNNInds[j]] < d_base) {
      d_base = dists[kNNInds[j]];
    }
  }

  // Calculate our weighting of each neighbour, and the total sum of these weights.
  std::vector<double> w(k);
  double sumw = 0.0;
  const double theta = opts.thetas[t];

  for (int j = 0; j < k; j++) {
    if (dists[kNNInds[j]] != MISSING) {
      w[j] = exp(-theta * (dists[kNNInds[j]] / d_base));
    } else {
      w[j] = 0;
    }
    sumw = sumw + w[j];
  }

  // Make the simplex projection/prediction.
  double r = 0.0;
  for (int j = 0; j < k; j++) {
    r = r + M.y(kNNInds[j]) * (w[j] / sumw);
  }

  // Store the results & return value.
  ystar(t, Mp_i) = r;
  rc(t, Mp_i) = SUCCESS;
}

void smap_prediction(int Mp_i, int t, const Options& opts, const Manifold& M, const Manifold& Mp, int k,
                     const std::vector<double>& dists, std::vector<int>& kNNInds, Eigen::Map<Eigen::MatrixXd> ystar,
                     Eigen::Map<Eigen::MatrixXd> coeffs, Eigen::Map<Eigen::MatrixXi> rc, int* kUsed)
{
  if (opts.algorithm == "llr") {
    // llr algorithm is not needed at this stage
    rc(t, Mp_i) = NOT_IMPLEMENTED;
    return;
  }

  // Remove neighbours which have a missing value for their distance
  kNNInds.erase(std::remove_if(kNNInds.begin(), kNNInds.end(),
                               [&dists, &M](int obs) { return dists[obs] == MISSING || M.any_missing(obs); }),
                kNNInds.end());
  k = kNNInds.size();

  *kUsed = k;

  if (k == 0) {
    ystar(t, Mp_i) = MISSING;
    rc(t, Mp_i) = SUCCESS;
    return;
  }

  // Pull out the nearest neighbours from the manifold, and
  // simultaneously prepend a column of ones in front of the manifold data.
  Eigen::MatrixXd X_ls_cj(k, M.E_actual() + 1);
  X_ls_cj << Eigen::VectorXd::Ones(k), M.map()(kNNInds, Eigen::all);

  // Calculate the weight for each neighbour
  Eigen::Map<const Eigen::VectorXd> distsMap(&(dists[0]), dists.size());
  Eigen::VectorXd d = distsMap(kNNInds);
  d /= d.mean();
  Eigen::VectorXd w = Eigen::exp(-opts.thetas[t] * d.array());

  // Scale everything by our weights vector
  X_ls_cj.array().colwise() *= w.array();
  Eigen::VectorXd y_ls = M.yMap()(kNNInds).array() * w.array();

  // The pseudo-inverse of X can be calculated as (X^T * X)^(-1) * X^T
  // see https://scicomp.stackexchange.com/a/33375
  Eigen::BDCSVD<Eigen::MatrixXd> svd(X_ls_cj.transpose() * X_ls_cj, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd ics = svd.solve(X_ls_cj.transpose() * y_ls);

  double r = ics(0);
  for (int j = 0; j < M.E_actual(); j++) {
    if (Mp(Mp_i, j) != MISSING) {
      r += Mp(Mp_i, j) * ics(j + 1);
    }
  }

  // If the 'savesmap' option is given, save the 'ics' coefficients
  // for the largest value of theta.
  if (opts.saveSMAPCoeffs && t == opts.thetas.size() - 1) {
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
ThreadPool workerPool, taskRunnerPool;

std::future<void> edm_task_async(Options opts, const ManifoldGenerator* generator, int E,
                                 std::vector<bool> trainingRows, std::vector<bool> predictionRows, IO* io,
                                 Prediction* pred, bool keep_going(), void all_tasks_finished(void))
{
  workerPool.set_num_workers(opts.nthreads);
  taskRunnerPool.set_num_workers(num_physical_cores());

  if (opts.taskNum == 0) {
    numTasksStarted = 0;
    numTasksFinished = 0;
  }

  numTasksStarted += 1;

  return taskRunnerPool.enqueue(
    [opts, generator, E, trainingRows, predictionRows, io, pred, keep_going, all_tasks_finished] {
      edm_task(opts, generator, E, trainingRows, predictionRows, io, pred, keep_going, all_tasks_finished);
    });
}

void edm_task(Options opts, const ManifoldGenerator* generator, int E, std::vector<bool> trainingRows,
              std::vector<bool> predictionRows, IO* io, Prediction* pred, bool keep_going(),
              void all_tasks_finished(void))
{
  bool multiThreaded = opts.nthreads > 1;

  if (multiThreaded) {
    workerPool.set_num_workers(opts.nthreads);
  }

  if (opts.taskNum == 0) {
    numTasksStarted = 0;
    numTasksFinished = 0;
  }

  numTasksStarted += 1;

  Manifold M, Mp;
  if (multiThreaded) {
    std::future<void> f1 = workerPool.enqueue([&] { M = generator->create_manifold(E, trainingRows, false); });
    std::future<void> f2 = workerPool.enqueue([&] { Mp = generator->create_manifold(E, predictionRows, true); });
    f1.get();
    f2.get();
  } else {
    M = generator->create_manifold(E, trainingRows, false);
    Mp = generator->create_manifold(E, predictionRows, true);
  }

  int numThetas = (int)opts.thetas.size();
  int numPredictions = Mp.nobs();
  int numCoeffCols = M.E_actual() + 1;

  auto predToTrainSelfMap = find_overlaps(trainingRows, predictionRows, numPredictions, opts.copredict);

  auto ystar = std::make_unique<double[]>(numThetas * numPredictions);
  std::fill_n(ystar.get(), numThetas * numPredictions, MISSING);
  Eigen::Map<Eigen::MatrixXd> ystarView(ystar.get(), numThetas, numPredictions);

  // If we're saving the coefficients (i.e. in xmap mode), then we're not running with multiple 'theta' values.
  auto coeffs = std::make_unique<double[]>(numPredictions * numCoeffCols);
  std::fill_n(coeffs.get(), numPredictions * numCoeffCols, MISSING);
  Eigen::Map<Eigen::MatrixXd> coeffsView(coeffs.get(), numPredictions, numCoeffCols);

  auto rc = std::make_unique<retcode[]>(numThetas * numPredictions);
  Eigen::Map<Eigen::Matrix<retcode, -1, -1>> rcView(rc.get(), numThetas, numPredictions);

  std::vector<int> kUsed;
  for (int i = 0; i < numPredictions; i++) {
    kUsed.push_back(-1);
  }

  if (opts.numTasks > 1 && opts.taskNum == 0) {
    io->progress_bar(0.0);
  }

  if (multiThreaded) {
    if (opts.distributeThreads) {
      distribute_threads(workerPool.workers);
    }

    std::vector<std::future<void>> results(numPredictions);
    for (int i = 0; i < numPredictions; i++) {
      results[i] = workerPool.enqueue([&, i] {
        make_prediction(i, opts, M, Mp, ystarView, rcView, coeffsView, &(kUsed[i]), predToTrainSelfMap[i], keep_going);
      });
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
  } else {
    if (opts.numTasks == 1) {
      io->progress_bar(0.0);
    }
    for (int i = 0; i < numPredictions; i++) {
      if (keep_going != nullptr && keep_going() == false) {
        break;
      }
      make_prediction(i, opts, M, Mp, ystarView, rcView, coeffsView, &(kUsed[i]), predToTrainSelfMap[i], keep_going);
      if (opts.numTasks == 1) {
        io->progress_bar((i + 1) / ((double)numPredictions));
      }
    }
  }

  // Store the results, so long as we weren't interrupted by a 'break'.
  if (keep_going == nullptr || keep_going() == true) {
    // Start by calculating the MAE & rho of prediction, if requested
    for (int t = 0; t < numThetas * opts.calcRhoMAE; t++) {
      PredictionStats stats;

      std::vector<double> y1, y2;
      for (int i = 0; i < Mp.ySize(); i++) {
        if (Mp.y(i) != MISSING && ystar[i] != MISSING) {
          y1.push_back(Mp.y(i));
          y2.push_back(ystarView(t, i));
        }
      }

      if (!(y1.empty() || y2.empty())) {
        stats.mae = mean_absolute_error(y1, y2);
        stats.rho = correlation(y1, y2);
      } else {
        stats.mae = MISSING;
        stats.rho = MISSING;
      }

      stats.taskNum = opts.taskNum + t;
      stats.calcRhoMAE = opts.calcRhoMAE;

      pred->stats.push_back(stats);
    }

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
      pred->predictionRows = std::move(predictionRows);
    }

    if (opts.saveKUsed) {
      pred->kUsed = kUsed;
    }

    pred->cmdLine = opts.cmdLine;

    pred->numThetas = numThetas;
    pred->numPredictions = numPredictions;
    pred->numCoeffCols = numCoeffCols;

    if (opts.numTasks > 1) {
      io->progress_bar((numTasksFinished + 1) / ((double)opts.numTasks));
    }
  }

  numTasksFinished += numThetas;

  if (numTasksFinished == opts.numTasks) {
    if (all_tasks_finished != nullptr) {
      all_tasks_finished();
    }
  }
}
