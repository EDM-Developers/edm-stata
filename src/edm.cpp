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
#include "train_predict_split.h"

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
std::vector<int> kNearestNeighboursIndices(const std::vector<double>& dists, int k, std::vector<int> idx)
{
  // If we asked for all of the neighbours to be considered, just return this index vector directly.
  if (k >= idx.size()) {
    return idx;
  }

  if (k >= (int)(idx.size() / 2)) {
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

  // Remove the indices for the points which are not in the k nearest neighbour set.
  idx.erase(idx.begin() + k, idx.end());

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
// and using this as the prediction. As such, we throw away any neighbours which have a distance of 0 from
// the target point.
void make_prediction(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp,
                     Eigen::Map<Eigen::MatrixXd> ystar, Eigen::Map<Eigen::MatrixXi> rc,
                     Eigen::Map<Eigen::MatrixXd> coeffs, int* kUsed, bool keep_going() = nullptr)
{
  // An impatient user may want to cancel a long-running EDM command, so we occasionally check using this
  // callback to see whether we ought to keep going with this EDM command. Of course, this adds a tiny inefficiency,
  // but there doesn't seem to be a simple way to easily kill running worker threads across all OSs.
  if (keep_going != nullptr && keep_going() == false) {
    return;
  }

  std::vector<double> dists;

  if (opts.distance == Distance::Wasserstein) {
    dists = wasserstein_distances(Mp_i, opts, M, Mp);
  } else {
    dists = other_distances(Mp_i, opts, M, Mp);
  }

  if (keep_going != nullptr && keep_going() == false) {
    return;
  }

  // Create a list of possible neighbour indices.
  // Screen out missing or zero distances.
  // Also, for S-map we can't have any missing values in the Manifold, so insist on that also.
  std::vector<int> possibleNeighbourIndices;
  for (int i = 0; i < dists.size(); i++) {
    if (dists[i] == 0) {
      dists[i] = MISSING;
    }

    if (dists[i] != MISSING) {
      if (!(opts.algorithm == Algorithm::SMap && M.any_missing(i))) {
        possibleNeighbourIndices.push_back(i);
      }
    }
  }

  // Do we have enough distances to find k neighbours?
  int numValidDistances = possibleNeighbourIndices.size();
  int k = opts.k;
  *kUsed = numValidDistances;
  if (k > numValidDistances) {
    if (opts.forceCompute) {
      k = numValidDistances;
      if (k == 0) {
        return;
      }
    } else {
      rc(0, Mp_i) = INSUFFICIENT_UNIQUE;
      return;
    }
  }

  if (k == 0) {
    rc(0, Mp_i) = SUCCESS;
    return;
  }

  std::vector<int> kNNInds = kNearestNeighboursIndices(dists, k, possibleNeighbourIndices);

  if (keep_going != nullptr && keep_going() == false) {
    return;
  }

  if (opts.algorithm == Algorithm::Simplex) {
    for (int t = 0; t < opts.thetas.size(); t++) {
      simplex_prediction(Mp_i, t, opts, M, k, dists, kNNInds, ystar, rc, kUsed);
    }
  } else if (opts.algorithm == Algorithm::SMap) {
    for (int t = 0; t < opts.thetas.size(); t++) {
      smap_prediction(Mp_i, t, opts, M, Mp, k, dists, kNNInds, ystar, coeffs, rc, kUsed);
    }
  } else {
    rc(0, Mp_i) = INVALID_ALGORITHM;
  }
}

void simplex_prediction(int Mp_i, int t, const Options& opts, const Manifold& M, int k,
                        const std::vector<double>& dists, const std::vector<int>& kNNInds,
                        Eigen::Map<Eigen::MatrixXd> ystar, Eigen::Map<Eigen::MatrixXi> rc, int* kUsed)
{
  // Find the smallest distance (closest neighbour) among the supplied neighbours.
  double minDist = MISSING;
  for (int j = 0; j < k; j++) {
    if (dists[kNNInds[j]] < minDist) {
      minDist = dists[kNNInds[j]];
    }
  }

  // Calculate our weighting of each neighbour, and the total sum of these weights.
  std::vector<double> w(k);
  double sumw = 0.0;
  const double theta = opts.thetas[t];

  int numNonZeroWeights = 0;
  for (int j = 0; j < k; j++) {
    w[j] = exp(-theta * (dists[kNNInds[j]] / minDist));
    sumw = sumw + w[j];
    numNonZeroWeights += (w[j] > 0);
  }

  // For the sake of debugging, count how many neighbours we end up with.
  if (opts.saveKUsed) {
    *kUsed = numNonZeroWeights;
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
  // Pull out the nearest neighbours from the manifold, and
  // simultaneously prepend a column of ones in front of the manifold data.
  Eigen::MatrixXd X_ls_cj(k, M.E_actual() + 1);
  X_ls_cj << Eigen::VectorXd::Ones(k), M.map()(kNNInds, Eigen::all);

  // Calculate the weight for each neighbour
  Eigen::Map<const Eigen::VectorXd> distsMap(&(dists[0]), dists.size());
  Eigen::VectorXd d = distsMap(kNNInds);
  d /= d.mean();
  Eigen::VectorXd w = Eigen::exp(-opts.thetas[t] * d.array());

  // For the sake of debugging, count how many neighbours we end up with.
  if (opts.saveKUsed) {
    int numNonZeroWeights = 0;
    for (double& w_i : w) {
      if (w_i > 0) {
        numNonZeroWeights += 1;
      }
    }
    *kUsed = numNonZeroWeights;
  }

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

std::atomic<int> numTasksStarted = 0;
std::atomic<int> numTasksFinished = 0;
ThreadPool workerPool(0), taskRunnerPool(0);

Prediction edm_task(Options opts, ManifoldGenerator generator, int E, std::vector<bool> trainingRows,
                    std::vector<bool> predictionRows, IO* io, bool keep_going(), void all_tasks_finished(void))
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
    std::future<void> f1 = workerPool.enqueue([&] { M = generator.create_manifold(E, trainingRows, false); });
    std::future<void> f2 = workerPool.enqueue([&] { Mp = generator.create_manifold(E, predictionRows, true); });
    f1.get();
    f2.get();
  } else {
    M = generator.create_manifold(E, trainingRows, false);
    Mp = generator.create_manifold(E, predictionRows, true);
  }

  int numThetas = (int)opts.thetas.size();
  int numPredictions = Mp.nobs();
  int numCoeffCols = M.E_actual() + 1;

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
    std::vector<std::future<void>> results(numPredictions);
    for (int i = 0; i < numPredictions; i++) {
      results[i] = workerPool.enqueue(
        [&, i] { make_prediction(i, opts, M, Mp, ystarView, rcView, coeffsView, &(kUsed[i]), keep_going); });
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
      make_prediction(i, opts, M, Mp, ystarView, rcView, coeffsView, &(kUsed[i]), keep_going);
      if (opts.numTasks == 1) {
        io->progress_bar((i + 1) / ((double)numPredictions));
      }
    }
  }

  Prediction pred;

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

      pred.stats.push_back(stats);
    }

    // Check if any mf_smap_single call failed, and if so find the most serious error
    pred.rc = *std::max_element(rc.get(), rc.get() + numThetas * numPredictions);

    // If we're storing the prediction and/or the SMAP coefficients, put them
    // into the resulting Prediction struct. Otherwise, let them be deleted.
    if (opts.savePrediction) {
      // Take only the predictions for the largest theta value.
      if (numThetas == 1) {
        pred.ystar = std::move(ystar);
      } else {
        pred.ystar = std::make_unique<double[]>(numPredictions);
        for (int i = 0; i < numPredictions; i++) {
          pred.ystar[i] = ystarView(numThetas - 1, i);
        }
      }
    } else {
      pred.ystar = nullptr;
    }

    if (opts.saveSMAPCoeffs) {
      pred.coeffs = std::move(coeffs);
    } else {
      pred.coeffs = nullptr;
    }

    if (opts.savePrediction || opts.saveSMAPCoeffs) {
      pred.predictionRows = std::move(predictionRows);
    }

    if (opts.saveKUsed) {
      pred.kUsed = kUsed;
    }

    pred.cmdLine = opts.cmdLine;
    pred.copredict = opts.copredict;

    pred.numThetas = numThetas;
    pred.numPredictions = numPredictions;
    pred.numCoeffCols = numCoeffCols;

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

  return std::move(pred);
}

std::future<Prediction> launch_edm_task(ManifoldGenerator generator, Options opts, int taskNum, int E, int k,
                                        bool savePrediction, bool saveSMAPCoeffs, std::vector<bool> trainingRows,
                                        std::vector<bool> predictionRows, IO* io, bool keep_going(),
                                        void all_tasks_finished(void));

std::vector<std::future<Prediction>> launch_task_group(
  ManifoldGenerator generator, const Options& opts, std::vector<int> Es, std::vector<int> libraries, int k, int numReps,
  int crossfold, bool explore, bool full, bool saveFinalPredictions, bool saveSMAPCoeffs, bool copredictMode,
  std::vector<bool> usable, std::vector<double> co_x, std::vector<bool> coTrainingRows,
  std::vector<bool> coPredictionRows, std::string rngState, double nextRV, IO* io, bool keep_going(),
  void all_tasks_finished(void))
{
  // Construct the instance which will (repeatedly) split the data into either the training manifold
  // or the prediction manifold; sometimes this is randomised so the RNG state may need to be set.
  bool requiresRandomNumbers = TrainPredictSplitter::requiresRandomNumbers(crossfold, full);

  TrainPredictSplitter splitter;
  if (requiresRandomNumbers && !rngState.empty()) {
    splitter = TrainPredictSplitter(explore, full, crossfold, usable, rngState, nextRV);
  } else {
    splitter = TrainPredictSplitter(explore, full, crossfold, usable);
  }

  bool newTrainPredictSplit = true;
  int taskNum = 0;

  int numStandardTasks = numReps * Es.size() * (explore ? 1 : libraries.size());
  int numTasks = numStandardTasks + copredictMode;
  int E, kAdj, library;

  Options sharedOpts = opts;
  sharedOpts.copredict = false;
  sharedOpts.numTasks = numTasks;

  std::vector<std::future<Prediction>> futures;

  for (int iter = 1; iter <= numReps; iter++) {
    if (explore) {
      newTrainPredictSplit = true;
    }

    // If explore, set the library size here
    if (explore) {
      int trainSize = splitter.next_training_size(iter);
      libraries.clear();
      libraries.push_back(trainSize);
    }

    for (int i = 0; i < Es.size(); i++) {
      E = Es[i];

      for (int l = 0; l < libraries.size(); l++) {
        if (!explore) {
          newTrainPredictSplit = true;
        }

        library = libraries[l];

        // Set the number of neighbours to use
        if (k > 0) {
          kAdj = k;
        } else if (k < 0) {
          kAdj = library;
        } else if (k == 0) {
          bool isSMap = opts.algorithm == Algorithm::SMap;
          int defaultK = generator.E_actual(E) + 1 + isSMap;
          kAdj = defaultK < library ? defaultK : library;
        }

        taskNum += 1;

        bool savePrediction;
        if (explore) {
          savePrediction = saveFinalPredictions && ((crossfold > 0) || (taskNum == numStandardTasks));
        } else {
          savePrediction = saveFinalPredictions && (taskNum == numStandardTasks);
        }

        if (newTrainPredictSplit) {
          splitter.update_train_predict_split(library, iter);
          newTrainPredictSplit = false;
        }

        futures.emplace_back(launch_edm_task(generator, opts, taskNum - 1, E, kAdj, savePrediction, saveSMAPCoeffs,
                                             splitter.trainingRows(), splitter.predictionRows(), io, keep_going,
                                             all_tasks_finished));
      }
    }
  }

  if (copredictMode) {
    generator.add_coprediction_data(co_x);

    // Always saving prediction vector in coprediction mode.
    // Never calculating rho & MAE statistics in this mode.
    // Never saving SMAP coefficients in coprediction mode.
    Options copredOpts = opts;
    copredOpts.copredict = true;
    bool savePrediction = true;
    copredOpts.calcRhoMAE = false;
    saveSMAPCoeffs = false;

    taskNum += 1;

    futures.push_back(
      std::move(launch_edm_task(generator, copredOpts, taskNum - 1, E, kAdj, savePrediction, saveSMAPCoeffs,
                                coTrainingRows, coPredictionRows, io, keep_going, all_tasks_finished)));
  }

  return futures;
}

std::future<Prediction> edm_task_async(Options opts, ManifoldGenerator generator, int E, std::vector<bool> trainingRows,
                                       std::vector<bool> predictionRows, IO* io, bool keep_going(),
                                       void all_tasks_finished(void))
{
  workerPool.set_num_workers(opts.nthreads);
  taskRunnerPool.set_num_workers(num_physical_cores());

  if (opts.taskNum == 0) {
    numTasksStarted = 0;
    numTasksFinished = 0;
  }

  numTasksStarted += 1;

  return taskRunnerPool.enqueue([opts, generator, E, trainingRows, predictionRows, io, keep_going, all_tasks_finished] {
    return edm_task(opts, generator, E, trainingRows, predictionRows, io, keep_going, all_tasks_finished);
  });
}

std::future<Prediction> launch_edm_task(ManifoldGenerator generator, Options opts, int taskNum, int E, int k,
                                        bool savePrediction, bool saveSMAPCoeffs, std::vector<bool> trainingRows,
                                        std::vector<bool> predictionRows, IO* io, bool keep_going(),
                                        void all_tasks_finished(void))
{
  opts.taskNum = taskNum;
  opts.k = k;
  opts.savePrediction = savePrediction;
  opts.saveSMAPCoeffs = saveSMAPCoeffs;

  // Expand the metrics vector now we know the value of E
  std::vector<Metric> metrics;

  for (int repeat = 0; repeat < E; repeat++) {
    metrics.push_back(opts.metrics[0]);
  }

  // Add metrics for each dt variable (it is always treated as a continuous value)
  for (int repeat = 0; repeat < generator.E_dt(E); repeat++) {
    metrics.push_back(Metric::Diff);
  }

  int j = 1;
  for (int E_extra_count : generator.E_extras_counts(E)) {
    for (int repeat = 0; repeat < E_extra_count; repeat++) {
      metrics.push_back(opts.metrics[j]);
    }
    j += 1;
  }

  opts.metrics = metrics;

  return edm_task_async(opts, generator, E, trainingRows, predictionRows, io, keep_going, all_tasks_finished);
}
