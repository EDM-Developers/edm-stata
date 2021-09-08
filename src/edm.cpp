/*
 * Implementation of EDM methods, including S-map and cross-mapping
 *
 * - Patrick Laub, Department of Management and Marketing,
 *   The University of Melbourne, patrick.laub@unimelb.edu.au
 * - Edoardo Tescari, Melbourne Data Analytics Platform,
 *  The University of Melbourne, e.tescari@unimelb.edu.au
 *
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

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/SVD>
#include <algorithm> // std::partial_sort
#include <cmath>
#include <fstream> // just to create low-level input dumps
#include <chrono>

#include <arrayfire.h>
#include <nvToolsExt.h>

constexpr bool useAFArm = true; //Flag to execute make_prediction using AF ports - debugging purposes

std::atomic<int> numTasksStarted = 0;
std::atomic<int> numTasksFinished = 0;
ThreadPool workerPool(0), taskRunnerPool(0);

std::vector<std::future<Prediction>> launch_task_group(const ManifoldGenerator& generator, Options opts,
                                                       const std::vector<int>& Es, const std::vector<int>& libraries,
                                                       int k, int numReps, int crossfold, bool explore, bool full,
                                                       bool saveFinalPredictions, bool saveFinalCoPredictions,
                                                       bool saveSMAPCoeffs, bool copredictMode,
                                                       const std::vector<bool>& usable, const std::string& rngState,
                                                       IO* io, bool keep_going(), void all_tasks_finished())
{
  static bool initOnce = [&]() {
    af::setMemStepSize(1024 * 1024 * 5);
    workerPool.set_num_workers(opts.nthreads);
    // taskRunnerPool.set_num_workers(num_physical_cores());
    taskRunnerPool.set_num_workers(1); // Avoid oversubscribing to the GPU
    return true;
  }();

  // Construct the instance which will (repeatedly) split the data
  // into either the training manifold or the prediction manifold.
  TrainPredictSplitter splitter = TrainPredictSplitter(explore, full, crossfold, usable, rngState);

  int numLibraries = (explore ? 1 : libraries.size());

  opts.numTasks = numReps * Es.size() * numLibraries;
  opts.configNum = 0;
  opts.taskNum = 0;

  int maxE = Es[Es.size() - 1];

  std::vector<bool> cousable;

  if (copredictMode) {
    opts.numTasks *= 2;
    cousable = generator.generate_usable(maxE, true);
  }

  int E, kAdj, library, trainSize;

  std::vector<std::future<Prediction>> futures;

  bool newTrainPredictSplit = true;

  // Note: the 'numReps' either refers to the 'replicate' option
  // used for bootstrap resampling, or the 'crossfold' number of
  // cross-validation folds. Both options can't be used together,
  // so numReps = max(replicate, crossfold).
  for (int iter = 1; iter <= numReps; iter++) {
    if (explore) {
      newTrainPredictSplit = true;
      trainSize = splitter.next_training_size(iter);
    }

    for (int i = 0; i < Es.size(); i++) {
      E = Es[i];

      // 'libraries' is implicitly set to one value in explore mode
      // though in xmap mode it is a user-supplied list which we loop over.
      for (int l = 0; l == 0 || l < libraries.size(); l++) {
        if (!explore) {
          newTrainPredictSplit = true;
        }

        if (explore) {
          library = trainSize;
        } else {
          library = libraries[l];
        }

        // Set the number of neighbours to use
        if (k > 0) {
          kAdj = k;
        } else if (k < 0) {
          kAdj = -1; // Leave a sentinel value so we know to skip the nearest neighbours calculation
        } else if (k == 0) {
          bool isSMap = opts.algorithm == Algorithm::SMap;
          int defaultK = generator.E_actual(E) + 1 + isSMap;
          kAdj = defaultK < library ? defaultK : library;
        }

        bool lastConfig = (E == maxE) && (l + 1 == numLibraries);

        if (explore) {
          opts.savePrediction = saveFinalPredictions && ((iter == numReps) || (crossfold > 0)) && lastConfig;
        } else {
          opts.savePrediction = saveFinalPredictions && (iter == numReps) && lastConfig;
        }
        opts.saveSMAPCoeffs = saveSMAPCoeffs;

        if (newTrainPredictSplit) {
          splitter.update_train_predict_split(library, iter);
          newTrainPredictSplit = false;
        }

        opts.copredict = false;
        opts.k = kAdj;

        futures.emplace_back(launch_edm_task(generator, opts, E, splitter.trainingRows(), splitter.predictionRows(), io,
                                             keep_going, all_tasks_finished));

        opts.taskNum += 1;

        if (copredictMode) {
          opts.copredict = true;
          if (explore) {
            opts.savePrediction = saveFinalCoPredictions && ((iter == numReps) || (crossfold > 0)) && lastConfig;
          } else {
            opts.savePrediction = saveFinalCoPredictions && ((iter == numReps)) && lastConfig;
          }
          opts.saveSMAPCoeffs = false;
          futures.emplace_back(
            launch_edm_task(generator, opts, E, splitter.trainingRows(), cousable, io, keep_going, all_tasks_finished));

          opts.taskNum += 1;
        }

        opts.configNum += opts.thetas.size();
      }
    }
  }

  return futures;
}

std::future<Prediction> launch_edm_task(const ManifoldGenerator& generator, Options opts, int E,
                                        const std::vector<bool>& trainingRows, const std::vector<bool>& predictionRows,
                                        IO* io, bool keep_going(), void all_tasks_finished())
{
  // Expand the 'metrics' vector now that we know the value of E.
  std::vector<Metric> metrics;

  // For the Wasserstein distance, it's more convenient to have one 'metric' for each variable (before taking lags).
  // However, for the L^1 / L^2 distances, it's more convenient to have one 'metric' for each individual
  // point of each observations, so metrics.size() == M.E_actual().
  if (opts.distance == Distance::Wasserstein) {
    // Add a metric for the main variable and for the dt variable.
    // These are always treated as a continuous values (though perhaps in the future this will change).
    metrics.push_back(Metric::Diff);
    if (generator.E_dt(E) > 0) {
      metrics.push_back(Metric::Diff);
    }

    // Add in the metrics for the 'extra' variables as they were supplied to us.
    for (int k = 0; k < generator.numExtras(); k++) {
      metrics.push_back(opts.metrics[k]);
    }
  } else {
    // Add metrics for the main variable and the dt variable and their lags.
    // These are always treated as a continuous values (though perhaps in the future this will change).
    for (int lagNum = 0; lagNum < E + generator.E_dt(E); lagNum++) {
      metrics.push_back(Metric::Diff);
    }

    // The user specified how to treat the extra variables.
    for (int k = 0; k < generator.numExtras(); k++) {
      int numLags = (k < generator.numExtrasLagged()) ? E : 1;
      for (int lagNum = 0; lagNum < numLags; lagNum++) {
        metrics.push_back(opts.metrics[k]);
      }
    }
  }

  opts.metrics = metrics;

  if (opts.taskNum == 0) {
    numTasksStarted = 0;
    numTasksFinished = 0;
  }

  numTasksStarted += 1;

  // This hack is simply to dump some really low level data structures
  // purely for the purpose of generating microbenchmarks.
  if (io != nullptr && io->verbosity > 4) {
    json lowLevelInputDump;
    lowLevelInputDump["generator"] = generator;
    lowLevelInputDump["opts"] = opts;
    lowLevelInputDump["E"] = E;
    lowLevelInputDump["trainingRows"] = trainingRows;
    lowLevelInputDump["predictionRows"] = predictionRows;

    std::ofstream o("lowLevelInputDump.json");
    o << lowLevelInputDump << std::endl;
  }

  // Note, we can't have missing data inside the training manifold when using the S-Map algorithm
  bool skipMissing = (opts.algorithm == Algorithm::SMap);

  Manifold M = generator.create_manifold(E, trainingRows, opts.copredict, false, opts.dtWeight, skipMissing);
  Manifold Mp = generator.create_manifold(E, predictionRows, opts.copredict, true, opts.dtWeight);

  return taskRunnerPool.enqueue([opts, M, Mp, predictionRows, io, keep_going, all_tasks_finished] {
    return edm_task(opts, M, Mp, predictionRows, io, keep_going, all_tasks_finished);
  });
}

Prediction edm_task(const Options opts, const Manifold M, const Manifold Mp, const std::vector<bool> predictionRows,
                    IO* io, bool keep_going(), void all_tasks_finished())
{
  af::setDevice(0); //TODO potentially can cycle through GPUS if > 1

  // Char is the internal representation of bool in ArrayFire
  std::vector<char> mopts;
  for(int j = 0; j < M.E_actual(); j++) {
      mopts.push_back(opts.metrics[j] == Metric::Diff);
  }

  af::array metricOpts(M.E_actual(), mopts.data());

  const ManifoldOnGPU gpuM  = M;
  const ManifoldOnGPU gpuMp = Mp;

  constexpr bool useAF = true;  //Being on trump mutli-threaded codepath
  bool multiThreaded = opts.nthreads > 1;
  int numThetas = (int)opts.thetas.size();
  int numPredictions = Mp.nobs();
  int numCoeffCols = M.E_actual() + 1;

  auto ystar = std::make_unique<double[]>(numThetas * numPredictions);
  std::fill_n(ystar.get(), numThetas * numPredictions, MISSING_D);
  Eigen::Map<MatrixXd> ystarView(ystar.get(), numThetas, numPredictions);

  // If we're saving the coefficients (i.e. in xmap mode), then we're not running with multiple 'theta' values.
  auto coeffs = std::make_unique<double[]>(numPredictions * numCoeffCols);
  std::fill_n(coeffs.get(), numPredictions * numCoeffCols, MISSING_D);
  Eigen::Map<MatrixXd> coeffsView(coeffs.get(), numPredictions, numCoeffCols);

  auto rc = std::make_unique<retcode[]>(numThetas * numPredictions);
  std::fill_n(rc.get(), numThetas * numPredictions, UNKNOWN_ERROR);
  Eigen::Map<MatrixXi> rcView(rc.get(), numThetas, numPredictions);

  std::vector<int> kUsed;
  for (int i = 0; i < numPredictions; i++) {
    kUsed.push_back(-1);
  }

  if (opts.numTasks > 1 && opts.taskNum == 0) {
    io->progress_bar(0.0);
  }

  if (multiThreaded && !useAF) {
    std::vector<std::future<void>> results(numPredictions);
    workerPool.sync();
    printf("Starting: %d\n", opts.taskNum);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numPredictions; i++) {
      results[i] = workerPool.enqueue(
        [&, i] { make_prediction(i, opts, M, Mp, ystarView, rcView, coeffsView, &(kUsed[i]), keep_going); }
        );
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
    workerPool.sync();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("CPU(t=%d): Task(%lu) took %lf seconds for %d predictions \n",
           opts.nthreads, opts.taskNum, diff.count(), numPredictions);
  } else {
    if (useAF) {
      af::sync(0);
      auto start = std::chrono::high_resolution_clock::now();
      af_make_prediction(numPredictions, opts, M, Mp,
              gpuM, gpuMp, metricOpts, ystarView, rcView, coeffsView, kUsed, keep_going);
      af::sync(0);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start;
      printf("GPU: Task(%lu) took %lf seconds for %d predictions \n", opts.taskNum, diff.count(), numPredictions);
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
  }

  Prediction pred;

  // Store the results, so long as we weren't interrupted by a 'break'.
  if (keep_going == nullptr || keep_going() == true) {
    // Start by calculating the MAE & rho of prediction, if requested
    for (int t = 0; t < numThetas * opts.calcRhoMAE; t++) {
      PredictionStats stats;

      // TODO POTENTIAL SPEEDUP: if ystar and y exist on GPU
      //      this could potentially be faster on GPU for larger nobs
      std::vector<double> y1, y2;

      for (int i = 0; i < Mp.ySize(); i++) {
        if (Mp.y(i) != MISSING_D && ystarView(t, i) != MISSING_D) {
          y1.push_back(Mp.y(i));
          y2.push_back(ystarView(t, i));
        }
      }

      if (!(y1.empty() || y2.empty())) {
        stats.mae = mean_absolute_error(y1, y2);
        stats.rho = correlation(y1, y2);
      } else {
        stats.mae = MISSING_D;
        stats.rho = MISSING_D;
      }

      pred.stats.push_back(stats);
    }

    pred.configNum = opts.configNum;

    // Check if any make_prediction call failed, and if so find the most serious error
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

  numTasksFinished += 1;

  if (numTasksFinished == opts.numTasks) {
    if (all_tasks_finished != nullptr) {
      all_tasks_finished();
    }
  }

  return pred;
}

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
void make_prediction(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp, Eigen::Map<MatrixXd> ystar,
                     Eigen::Map<MatrixXi> rc, Eigen::Map<MatrixXd> coeffs, int* kUsed, bool keep_going())
{
  af::setDevice(0);

  // An impatient user may want to cancel a long-running EDM command, so we occasionally check using this
  // callback to see whether we ought to keep going with this EDM command. Of course, this adds a tiny inefficiency,
  // but there doesn't seem to be a simple way to easily kill running worker threads across all OSs.
  if (keep_going != nullptr && keep_going() == false) {
    rc(0, Mp_i) = BREAK_HIT;
    return;
  }

  // Create a list of indices which may potentially be the neighbours of Mp(Mp_i,.)
  std::vector<int> tryInds =
          useAFArm ? af_potential_neighbour_indices(Mp_i, opts, M, Mp)
                   : potential_neighbour_indices(Mp_i, opts, M, Mp);

  DistanceIndexPairs potentialNN;
  if (opts.distance == Distance::Wasserstein) {
    potentialNN =
       useAFArm ? af_wasserstein_distances(Mp_i, opts, M, Mp, tryInds)
                : wasserstein_distances(Mp_i, opts, M, Mp, tryInds);
  } else {
    potentialNN =
       useAFArm ? af_lp_distances(Mp_i, opts, M, Mp, tryInds)
                : lp_distances(Mp_i, opts, M, Mp, tryInds);
  }

  if (keep_going != nullptr && keep_going() == false) {
    rc(0, Mp_i) = BREAK_HIT;
    return;
  }

  // Do we have enough distances to find k neighbours?
  int numValidDistances = potentialNN.inds.size();
  int k = opts.k;
  *kUsed = numValidDistances;
  if (k > numValidDistances) {
    if (opts.forceCompute) {
      k = numValidDistances;
    } else {
      rc(0, Mp_i) = INSUFFICIENT_UNIQUE;
      return;
    }
  }

  if (k == 0) {
    // Whether we throw an error or just silently ignore this prediction
    // depends on whether we are in 'strict' mode or not.
    rc(0, Mp_i) = opts.forceCompute ? SUCCESS : INSUFFICIENT_UNIQUE;
    return;
  }

  // If we asked for all of the neighbours to be considered (e.g. with k = -1), return this index vector directly.
  DistanceIndexPairs kNNs;
  if (k < 0 || k == potentialNN.inds.size()) {
    kNNs = potentialNN;
  } else {
    kNNs =
       useAFArm ? af_kNearestNeighbours(potentialNN, k)
                : kNearestNeighbours(potentialNN, k);
  }

  if (keep_going != nullptr && keep_going() == false) {
    rc(0, Mp_i) = BREAK_HIT;
    return;
  }

  if (opts.algorithm == Algorithm::Simplex) {
    if (useAFArm) {
      af_simplex_prediction(Mp_i, opts, M, kNNs.dists, kNNs.inds, ystar, rc, kUsed);
    } else {
      for (int t = 0; t < opts.thetas.size(); t++) {
        simplex_prediction(Mp_i, t, opts, M, kNNs.dists, kNNs.inds, ystar, rc, kUsed);
      }
    }
  } else if (opts.algorithm == Algorithm::SMap) {
    if (useAFArm) {
      af_smap_prediction(Mp_i, opts, M, Mp, kNNs.dists, kNNs.inds, ystar, coeffs, rc, kUsed);
    } else {
      for (int t = 0; t < opts.thetas.size(); t++) {
        smap_prediction(Mp_i, t, opts, M, Mp, kNNs.dists, kNNs.inds, ystar, coeffs, rc, kUsed);
      }
    }
  } else {
    rc(0, Mp_i) = INVALID_ALGORITHM;
  }
}

std::vector<int> potential_neighbour_indices(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp)
{
  bool skipOtherPanels = opts.panelMode && (opts.idw < 0);

  std::vector<int> inds;

  for (int i = 0; i < M.nobs(); i++) {
    if (skipOtherPanels && (M.panel(i) != Mp.panel(Mp_i))) {
      continue;
    }

    inds.push_back(i);
  }

  return inds;
}

// For a given point, find the k nearest neighbours of this point.
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
DistanceIndexPairs kNearestNeighbours(const DistanceIndexPairs& potentialNeighbours, int k)
{
  std::vector<int> idx(potentialNeighbours.inds.size());
  std::iota(idx.begin(), idx.end(), 0);

  if (k >= (int)(idx.size() / 2)) {
    auto comparator = [&potentialNeighbours](int i1, int i2) {
      return potentialNeighbours.dists[i1] < potentialNeighbours.dists[i2];
    };
    std::stable_sort(idx.begin(), idx.end(), comparator);
  } else {
    auto stableComparator = [&potentialNeighbours](int i1, int i2) {
      if (potentialNeighbours.dists[i1] != potentialNeighbours.dists[i2])
        return potentialNeighbours.dists[i1] < potentialNeighbours.dists[i2];
      else
        return i1 < i2;
    };
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), stableComparator);
  }

  std::vector<int> kNNInds(k);
  std::vector<double> kNNDists(k);

  for (int i = 0; i < k; i++) {
    kNNInds[i] = potentialNeighbours.inds[idx[i]];
    kNNDists[i] = potentialNeighbours.dists[idx[i]];
  }

  return { kNNInds, kNNDists };
}

// An alternative version of 'kNearestNeighbours' which doesn't sort the neighbours.
// This version splits ties differently on different OS's, so it can't be used directly,
// though perhaps a platform-independent implementation of std::nth_element would solve this problem.
DistanceIndexPairs kNearestNeighboursUnstable(const DistanceIndexPairs& potentialNeighbours, int k)
{
  std::vector<int> indsToPartition(potentialNeighbours.inds.size());
  std::iota(indsToPartition.begin(), indsToPartition.end(), 0);

  auto comparator = [&potentialNeighbours](int i1, int i2) {
    return potentialNeighbours.dists[i1] < potentialNeighbours.dists[i2];
  };
  std::nth_element(indsToPartition.begin(), indsToPartition.begin() + k, indsToPartition.end(), comparator);

  std::vector<int> kNNInds(k);
  std::vector<double> kNNDists(k);

  for (int i = 0; i < k; i++) {
    kNNInds[i] = potentialNeighbours.inds[indsToPartition[i]];
    kNNDists[i] = potentialNeighbours.dists[indsToPartition[i]];
  }

  return { kNNInds, kNNDists };
}

void simplex_prediction(int Mp_i, int t, const Options& opts, const Manifold& M, const std::vector<double>& dists,
                        const std::vector<int>& kNNInds, Eigen::Map<MatrixXd> ystar, Eigen::Map<MatrixXi> rc,
                        int* kUsed)
{
  int k = kNNInds.size();

  // Find the smallest distance (closest neighbour) among the supplied neighbours.
  double minDist = *std::min_element(dists.begin(), dists.end());

  // Calculate our weighting of each neighbour, and the total sum of these weights.
  std::vector<double> w(k);
  double sumw = 0.0;
  const double theta = opts.thetas[t];

  int numNonZeroWeights = 0;
  for (int j = 0; j < k; j++) {
    w[j] = exp(-theta * (dists[j] / minDist));
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

void smap_prediction(int Mp_i, int t, const Options& opts, const Manifold& M, const Manifold& Mp,
                     const std::vector<double>& dists, const std::vector<int>& kNNInds, Eigen::Map<MatrixXd> ystar,
                     Eigen::Map<MatrixXd> coeffs, Eigen::Map<MatrixXi> rc, int* kUsed)
{
  int k = kNNInds.size();

  // Pull out the nearest neighbours from the manifold, and
  // simultaneously prepend a column of ones in front of the manifold data.
  MatrixXd X_ls_cj(k, M.E_actual() + 1);
  X_ls_cj << Eigen::VectorXd::Ones(k), M.map()(kNNInds, Eigen::all);

  // Calculate the weight for each neighbour
  Eigen::Map<const Eigen::VectorXd> distsMap(&(dists[0]), dists.size());
  Eigen::VectorXd w = Eigen::exp(-opts.thetas[t] * (distsMap.array() / distsMap.mean()));

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

  // The old way to solve this system:
  // Eigen::BDCSVD<MatrixXd> svd(X_ls_cj, Eigen::ComputeThinU | Eigen::ComputeThinV);
  //  Eigen::VectorXd ics = svd.solve(y_ls);

  // The pseudo-inverse of X can be calculated as (X^T * X)^(-1) * X^T
  // see https://scicomp.stackexchange.com/a/33375
  const int svdOpts = Eigen::ComputeThinU | Eigen::ComputeThinV; // 'ComputeFull*' would probably work identically here.
  Eigen::BDCSVD<MatrixXd> svd(X_ls_cj.transpose() * X_ls_cj, svdOpts);
  Eigen::VectorXd ics = svd.solve(X_ls_cj.transpose() * y_ls);

  double r = ics(0);
  for (int j = 0; j < M.E_actual(); j++) {
    if (Mp(Mp_i, j) != MISSING_D) {
      r += Mp(Mp_i, j) * ics(j + 1);
    }
  }

  // If the 'savesmap' option is given, save the 'ics' coefficients
  // for the largest value of theta.
  if (opts.saveSMAPCoeffs && t == opts.thetas.size() - 1) {
    for (int j = 0; j < M.E_actual() + 1; j++) {
      if (ics(j) == 0.) {
        coeffs(Mp_i, j) = MISSING_D;
      } else {
        coeffs(Mp_i, j) = ics(j);
      }
    }
  }

  ystar(t, Mp_i) = r;
  rc(t, Mp_i) = SUCCESS;
}


/////////////////////////////////////////////////////////////// ArrayFire PORTED versions BEGIN HERE


std::vector<int> af_potential_neighbour_indices(int Mp_i, const Options& opts,
                                                const Manifold& M, const Manifold& Mp)
{
  using af::array;
  using af::anyTrue;
  using af::iota;

  const bool skipOtherPanels = opts.panelMode && (opts.idw < 0);
  const bool skipMissingData = (opts.algorithm == Algorithm::SMap);

  array result;

  if (skipOtherPanels && skipMissingData) {
      array m2dData(M.E_actual(), M.nobs(), M.flatf64().get()); // Matrix
      array mP1Ids(M.nobs(), M.panelIds().data());              // Vector

      array anyMsng2D   = (m2dData == M.missing());
      array anyMsngDta  = anyTrue(anyMsng2D, 0);                // Vector
      array valids      = !(anyMsngDta || (mP1Ids != Mp.panel(Mp_i)));

      result = where(valids).as(s32); // Valid inds.push_back(i) from non-af code
  } else if (skipOtherPanels) {
      array mP1Ids(M.nobs(), M.panelIds().data());              // Vector

      array valids = !(mP1Ids != Mp.panel(Mp_i));

      result = where(valids).as(s32); // Valid inds.push_back(i) from non-af code
  } else if (skipMissingData) {
      array m2dData(M.E_actual(), M.nobs(), M.flatf64().get()); // Matrix

      array anyMsng2D   = (m2dData == M.missing());
      array anyMsngDta  = anyTrue(anyMsng2D, 0);                // Vector
      array valids      = !(anyMsngDta);                        // Vector

      result = where(valids).as(s32); // Valid inds.push_back(i) from non-af code
  } else {
      using af::dim4;

      result = iota(dim4(M.nobs()), dim4(1), s32);
  }
  //return result;

  std::vector<int> inds(result.elements());
  if (inds.size()) {
      result.host(inds.data());
  }
  return inds;
}

DistanceIndexPairs af_kNearestNeighbours(const DistanceIndexPairs& potentialNeighbours, int k)
{
  using af::array;
  using af::iota;
  using af::sort;

  const size_t pnIndSize = potentialNeighbours.inds.size();

  array inIndices(pnIndSize, potentialNeighbours.inds.data());
  array inDists(pnIndSize, potentialNeighbours.dists.data());

  array odists, oidx;
  sort(odists, oidx, inDists, inIndices);

  const af::seq ks(k);
  array aKNNInds  = oidx(ks); // Assuming Indices in potentialNeighbours are not sorted
  array aKNNDists = odists(ks);

  // return { aKNNInds, aKNNDists };
  std::vector<int> kNNInds(k);
  std::vector<double> kNNDists(k);

  aKNNInds.host(kNNInds.data());
  aKNNDists.host(kNNDists.data());

  return { kNNInds, kNNDists };
}

void af_simplex_prediction(int Mp_i, const Options& opts, const Manifold& M,
                           const std::vector<double>& dists,
                           const std::vector<int>& kNNInds, Eigen::Map<MatrixXd> ystar,
                           Eigen::Map<MatrixXi> rc, int* kUsed)
{
  using af::array;

  const dim_t indsCount   = kNNInds.size();
  const dim_t thetasCount = opts.thetas.size();

  // kNNInds and dists are always of same size
  array  ainds(indsCount, kNNInds.data());
  array adists(indsCount, dists.data());
  array    aMy( M.nobs(), M.yvec().data());

  // Create [1 thetasCount 1 1] shape so that
  // we can reduce along dimension 0 later
  array othetas(1, thetasCount, opts.thetas.data());

  double minDist = af::min<double>(adists);

  array thetas  = af::tile(othetas, indsCount, 1);            // reshape to [indsCount thetasCount 1 1]
  array tadist  = af::tile(adists, 1, thetasCount);           // reshape to [indsCount thetasCount 1 1]
  array weights = af::exp( -thetas * ( tadist / minDist ) );
  array sumw    = af::sum(weights, 0);                        // Reduce for each theta
  array y       = aMy(ainds);
  array ty      = af::tile(y, 1, thetasCount);                // reshape to [indsCount thetasCount 1 1]
  array tsumw   = af::tile(sumw, indsCount, 1);               // reshape to [indsCount thetasCount 1 1]
  array rt      = ty * (weights / tsumw);
  array r       = af::sum(rt, 0);

  std::vector<double> rs(r.elements());
  if (r.elements() > 0) {
    r.host(rs.data());
    for (int t = 0; t < thetasCount; ++t) {
      ystar(t, Mp_i) = rs[t];
      rc(t, Mp_i) = SUCCESS;
    }
  }
  if (opts.saveKUsed) {
    *kUsed = af::count<int>(weights);
  }
}

void af_smap_prediction(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp,
                        const std::vector<double>& dists, const std::vector<int>& kNNInds,
                        Eigen::Map<MatrixXd> ystar, Eigen::Map<MatrixXd> coeffs,
                        Eigen::Map<MatrixXi> rc, int* kUsed)
{
  using af::array;
  using af::dim4;
  using af::end;
  using af::matmulTN;
  using af::mean;
  using af::moddims;
  using af::pinverse;
  using af::select;
  using af::seq;
  using af::span;
  using af::tile;

  const dim_t indsCount   = kNNInds.size();
  const dim_t thetasCount = opts.thetas.size();

  // E_actual is the axis with shortest stride i.e unit stride, M.nobs() being 2nd axies length
  array mData1( M.E_actual(),  M.nobs(), M.flatf64().get());
  array mData2(Mp.E_actual(), Mp.nobs(), Mp.flatf64().get());

  // kNNInds and dists are always of same size
  array  ainds(indsCount, kNNInds.data());
  array adists(indsCount, dists.data());
  array    aMy( M.nobs(), M.yvec().data());

  // Create [1 thetasCount 1 1] shape so that we can reduce along dimension 0 later
  array othetas(1, thetasCount, opts.thetas.data());

  double meanDist = mean<double>(adists);
  array thetas    = tile(othetas, indsCount, 1);                // reshape to [indsCount thetasCount 1 1]
  array tadist    = tile(adists, 1, thetasCount);               // reshape to [indsCount thetasCount 1 1]
  array weights   = af::exp( -thetas * ( tadist / meanDist ) );
  array y         = aMy(ainds);
  array ty        = tile(y, 1, thetasCount);                    // reshape to [indsCount thetasCount 1 1]
  array y_ls      = ty * weights;                               // [indsCount thetasCount 1 1] shape

  const dim_t MEactualp1 = M.E_actual() + 1;
  array X_ls_cj          = af::constant(1.0, dim4(MEactualp1, indsCount), f64);
  X_ls_cj(seq(1, end), span) = mData1(span, ainds);

  array weightsR  = moddims(weights, 1, indsCount, thetasCount);// [         1 indsCount thetasCount 1]
  array weightsT  = tile(weightsR, MEactualp1);                 // [MEactualp1 indsCount thetasCount 1]
  array X_ls_cj_T = tile(X_ls_cj, 1, 1, thetasCount);           // [MEactualp1 indsCount thetasCount 1]
  array y_ls_R    = moddims(y_ls, indsCount, 1, thetasCount);   // [ indsCount         1 thetasCount 1]

  X_ls_cj_T *= weightsT;

  array icsOuts = matmulTN(pinverse(X_ls_cj_T, 1e-9), y_ls_R);
  icsOuts       = moddims(icsOuts, MEactualp1, thetasCount);    // This should be just meta-data change
  array Mp_i_j  = tile(mData2(span, Mp_i), 1, thetasCount);

  array r2d = icsOuts(seq(1, end), span) * ((Mp_i_j != double(MISSING_D)) * Mp_i_j);
  array r   = icsOuts(0, span) + sum(r2d, 0);

  std::vector<double> rs(r.elements());
  if (r.elements() > 0) {
    r.host(rs.data());
    for (int t = 0; t < thetasCount; ++t) {
      ystar(t, Mp_i) = rs[t];
      rc(t, Mp_i) = SUCCESS;
    }
  }
  // If the 'savesmap' option is given, save the 'ics' coefficients
  // for the largest value of theta.
  if (opts.saveSMAPCoeffs) {
    array lastTheta  = icsOuts(span, thetasCount - 1);
    array icsl       = select(lastTheta == 0.0, double(MISSING_D), lastTheta);
    std::vector<double> host_ics(MEactualp1);
    icsl.host(host_ics.data());
    for (int j = 0; j < MEactualp1; j++) {
      coeffs(Mp_i, j) = host_ics[j];
    }
  }
  // For the sake of debugging, count how many neighbours we end up with.
  if (opts.saveKUsed) {
    *kUsed = af::count<int>(weights);
  }
}

// Returns b8 array of shape [mnobs npreds 1 1] when either of skip flags are true
//        otherwise of shape [mnobs 1 1 1]
af::array afPotentialNeighbourIndices(const int& npreds, const bool& skipOtherPanels, const bool& skipMissingData,
                                      const ManifoldOnGPU& M, const ManifoldOnGPU& Mp)
{
  using af::array;
  using af::anyTrue;
  using af::dim4;
  using af::iota;
  using af::seq;
  using af::tile;

  auto range = nvtxRangeStartA(__FUNCTION__);

  const dim_t mnobs = M.nobs;

  array result;
  if (skipOtherPanels && skipMissingData) {
      array npredsMp  = Mp.panel(seq(npreds));
      array panelM    = tile(M.panel, 1, npreds);
      array panelMp   = tile(npredsMp.T(), mnobs);
      array mssngM    = (M.mdata == M.missing);
      array msngCols  = anyTrue(mssngM, 0);
      array msngFlags = tile(msngCols.T(), 1, npreds);

      result = !(msngFlags || (panelM != panelMp));
  } else if (skipOtherPanels) {
      array npredsMp = Mp.panel(seq(npreds));
      array panelM   = tile(M.panel, 1, npreds);
      array panelMp  = tile(npredsMp.T(), mnobs);

      result = !(panelM != panelMp);
  } else if (skipMissingData) {
      result = tile(!(anyTrue(M.mdata == M.missing, 0).T()), 1, npreds);
  } else {
      result = af::constant(1.0, M.nobs, npreds, b8);
  }
  nvtxRangeEnd(range);
  return result;
}

void afSortSearchSpace(af::array& pValids, af::array& sDists, af::array& yvecs, af::array& smData,
        const af::array& vDists, const af::array& yvec, const af::array& mdata,
        const Algorithm algo, const int eacts, const int mnobs, const int npreds)
{
  using af::array;
  using af::dim4;
  using af::iota;
  using af::moddims;
  using af::sort;
  using af::tile;

  auto searchRange = nvtxRangeStartA("sortInfo");
  array maxs = af::max(pValids * vDists, 0);
  array pDists = pValids * vDists + (1 - pValids) * tile(maxs + 100, mnobs);

  array indices;
  sort(sDists, indices, pDists);

  yvecs = moddims(yvec(indices), mnobs, npreds);

  array vIdx = indices + iota(dim4(1, npreds), dim4(mnobs)) * mnobs;

  pValids = moddims(pValids(vIdx), mnobs, npreds);

  // Manifold data also needs to be reorder for SMap prediction
  if (algo == Algorithm::SMap) {
    array tmdata = tile(mdata, 1, 1, npreds);
    array soffs  = iota(dim4(1, 1, npreds), dim4(eacts, mnobs)) * (eacts * mnobs);
    array d0offs = iota(dim4(eacts), dim4(1, mnobs, npreds));

    indices = tile(moddims(indices, 1, mnobs, npreds), eacts) * eacts;
    indices += (soffs + d0offs);

    smData = moddims(tmdata(indices), eacts, mnobs, npreds);
  }

  nvtxRangeEnd(searchRange);
}

void afSimplexPrediction(af::array& retcodes, af::array& ystar, af::array& kused,
                         const int npreds, const Options& opts,
                         const af::array& yvecs,
                         const DistanceIndexPairsOnGPU& pair,
                         const af::array& thetas)
{
  using af::array;
  using af::sum;
  using af::tile;

  auto range = nvtxRangeStartA(__FUNCTION__);

  const array& valids = pair.inds;
  const array& dists  = pair.dists;
  const int mnobs     = valids.dims(0);
  const int tcount    = opts.thetas.size();

  array weights;
  {
    array minDist = tile(dists(0, af::span), mnobs, 1, tcount);
    array tadist  = tile(dists, 1, 1, tcount);

    weights = tile(valids, 1, 1, tcount) * af::exp(-thetas * (tadist / minDist));
  }
  array r4thetas = tile(yvecs, 1, 1, tcount) * (weights / tile(sum(weights, 0), mnobs));

  ystar = moddims(sum(r4thetas, 0), npreds, tcount);
  retcodes = af::constant(SUCCESS, npreds, tcount);

  if (opts.saveKUsed) {
    kused = moddims(af::count(weights, 0), npreds, tcount);
  }
  nvtxRangeEnd(range);
}

void afSMapPrediction(af::array& retcodes, af::array& kused,
                      af::array& ystar, af::array& coeffs,
                      const int npreds, const Options& opts,
                      const ManifoldOnGPU& M, const ManifoldOnGPU& Mp,
                      const DistanceIndexPairsOnGPU& pair, const af::array& mdata,
                      const af::array& yvecs, const af::array& thetas)
{
  using af::array;
  using af::constant;
  using af::dim4;
  using af::end;
  using af::matmulTN;
  using af::mean;
  using af::moddims;
  using af::pinverse;
  using af::select;
  using af::seq;
  using af::span;
  using af::tile;

  auto range = nvtxRangeStartA(__FUNCTION__);

  const array& valids  = pair.inds;
  const array& dists   = pair.dists;
  const int mnobs      = M.nobs;
  const int tcount     = opts.thetas.size();
  const int MEactualp1 = M.E_actual + 1;

  if (true) {
    array meanDists = tile((mnobs * mean(valids * dists, 0) / count(valids, 0)), mnobs, 1);
    array mdValids  = tile(moddims(valids, 1, mnobs, npreds), M.E_actual);
    array Mp_i_j    = Mp.mdata(span, seq(npreds));
    array scaleval  = ((Mp_i_j != double(MISSING_D)) * Mp_i_j);

    // Allocate Output arrays
    ystar = array(tcount, npreds, f64);

    for (int t = 0; t < tcount; ++t)
    {
      double theta = opts.thetas[t];

      array weights = valids * af::exp(-theta * (dists / meanDists));
      array y_ls    = weights * yvecs;

      array icsOuts;
      if (true) {
        icsOuts = array(MEactualp1, npreds, f64);
        for (int p = 0; p < npreds; ++p)
        {
          array X_ls_cj = constant(1.0, dim4(MEactualp1, mnobs), f64);

          X_ls_cj(seq(1, end), span) = mdValids(span, span, p) * mdata(span, span, p);

          X_ls_cj *= tile(moddims(weights(span, p), 1, mnobs), MEactualp1);

          icsOuts(span, p) = matmulTN(pinverse(X_ls_cj, 1e-9), y_ls(span, p));
        }
      } else {
        array X_ls_cj = constant(1.0, dim4(MEactualp1, mnobs, npreds), f64);

        X_ls_cj(seq(1, end), span) = mdValids * mdata;

        X_ls_cj *= tile(moddims(weights, 1, mnobs, npreds), MEactualp1);

        icsOuts = matmulTN(pinverse(X_ls_cj, 1e-9), moddims(y_ls, mnobs, 1, npreds));
        icsOuts = moddims(icsOuts, MEactualp1, npreds);
      }

      array r2d = icsOuts(seq(1, end), span) * scaleval;
      array r   = icsOuts(0, span) + sum(r2d, 0);

      ystar(t, span) = r;

      if (t == tcount - 1) {
        if (opts.saveSMAPCoeffs) {
          coeffs = select(icsOuts == 0.0, double(MISSING_D), icsOuts).T();
        }
        if (opts.saveKUsed) {
          kused = af::count(weights, 0);
        }
      }
    }
  } else {
    array weights, y_ls;
    {
      array meanDists  = (mnobs * mean(valids * dists, 0) / count(valids, 0));
      array meanDistsT = tile(meanDists, mnobs, 1, tcount);
      array ptDists    = tile(dists, 1, 1, tcount);
      array validsT    = tile(valids, 1, 1, tcount);

      weights = validsT * af::exp(-thetas * (ptDists / meanDistsT));
      y_ls    = weights * tile(yvecs, 1, 1, tcount);
    }

    array mdValids = tile(moddims(valids, 1, mnobs, npreds), M.E_actual);
    array X_ls_cj  = constant(1.0, dim4(MEactualp1, mnobs, npreds), f64);

    X_ls_cj(seq(1, end), span) = mdValids * mdata;

    array X_ls_cj_T = tile(X_ls_cj, 1, 1, 1, tcount);

    X_ls_cj_T *= tile(moddims(weights, 1, mnobs, npreds, tcount), MEactualp1);

    array icsOuts = matmulTN(pinverse(X_ls_cj_T, 1e-9), moddims(y_ls, mnobs, 1, npreds, tcount));

    icsOuts      = moddims(icsOuts, MEactualp1, npreds, tcount);
    array Mp_i_j = tile(Mp.mdata(span, seq(npreds)), 1, 1, tcount);
    array r2d    = icsOuts(seq(1, end), span, span) * ((Mp_i_j != double(MISSING_D)) * Mp_i_j);
    array r      = icsOuts(0, span, span) + sum(r2d, 0);

    ystar    = moddims(r, npreds, tcount).T();
    retcodes = constant(SUCCESS, npreds, tcount);
    if (opts.saveSMAPCoeffs) {
      array lastTheta  = icsOuts(span, span, tcount - 1);

      coeffs = select(lastTheta == 0.0, double(MISSING_D), lastTheta).T();
    }
    if (opts.saveKUsed) {
      kused = af::count(weights, 0)(span, end);
    }
  }

  retcodes = constant(SUCCESS, npreds, tcount);
  nvtxRangeEnd(range);
}

void af_make_prediction(const int npreds, const Options& opts,
                        const Manifold& hostM, const Manifold& hostMp,
                        const ManifoldOnGPU& M, const ManifoldOnGPU& Mp, const af::array& metricOpts,
                        Eigen::Map<MatrixXd> ystar, Eigen::Map<MatrixXi> rc,
                        Eigen::Map<MatrixXd> coeffs,
                        std::vector<int>& kUseds, bool keep_going())
{
  using af::array;
  using af::constant;
  using af::dim4;
  using af::iota;

  auto mpRange = nvtxRangeStartA(__FUNCTION__);
  const int numThetas = opts.thetas.size();

  if (opts.algorithm != Algorithm::Simplex && opts.algorithm != Algorithm::SMap) {
      array retcodes = constant(INVALID_ALGORITHM, npreds, numThetas, s32);
      retcodes.host(rc.data());
      return;
  }
  // Cross-paltform way to pre-empt threads to interrupt EDM command amid stages
  if (keep_going != nullptr && keep_going() == false) {
    return;
  }
  using af::span;
  using af::tile;
  using af::where;

  const bool skipOtherPanels = opts.panelMode && (opts.idw < 0);
  const bool skipMissingData = (opts.algorithm == Algorithm::SMap);

  array thetas = tile(array(1, 1, numThetas, opts.thetas.data()), M.nobs, npreds);

  auto pValids =
      afPotentialNeighbourIndices(npreds, skipOtherPanels, skipMissingData, M, Mp);

  auto validDistPair =
      afLPDistances(npreds, opts, M, Mp, metricOpts);

  auto kisRange = nvtxRangeStartA("kItemsSelection");
  // TODO add code path for wasserstein later
  pValids = pValids && validDistPair.inds;

  array retcodes;
  array kValids = iota(dim4(M.nobs), dim4(1, npreds));
  {
    const bool isfoco = opts.forceCompute;

    array kused = af::count(pValids, 0);
    // TODO check if saving at this point is necessary // kused.host(kUseds.data());

    array kchck = opts.k > kused;
    array kvals = kchck * (isfoco * kused) + (1 - kchck) * opts.k;
    array findk = select(kvals < 0 || kvals == kused, kused, kvals);

    kValids = kValids < tile(findk, M.nobs);
    //TODO
    //retcodes = kchck * !isfoco * int(INSUFFICIENT_UNIQUE) +
    //           kchck * isfoco * int(SUCCESS) +
    //           (kvals == 0) * int(SUCCESS);
  }
  nvtxRangeEnd(kisRange);

  //smData is set only if algo is SMap
  array sDists, yvecs, smData;

  afSortSearchSpace(pValids, sDists, yvecs, smData,
          validDistPair.dists, M.yvec, M.mdata,
          opts.algorithm, M.E_actual, M.nobs, npreds);

  pValids = pValids && kValids;

  array ystars, kused;
  if (opts.algorithm == Algorithm::Simplex) {
    afSimplexPrediction(retcodes, ystars, kused, npreds, opts,
                        yvecs, {pValids, sDists}, thetas);
  } else if (opts.algorithm == Algorithm::SMap) {
    array dcoeffs;
    afSMapPrediction(retcodes, kused, ystars, dcoeffs, npreds,
                     opts, M, Mp, {pValids, sDists}, smData, yvecs, thetas);
    if (opts.saveSMAPCoeffs) {
      dcoeffs.T().host(coeffs.data());
    }
  }

  auto returnRange = nvtxRangeStartA("ReturnValues");
  ystars.T().host(ystar.data());
  retcodes.T().host(rc.data());
  if (opts.saveKUsed) {
    kused.host(kUseds.data());
  }
  nvtxRangeEnd(returnRange);
  nvtxRangeEnd(mpRange);
}
