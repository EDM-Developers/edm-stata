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
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  if (k >= (int)v.size()) {
    k = (int)v.size();
  }

  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

  return idx;
}

void simplex(int Mp_i, int t, Options opts, const Manifold& M, int k, const std::vector<double>& d,
             const std::vector<size_t>& ind, span_2d_double ystar, span_2d_retcode rc);
void smap(int Mp_i, int t, Options opts, const Manifold& M, const Manifold& Mp, int k, const std::vector<double>& d,
          const std::vector<size_t>& ind, span_2d_double ystar, span_3d_double coeffs, span_2d_retcode rc);

void mf_smap_single(int Mp_i, Options opts, const Manifold& M, const Manifold& Mp, span_2d_double ystar,
                    span_2d_retcode rc, span_3d_double coeffs, bool keep_going() = nullptr)
{
  if (keep_going != nullptr && keep_going() == false) {
    for (int t = 0; t < opts.thetas.size(); t++) {
      rc(t, Mp_i) = UNKNOWN_ERROR;
    }
    return;
  }
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

  std::vector<size_t> ind = minindex(d, k);

  if (opts.algorithm == "" || opts.algorithm == "simplex") {
    for (int t = 0; t < opts.thetas.size(); t++) {
      simplex(Mp_i, t, opts, M, k, d, ind, ystar, rc);
    }
  } else if (opts.algorithm == "smap" || opts.algorithm == "llr") {
    for (int t = 0; t < opts.thetas.size(); t++) {
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
    w[j] = exp(-theta * sqrt(d[ind[j]] / d_base));
    sumw = sumw + w[j];
  }

  for (int j = 0; j < k; j++) {
    r = r + M.y(ind[j]) * (w[j] / sumw);
  }

  ystar(t, Mp_i) = r;
  rc(t, Mp_i) = SUCCESS;
}

void smap(int Mp_i, int t, Options opts, const Manifold& M, const Manifold& Mp, int k, const std::vector<double>& d,
          const std::vector<size_t>& ind, span_2d_double ystar, span_3d_double coeffs, span_2d_retcode rc)
{

  double d_base = d[ind[0]];
  std::vector<double> w(k);

  Eigen::MatrixXd X_ls(k, M.E_actual());
  std::vector<double> y_ls(k), w_ls(k);

  double mean_w = 0.;
  for (int j = 0; j < k; j++) {
    w[j] = sqrt(d[ind[j]]);
    mean_w = mean_w + w[j];
  }
  mean_w = mean_w / (double)k;

  double theta = opts.thetas[t];

  for (int j = 0; j < k; j++) {
    w[j] = exp(-theta * (w[j] / mean_w));
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
    if (opts.saveMode) {
      for (int j = 0; j < opts.varssv; j++) {
        if (ics(j) == 0.) {
          coeffs(t, Mp_i, j) = MISSING;
        } else {
          coeffs(t, Mp_i, j) = ics(j);
        }
      }
    }

    ystar(t, Mp_i) = r;
    rc(t, Mp_i) = SUCCESS;
  }
}

ThreadPool pool;

Prediction mf_smap_loop(Options opts, ManifoldGenerator generator, std::vector<bool> trainingRows,
                        std::vector<bool> predictionRows, const IO& io, bool keep_going(), void finished())
{
  Manifold M = generator.create_manifold(trainingRows, false);
  Manifold Mp = generator.create_manifold(predictionRows, true);

  size_t numThetas = opts.thetas.size();
  size_t numPredictions = Mp.nobs();

  Prediction pred;

  pred.numThetas = numThetas;
  pred.numPredictions = numPredictions;
  pred.numCoeffCols = opts.varssv;

  pred.ystar = std::make_unique<double[]>(numThetas * numPredictions);
  auto ystar = span_2d_double(pred.ystar.get(), (int)numThetas, (int)numPredictions);

  pred.coeffs = std::make_unique<double[]>(numThetas * numPredictions * opts.varssv);
  auto coeffs = span_3d_double(pred.coeffs.get(), (int)numThetas, (int)numPredictions, (int)opts.varssv);

  auto rc_data = std::make_unique<retcode[]>(numThetas * numPredictions);
  auto rc = span_2d_retcode(rc_data.get(), (int)numThetas, (int)numPredictions);

  auto start = std::chrono::high_resolution_clock::now();

  if (opts.nthreads <= 1) {
    opts.nthreads = 0;
  }

  pool.set_num_tasks(numPredictions);
  pool.set_num_workers(opts.nthreads);

  if (opts.distributeThreads) {
    distribute_threads(pool.workers);
  }

  std::vector<std::future<void>> results(numPredictions);
  if (opts.nthreads > 1) {
    for (int i = 0; i < numPredictions; i++) {
      results[i] = pool.enqueue([&, i] { mf_smap_single(i, opts, M, Mp, ystar, rc, coeffs, keep_going); });
    }
  }

  io.progress_bar(0.0);
  for (int i = 0; i < numPredictions; i++) {
    if (opts.nthreads == 0) {
      if (keep_going != nullptr && keep_going() == false) {
        break;
      }
      mf_smap_single(i, opts, M, Mp, ystar, rc, coeffs, nullptr);
    } else {
      results[i].get();
    }

    io.progress_bar((i + 1) / ((double)numPredictions));
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  if (keep_going != nullptr && keep_going() == false) {
    return { UNKNOWN_ERROR, {}, {} };
  }

  // Check if any mf_smap_single call failed, and if so find the most serious error
  pred.rc = *std::max_element(rc_data.get(), rc_data.get() + numThetas * numPredictions);

  // Calculate the MAE & rho of prediction, if requested
  pred.mae = MISSING;
  pred.rho = MISSING;

  if (opts.calcRhoMAE) {
    std::vector<double> y1, y2;
    for (int i = 0; i < Mp.ySize(); i++) {
      if (Mp.y(i) != MISSING && pred.ystar[i] != MISSING) {
        y1.push_back(Mp.y(i));
        y2.push_back(pred.ystar[i]);
      }
    }

    Eigen::Map<const Eigen::ArrayXd> y1Map(y1.data(), y1.size());
    Eigen::Map<const Eigen::ArrayXd> y2Map(y2.data(), y2.size());

    pred.mae = (y1Map - y2Map).abs().mean();

    const Eigen::ArrayXd y1Cent = y1Map - y1Map.mean();
    const Eigen::ArrayXd y2Cent = y2Map - y2Map.mean();

    pred.rho = (y1Cent * y2Cent).sum() / (std::sqrt((y1Cent * y1Cent).sum()) * std::sqrt((y2Cent * y2Cent).sum()));
  }

  pred.taskNum = opts.taskNum;
  pred.xmap = opts.xmap;
  pred.xmapDirectionNum = opts.xmapDirectionNum;
  pred.calcRhoMAE = opts.calcRhoMAE;

  if (finished != nullptr) {
    finished();
  }

  io.print_async(fmt::format("\nedm plugin took {} secs to make predictions\n", elapsed.count()));

  return std::move(pred);
}