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
#include <optional>

/* internal functions */
typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixView;

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

retcode mf_smap_single(int Mp_i, Options opts, const std::vector<double>& y, const Manifold& M, const Manifold& Mp,
                       std::vector<double>& ystar, std::optional<MatrixView>& Bi_map, bool keep_going() = nullptr)
{
  if (keep_going != nullptr && keep_going() == false) {
    return UNKNOWN_ERROR;
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
  // do we have enough of them to find 'l' neighbours?
  int k = opts.k;
  if (k > validDistances) {
    if (opts.forceCompute) {
      k = validDistances;
      if (k == 0) {
        return INSUFFICIENT_UNIQUE;
      }
    } else {
      return INSUFFICIENT_UNIQUE;
    }
  }

  std::vector<size_t> ind = minindex(d, k);

  double d_base = d[ind[0]];
  std::vector<double> w(k);

  double sumw = 0., r = 0.;
  if (opts.algorithm == "" || opts.algorithm == "simplex") {
    for (int j = 0; j < k; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = exp(-theta*pow((d[ind[j]] / d_base),(0.5))); */
      w[j] = exp(-opts.theta * sqrt(d[ind[j]] / d_base));
      sumw = sumw + w[j];
    }
    for (int j = 0; j < k; j++) {
      r = r + y[ind[j]] * (w[j] / sumw);
    }

    ystar[Mp_i] = r;
    return SUCCESS;

  } else if (opts.algorithm == "smap" || opts.algorithm == "llr") {

    Eigen::MatrixXd X_ls(k, M.E_actual());
    std::vector<double> y_ls(k), w_ls(k);

    double mean_w = 0.;
    for (int j = 0; j < k; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = pow(d[ind[j]],0.5); */
      w[j] = sqrt(d[ind[j]]);
      mean_w = mean_w + w[j];
    }
    mean_w = mean_w / (double)k;
    for (int j = 0; j < k; j++) {
      w[j] = exp(-opts.theta * (w[j] / mean_w));
    }

    int rowc = -1;
    for (int j = 0; j < k; j++) {
      if (y[ind[j]] == MISSING) {
        continue;
      }

      if (M.any_missing(ind[j])) {
        continue;
      }
      rowc++;
      if (opts.algorithm == "llr") {
        // llr algorithm is not needed at this stage
        return NOT_IMPLEMENTED;

      } else if (opts.algorithm == "smap") {
        y_ls[rowc] = y[ind[j]] * w[j];
        w_ls[rowc] = w[j];
        for (int i = 0; i < M.E_actual(); i++) {
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
      return NOT_IMPLEMENTED;
    } else {
      Eigen::BDCSVD<Eigen::MatrixXd> svd(X_ls_cj, Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::VectorXd ics = svd.solve(y_ls_cj);

      r = ics(0);
      for (int j = 1; j < M.E_actual() + 1; j++) {
        if (Mp(Mp_i, j - 1) != MISSING) {
          r += Mp(Mp_i, j - 1) * ics(j);
        }
      }

      // saving ics coefficients if savesmap option enabled
      if (opts.saveMode) {
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

ThreadPool pool;

Prediction mf_smap_loop(Options opts, const std::vector<double>& y, const Manifold& M, const Manifold& Mp, const IO& io,
                        bool keep_going(), void finished())
{
  size_t numPredictions = Mp.nobs();

  std::optional<std::vector<double>> flat_Bi_map{};
  std::optional<MatrixView> Bi_map{};
  if (opts.saveMode) {
    flat_Bi_map = std::vector<double>(numPredictions * opts.varssv);
    Bi_map = MatrixView(flat_Bi_map->data(), numPredictions, opts.varssv);
  }

  // OpenMP loop with call to mf_smap_single function
  std::vector<retcode> rc(numPredictions);
  std::vector<double> ystar(numPredictions);

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
      results[i] = pool.enqueue([&, i] { rc[i] = mf_smap_single(i, opts, y, M, Mp, ystar, Bi_map, keep_going); });
    }
  }

  io.progress_bar(0.0);
  for (int i = 0; i < numPredictions; i++) {
    if (opts.nthreads == 0) {
      if (keep_going != nullptr && keep_going() == false) {
        break;
      }
      rc[i] = mf_smap_single(i, opts, y, M, Mp, ystar, Bi_map, nullptr);
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
  retcode maxError = *std::max_element(rc.begin(), rc.end());

  if (finished != nullptr) {
    finished();
  }

  io.print_async(fmt::format("\nedm plugin took {} secs to make predictions\n", elapsed.count()));

  return { maxError, ystar, flat_Bi_map };
}
