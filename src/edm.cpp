/*
 * Implementation of EDM methods, including S-map and cross-mapping
 *
 * - Edoardo Tescari, Melbourne Data Analytics Platform,
 *  The University of Melbourne, e.tescari@unimelb.edu.au
 * - Patrick Laub, Department of Management and Marketing,
 *   The University of Melbourne, patrick.laub@unimelb.edu.au
 */

#include "edm.h"
#include "ctpl_stl.h"

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Core>
#include <Eigen/SVD>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include <algorithm> // std::partial_sort
#include <array>
#include <chrono>
#include <numeric> // std::iota
#include <stdexcept>
#include <string>

struct Prediction
{
  retcode rc;
  double y;
  Eigen::VectorXd coeffs;
};

/* minindex(v,k) returns the indices of the k minimums of v.  */
std::vector<size_t> minindex(const std::vector<double>& v, int k)
{
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  if (k >= (int)v.size()) {
    k = (int)v.size();
  }

  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

  return idx;
}

std::pair<std::vector<double>, int> get_distances(const Manifold& M, const Observation& b, double missingdistance)
{
  int validDistances = 0;
  std::vector<double> d(M.rows());

  for (int i = 0; i < M.rows(); i++) {
    double dist = 0.;
    bool missing = false;
    int numMissingDims = 0;
    for (int j = 0; j < M.cols(); j++) {
      double M_ij = M(i, j);
      double b_j = b(j);
      if ((M_ij == MISSING) || (b_j == MISSING)) {
        if (missingdistance == 0) {
          missing = true;
          break;
        }
        numMissingDims += 1;
      } else {
        dist += (M_ij - b_j) * (M_ij - b_j);
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

double simplex(double theta, const Eigen::ArrayXd& y, const Eigen::ArrayXd& d)
{
  Eigen::ArrayXd w = Eigen::exp(-theta * Eigen::sqrt(d / d[0]));
  w /= w.sum();
  return (y * w).sum();
}

std::vector<size_t> valid_neighbour_indices(const Manifold& M, const std::vector<double>& y,
                                            const std::vector<size_t>& ind, int l)
{
  std::vector<size_t> neighbourInds;

  for (int j = 0; j < l; j++) {
    if (y[ind[j]] == MISSING) {
      continue;
    }

    bool anyMissing = M.get_observation(ind[j]).any_missing();
    if (!anyMissing) {
      neighbourInds.push_back(j);
    }
  }

  return neighbourInds;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> setup_smap_llr(double theta, std::string algorithm,
                                                           const std::vector<double>& y, const Manifold& M,
                                                           const std::vector<double>& d, const std::vector<size_t>& ind,
                                                           int l, const std::vector<size_t>& neighbourInds)
{
  std::vector<double> w(l);
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
    w[j] = exp(-theta * (w[j] / mean_w));
  }

  for (int i = 0; i < neighbourInds.size(); i++) {
    if (algorithm == "llr") {
      // llr algorithm is not needed at this stage
    } else if (algorithm == "smap") {
      size_t k = neighbourInds[i];
      y_ls[i] = y[ind[k]] * w[k];
      w_ls[i] = w[k];
      for (int j = 0; j < M.cols(); j++) {
        X_ls(i, j) = M(ind[k], j) * w[k];
      }
    }
  }

  // Pull out the first 'rowc+1' elements of the y_ls vector and
  // concatenate the column vector 'w' with 'X_ls', keeping only
  // the first 'rowc+1' rows.
  Eigen::VectorXd y_ls_cj(neighbourInds.size());
  Eigen::MatrixXd X_ls_cj(neighbourInds.size(), M.cols() + 1);

  for (int i = 0; i < neighbourInds.size(); i++) {
    y_ls_cj(i) = y_ls[i];
    X_ls_cj(i, 0) = w_ls[i];
    for (int j = 0; j < X_ls.cols(); j++) {
      X_ls_cj(i, j + 1) = X_ls(i, j);
    }
  }

  return { X_ls_cj, y_ls_cj };
}

std::pair<double, Eigen::VectorXd> smap(const Eigen::MatrixXd& X_ls_cj, const Eigen::VectorXd y_ls_cj,
                                        const Observation& b)
{
  Eigen::BDCSVD<Eigen::MatrixXd> svd(X_ls_cj, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd ics = svd.solve(y_ls_cj);

  double r = ics(0);
  for (int j = 1; j < X_ls_cj.cols(); j++) {
    if (b(j - 1) != MISSING) {
      r += b(j - 1) * ics(j);
    }
  }

  return { r, ics };
}

Prediction mf_smap_single(int Mp_i, EdmOptions opts, const std::vector<double>& y, const Manifold& M,
                          const Manifold& Mp)
{
  Observation b = Mp.get_observation(Mp_i);

  auto [d, validDistances] = get_distances(M, b, opts.missingdistance);

  // If we only look at distances which are non-zero and non-missing,
  // do we have enough of them to find 'l' neighbours?
  int l = opts.l;
  if (l > validDistances) {
    if (opts.force_compute && validDistances > 0) {
      l = validDistances;
    } else {
      return { INSUFFICIENT_UNIQUE, {}, {} };
    }
  }

  std::vector<size_t> ind = minindex(d, l);
  Eigen::ArrayXd dNear(l), yNear(l);
  for (int i = 0; i < l; i++) {
    dNear[i] = d[ind[i]];
    yNear[i] = y[ind[i]];
  }

  if (opts.algorithm == "" || opts.algorithm == "simplex") {
    return { SUCCESS, simplex(opts.theta, yNear, dNear), {} };
  } else {

    auto neighbourInds = valid_neighbour_indices(M, y, ind, l);

    if (neighbourInds.size() == 0) {
      return { SUCCESS, MISSING, {} };
    }

    auto [X_ls_cj, y_ls_cj] = setup_smap_llr(opts.theta, opts.algorithm, y, M, d, ind, l, neighbourInds);
    auto [r, ics] = smap(X_ls_cj, y_ls_cj, b);
    return { SUCCESS, r, ics };
  }
}

ctpl::thread_pool pool;

EdmResult mf_smap_loop(EdmOptions opts, std::vector<double> y, Manifold M, Manifold Mp, int nthreads, IO io)
{
  try {
    const std::array<std::string, 4> validAlgs = { "", "llr", "simplex", "smap" };

    if (std::find(validAlgs.begin(), validAlgs.end(), opts.algorithm) == validAlgs.end()) {
      return { INVALID_ALGORITHM, {}, {} };
    }

    if (opts.algorithm == "llr") {
      return { NOT_IMPLEMENTED, {}, {} };
    }

    auto println = [io, opts](char* s, bool callflush = true, bool endl = true) {
      if (opts.verbosity > 1) {
        io.out(s);
        if (endl) {
          io.out((char*)"\n");
        }
        if (callflush) {
          io.flush();
        }
      }
    };

    if (nthreads != pool.size()) {
      println((char*)fmt::format("Resizing thread pool from {} to {} threads", pool.size(), nthreads).c_str());
      pool.resize(nthreads);
    }

    std::vector<retcode> rc(Mp.rows());
    std::vector<double> ystar(Mp.rows());
    std::vector<Eigen::VectorXd> coeffs(Mp.rows());
    std::vector<std::future<Prediction>> futures(Mp.rows());

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < Mp.rows(); i++) {
      futures[i] = pool.push([&, i](int) { return mf_smap_single(i, opts, y, M, Mp); });
    }

    for (int i = 0; i < Mp.rows(); i++) {
      Prediction pred = futures[i].get();
      if (opts.verbosity > 2 && i % (Mp.rows() / 10) == 0) {
        io.out(".");
        io.flush();
      }
      rc[i] = pred.rc;
      ystar[i] = pred.y;
      coeffs[i] = pred.coeffs;
    }
    println("");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    println((char*)fmt::format("EDM plugin took {} seconds to make predictions", elapsed.count()).c_str());

    double rho = 0.0, mae = 0.0;

    // Check if any mf_smap_single call failed, and if so find the most serious error
    retcode maxError = *std::max_element(rc.begin(), rc.end());

    // saving ics coefficients if savesmap option enabled
    std::vector<double> flat_Bi_map{};
    if (opts.save_mode) {
      flat_Bi_map.resize(Mp.rows() * opts.varssv);
      Eigen::Map<Eigen::MatrixXd> Bi_map(flat_Bi_map.data(), Mp.rows(), opts.varssv);

      for (int i = 0; i < Mp.rows(); i++) {
        Eigen::VectorXd ics = coeffs[i];
        for (int j = 0; j < opts.varssv; j++) {
          if (ics.size() == 0 || ics(j) == 0.) {
            Bi_map(i, j) = MISSING;
          } else {
            Bi_map(i, j) = ics(j);
          }
        }
      }
    }

    return { maxError, mae, rho, ystar, flat_Bi_map, opts };

  } catch (const std::exception& e) {
    io.error(e.what());
    io.error("\n");
  } catch (...) {
    io.error("Unknown error in edm plugin\n");
  }

  return { UNKNOWN_ERROR, {}, {}, {}, {}, {} };
}
