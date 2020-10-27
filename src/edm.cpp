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
#include <cmath>
#include <Eigen/SVD>
#include <algorithm> // std::partial_sort
#include <iostream>
#include <numeric> // std::iota
#include <optional>

/* internal functions */
typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixView;
typedef Eigen::Block<const MatrixView, 1, -1, true> MatrixRowView;

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

auto get_distances(const MatrixView& M, const MatrixRowView& b, double missingdistance)
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
  return std::make_pair(d, validDistances);
}

double simplex(double theta, const std::vector<double>& y, const std::vector<double>& d, const std::vector<size_t>& ind,
               int l)
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

std::vector<size_t> valid_neighbour_indices(const MatrixView& M, const std::vector<double>& y,
                                            const std::vector<size_t>& ind, int l)
{
  std::vector<size_t> neighbourInds;

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

    if (!anyMissing) {
      neighbourInds.push_back(j);
    }
  }

  return neighbourInds;
}

auto setup_smap_llr(double theta, std::string algorithm, const std::vector<double>& y, const MatrixView& M,
                    const std::vector<double>& d, const std::vector<size_t>& ind, int l,
                    const std::vector<size_t>& neighbourInds)
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

  return std::make_tuple(X_ls_cj, y_ls_cj);
}

auto smap(const Eigen::MatrixXd& X_ls_cj, const Eigen::VectorXd y_ls_cj, const MatrixRowView& b)
{
  Eigen::BDCSVD<Eigen::MatrixXd> svd(X_ls_cj, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd ics = svd.solve(y_ls_cj);

  double r = ics(0);
  for (int j = 1; j < X_ls_cj.cols(); j++) {
    if (b(j - 1) != MISSING) {
      r += b(j - 1) * ics(j);
    }
  }

  return std::make_pair(r, ics);
}

retcode mf_smap_single(int Mp_i, smap_opts_t opts, const std::vector<double>& y, const MatrixView& M,
                       const MatrixView& Mp, std::vector<double>& ystar, std::optional<MatrixView>& Bi_map)
{
  if (opts.algorithm == "llr") {
    // llr algorithm is not needed at this stage
    return NOT_IMPLEMENTED;
  }

  MatrixRowView b = Mp.row(Mp_i);

  auto [d, validDistances] = get_distances(M, b, opts.missingdistance);

  // If we only look at distances which are non-zero and non-missing,
  // do we have enough of them to find 'l' neighbours?
  int l = opts.l;
  if (l > validDistances) {
    if (opts.force_compute && validDistances > 0) {
      l = validDistances;
    } else {
      return INSUFFICIENT_UNIQUE;
    }
  }

  std::vector<size_t> ind = minindex(d, l);

  if (opts.algorithm == "" || opts.algorithm == "simplex") {
    ystar[Mp_i] = simplex(opts.theta, y, d, ind, l);
    return SUCCESS;

  } else if (opts.algorithm == "smap" || opts.algorithm == "llr") {

    auto neighbourInds = valid_neighbour_indices(M, y, ind, l);

    if (neighbourInds.size() == 0) {
      ystar[Mp_i] = MISSING;
      return SUCCESS;
    }

    auto [X_ls_cj, y_ls_cj] = setup_smap_llr(opts.theta, opts.algorithm, y, M, d, ind, l, neighbourInds);

    if (opts.algorithm == "llr") {
      // llr algorithm is not needed at this stage
    } else {
      auto [r, ics] = smap(X_ls_cj, y_ls_cj, b);
      ystar[Mp_i] = r;

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

      return SUCCESS;
    }
  }

  return INVALID_ALGORITHM;
}

smap_res_t mf_smap_loop(smap_opts_t opts, const std::vector<double>& y, const manifold_t& M, const manifold_t& Mp,
                        int nthreads)
{
  // Create Eigen matrixes which are views of the supplied flattened matrices
  MatrixView M_mat((double*)M.flat.data(), M.rows, M.cols);     //  count_train_set, mani
  MatrixView Mp_mat((double*)Mp.flat.data(), Mp.rows, Mp.cols); // count_predict_set, mani

  std::optional<std::vector<double>> flat_Bi_map{};
  std::optional<MatrixView> Bi_map{};
  if (opts.save_mode) {
    flat_Bi_map = std::vector<double>(Mp.rows * opts.varssv);
    Bi_map = MatrixView(flat_Bi_map->data(), Mp.rows, opts.varssv);
  }

  // OpenMP loop with call to mf_smap_single function
  std::vector<retcode> rc(Mp.rows);
  std::vector<double> ystar(Mp.rows);

  if (nthreads <= 1) {
    for (int i = 0; i < Mp.rows; i++) {
      rc[i] = mf_smap_single(i, opts, y, M_mat, Mp_mat, ystar, Bi_map);
    }
  } else {
    ThreadPool pool(nthreads);
    std::vector<std::future<void>> results;

    for (int i = 0; i < Mp.rows; i++) {
      results.emplace_back(pool.enqueue([&, i] { rc[i] = mf_smap_single(i, opts, y, M_mat, Mp_mat, ystar, Bi_map); }));
    }
  }

  // Check if any mf_smap_single call failed, and if so find the most serious error
  retcode maxError = *std::max_element(rc.begin(), rc.end());

  return smap_res_t{ maxError, ystar, flat_Bi_map };
}
