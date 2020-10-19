/*
 * Implementation of EDM methods, including S-map and cross-mapping
 *
 * - Edoardo Tescari, Melbourne Data Analytics Platform,
 *  The University of Melbourne, e.tescari@unimelb.edu.au
 * - Patrick Laub, Department of Management and Marketing,
 *   The University of Melbourne, patrick.laub@unimelb.edu.au
 */

#include "edm.h"
#include <math.h>
#include <stdlib.h>

#include <Eigen/SVD>
#include <algorithm> // std::partial_sort
#include <numeric>   // std::iota
#include <optional>

/* internal functions */
typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> MatrixView;

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

retcode mf_smap_single(int Mp_i, int l, double theta, std::string algorithm, int varssv, bool force_compute,
                       double missingdistance, const MatrixView& M, const MatrixView& Mp, const std::vector<double>& y,
                       std::vector<double>& ystar, std::optional<MatrixView>& Bi_map)
{
  int validDistances = 0;
  std::vector<double> d(M.rows());
  auto b = Mp.row(Mp_i);

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

  // If we only look at distances which are non-zero and non-missing,
  // do we have enough of them to find 'l' neighbours?
  if (l > validDistances) {
    if (force_compute) {
      l = validDistances;
      if (l == 0) {
        return INSUFFICIENT_UNIQUE;
      }
    } else {
      return INSUFFICIENT_UNIQUE;
    }
  }

  std::vector<size_t> ind = minindex(d, l);

  double d_base = d[ind[0]];
  std::vector<double> w(l);

  double sumw = 0., r = 0.;
  if (algorithm == "" || algorithm == "simplex") {
    for (int j = 0; j < l; j++) {
      /* TO BE ADDED: benchmark pow(expression,0.5) vs sqrt(expression) */
      /* w[j] = exp(-theta*pow((d[ind[j]] / d_base),(0.5))); */
      w[j] = exp(-theta * sqrt(d[ind[j]] / d_base));
      sumw = sumw + w[j];
    }
    for (int j = 0; j < l; j++) {
      r = r + y[ind[j]] * (w[j] / sumw);
    }

    ystar[Mp_i] = r;
    return SUCCESS;

  } else if (algorithm == "smap" || algorithm == "llr") {

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

    int rowc = -1;
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
      if (anyMissing) {
        continue;
      }
      rowc++;
      if (algorithm == "llr") {
        // llr algorithm is not needed at this stage
        return NOT_IMPLEMENTED;

      } else if (algorithm == "smap") {
        y_ls[rowc] = y[ind[j]] * w[j];
        w_ls[rowc] = w[j];
        for (int i = 0; i < M.cols(); i++) {
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
    Eigen::MatrixXd X_ls_cj(rowc + 1, M.cols() + 1);

    for (int i = 0; i < rowc + 1; i++) {
      y_ls_cj(i) = y_ls[i];
      X_ls_cj(i, 0) = w_ls[i];
      for (int j = 1; j < X_ls.cols() + 1; j++) {
        X_ls_cj(i, j) = X_ls(i, j - 1);
      }
    }

    if (algorithm == "llr") {
      // llr algorithm is not needed at this stage
      return NOT_IMPLEMENTED;
    } else {
      Eigen::BDCSVD<Eigen::MatrixXd> svd(X_ls_cj, Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::VectorXd ics = svd.solve(y_ls_cj);

      r = ics(0);
      for (int j = 1; j < M.cols() + 1; j++) {
        if (b(j - 1) != MISSING) {
          r += b(j - 1) * ics(j);
        }
      }

      // saving ics coefficients if savesmap option enabled
      if (Bi_map.has_value()) {
        for (int j = 0; j < varssv; j++) {
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

smap_res_t mf_smap_loop(int count_predict_set, int count_train_set, int mani, int Mpcol, int l, double theta,
                        std::string algorithm, bool save_mode, int varssv, bool force_compute, double missingdistance,
                        const std::vector<double>& y, const std::vector<double>& S, const std::vector<double>& flat_M,
                        const std::vector<double>& flat_Mp)
{
  // Create Eigen matrixes which are views of the supplied flattened matrices
  MatrixView M((double*)flat_M.data(), count_train_set, mani);
  MatrixView Mp((double*)flat_Mp.data(), count_predict_set, mani);

  std::optional<std::vector<double>> flat_Bi_map = std::nullopt;
  std::optional<MatrixView> Bi_map = std::nullopt;
  if (save_mode) {
    flat_Bi_map = std::vector<double>(count_predict_set * varssv);
    Bi_map = MatrixView(flat_Bi_map->data(), count_predict_set, varssv);
  }

  // OpenMP loop with call to mf_smap_single function
  int i;
  std::vector<retcode> rc(Mp.rows());
  std::vector<double> ystar(count_predict_set);

#pragma omp parallel for
  for (i = 0; i < Mp.rows(); i++) {
    rc[i] = mf_smap_single(i, l, theta, algorithm, varssv, force_compute, missingdistance, M, Mp, y, ystar, Bi_map);
  }

  // Check if any mf_smap_single call failed, and if so find the most serious error
  retcode maxError = *std::max_element(rc.begin(), rc.end());

  return smap_res_t{ maxError, ystar, flat_Bi_map };
}
