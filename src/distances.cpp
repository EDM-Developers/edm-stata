
#include "distances.h"
#include "EMD.h"

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>

// TODO: Use an Eigen Map/Matrix to avoid calculating off-diagonal entries twice.
std::unique_ptr<double[]> wasserstein_cost_matrix(const Manifold& M, const Manifold& Mp, int i, int j, double gamma,
                                                  double missingDistance, int& len_i, int& len_j)
{
  bool skipMissing = (missingDistance == 0);
  len_i = skipMissing ? M.num_not_missing(i) : M.E_actual();
  len_j = skipMissing ? Mp.num_not_missing(j) : Mp.E_actual();

  auto costMatrix = std::make_unique<double[]>(len_i * len_j);
  double* nextCost = costMatrix.get();

  for (int n = 0; n < len_i; n += 1) {
    for (int m = 0; m < len_j; m += 1) {

      double M_in = M(i, n);
      double Mp_jm = Mp(j, m);

      bool eitherMissing = (M_in == MISSING || Mp_jm == MISSING);
      if (skipMissing && eitherMissing) {
        continue;
      }

      if (eitherMissing) {
        *nextCost = missingDistance * missingDistance;
      } else {
        *nextCost = (M_in - Mp_jm) * (M_in - Mp_jm) + gamma * (n - m) * (n - m);
      }
      nextCost += 1;
    }
  }
  return std::move(costMatrix);
}

double approx_wasserstein(double* C, int len_i, int len_j, double eps, double stopErr)
{
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> costMatrix(C, len_i, len_j);

  double r = 1.0 / len_i;
  double c = 1.0 / len_j;

  Eigen::MatrixXd K = Eigen::exp(-costMatrix.array() / eps);
  Eigen::MatrixXd Kp = len_i * K.array();

  Eigen::VectorXd u = Eigen::VectorXd::Ones(len_i) / len_i;
  Eigen::VectorXd v = Eigen::VectorXd::Ones(len_j) / len_j;

  int maxIter = 10000;
  for (int iter = 0; iter < maxIter; iter++) {

    v = c / (K.transpose() * u).array();
    u = 1.0 / (Kp * v).array();

    if (iter % 10 == 0) {
      // Compute right marginal (diag(u) K diag(v))^T1
      Eigen::VectorXd tempColSums = (u.asDiagonal() * K * v.asDiagonal()).colwise().sum();
      double L2err = (tempColSums.array() - c).matrix().norm();
      if (L2err < stopErr) {
        break;
      }
    }
  }

  Eigen::MatrixXd transportPlan = u.asDiagonal() * K * v.asDiagonal();
  double dist = (transportPlan.array() * costMatrix.array()).sum();
  return dist;
}

double wasserstein(double* C, int len_i, int len_j)
{
  // Create vectors which are just 1/len_i and 1/len_j of length len_i and len_j.
  auto w_1 = std::make_unique<double[]>(len_i);
  auto w_2 = std::make_unique<double[]>(len_j);
  for (int i = 0; i < len_i; i++) {
    w_1[i] = 1.0 / len_i;
  }
  for (int i = 0; i < len_j; i++) {
    w_2[i] = 1.0 / len_j;
  }

  int maxIter = 10000;
  double cost;
  EMD_wrap(len_i, len_j, w_1.get(), w_2.get(), C, nullptr, nullptr, nullptr, &cost, maxIter);
  return cost;
}

std::vector<double> wasserstein_distances(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp,
                                          int& validDistances)
{

  std::vector<double> dists(M.nobs());

  double gamma = Mp.range() / Mp.time_range() * opts.aspectRatio;

  validDistances = 0;

  for (int i = 0; i < M.nobs(); i++) {
    int len_i, len_j;
    auto C = wasserstein_cost_matrix(M, Mp, i, Mp_i, gamma, opts.missingdistance, len_i, len_j);

    if (len_i > 0 && len_j > 0) {
      dists[i] = std::sqrt(wasserstein(C.get(), len_i, len_j));
      validDistances += 1;

    } else {
      dists[i] = MISSING;
    }
  }

  return dists;
}

std::vector<double> other_distances(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp,
                                    int& validDistances)
{

  std::vector<double> dists(M.nobs());

  validDistances = 0;

  // Compare every observation in the M manifold to the
  // Mp_i'th observation in the Mp manifold.
  for (int i = 0; i < M.nobs(); i++) {
    // Calculate the distance between M[i] and Mp[Mp_i]
    double dist_i = 0.0;

    for (int j = 0; j < M.E_actual(); j++) {
      // Get the sub-distance between M[i,j] and Mp[Mp_i, j]
      double dist_ij;

      // If either of these values is missing, the distance from
      // M[i,j] to Mp[Mp_i, j] is opts.missingdistance.
      // However, if the user doesn't specify this, then the entire
      // M[i] to Mp[Mp_i] distance is set as missing.
      if ((M(i, j) == MISSING) || (Mp(Mp_i, j) == MISSING)) {
        if (opts.missingdistance == 0) {
          dist_i = MISSING;
          break;
        } else {
          dist_ij = opts.missingdistance;
        }
      } else {
        // Neither M[i,j] nor Mp[Mp_i, j] is missing.
        if (opts.metrics[j] == Metric::Diff) {
          dist_ij = M(i, j) - Mp(Mp_i, j);
        } else { // Metric::CheckSame
          dist_ij = (M(i, j) != Mp(Mp_i, j));
        }
      }

      if (opts.distance == Distance::MeanAbsoluteError) {
        dist_i += abs(dist_ij) / M.E_actual();
      } else { // Distance::Euclidean
        dist_i += dist_ij * dist_ij;
      }
    }

    if (dist_i != MISSING) {
      validDistances += 1;
      if (opts.distance == Distance::MeanAbsoluteError) {
        dists[i] = dist_i;
      } else { // Distance::Euclidean
        dists[i] += sqrt(dist_i);
      }
    } else {
      dists[i] = MISSING;
    }
  }

  return dists;
}