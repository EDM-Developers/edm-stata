
#include "distances.h"
#include "EMD.h"

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>

std::unique_ptr<double[]> cost_matrix(const Manifold& M, const Manifold& Mp, int i, int j, double gamma,
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

      bool eitherMissing = (M_in == M.missing() || Mp_jm == Mp.missing());
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