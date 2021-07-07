
#include "distances.h"
#include "EMD.h"

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

  int maxIter = 100000;
  double cost;
  EMD_wrap(len_i, len_j, w_1.get(), w_2.get(), C, nullptr, nullptr, nullptr, &cost, maxIter);
  return cost;
}