#include "stats.h"

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>

double median(std::vector<double> u)
{
  if (u.size() % 2 == 0) {
    const auto median_it1 = u.begin() + u.size() / 2 - 1;
    const auto median_it2 = u.begin() + u.size() / 2;

    std::nth_element(u.begin(), median_it1, u.end());
    const auto e1 = *median_it1;

    std::nth_element(u.begin(), median_it2, u.end());
    const auto e2 = *median_it2;

    return (e1 + e2) / 2;
  } else {
    const auto median_it = u.begin() + u.size() / 2;
    std::nth_element(u.begin(), median_it, u.end());
    return *median_it;
  }
}

std::vector<int> rank(const std::vector<double>& v_temp)
{
  std::vector<std::pair<double, int>> v_sort(v_temp.size());

  for (int i = 0; i < v_sort.size(); ++i) {
    v_sort[i] = std::make_pair(v_temp[i], i);
  }

  sort(v_sort.begin(), v_sort.end());

  std::vector<int> result(v_temp.size());

  // N.B. Stata's rank starts at 1, not 0, so the "+1" is added here.
  for (int i = 0; i < v_sort.size(); ++i) {
    result[v_sort[i].second] = i + 1;
  }
  return result;
}

std::vector<double> remove_value(const std::vector<double>& vec, double target)
{
  std::vector<double> cleanedVec;
  for (const double& val : vec) {
    if (val != target) {
      cleanedVec.push_back(val);
    }
  }
  return cleanedVec;
}

double correlation(const std::vector<double>& y1, const std::vector<double>& y2)
{
  Eigen::Map<const Eigen::ArrayXd> y1Map(y1.data(), y1.size());
  Eigen::Map<const Eigen::ArrayXd> y2Map(y2.data(), y2.size());

  const Eigen::ArrayXd y1Cent = y1Map - y1Map.mean();
  ;
  const Eigen::ArrayXd y2Cent = y2Map - y2Map.mean();

  return (y1Cent * y2Cent).sum() / (std::sqrt((y1Cent * y1Cent).sum()) * std::sqrt((y2Cent * y2Cent).sum()));
}

double mean_absolute_error(const std::vector<double>& y1, const std::vector<double>& y2)
{
  Eigen::Map<const Eigen::ArrayXd> y1Map(y1.data(), y1.size());
  Eigen::Map<const Eigen::ArrayXd> y2Map(y2.data(), y2.size());
  double mae = (y1Map - y2Map).abs().mean();
  if (mae < 1e-10) {
    return 0;
  } else {
    return mae;
  }
}