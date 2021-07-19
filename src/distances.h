#pragma once

#include "common.h"

std::vector<double> other_distances(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp, int skipRow,
                                    int& validDistances);
std::vector<double> wasserstein_distances(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp, int skipRow,
                                          int& validDistances);