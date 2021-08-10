#pragma once

#include "common.h"

DistanceIndexPairs lp_distances(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp,
                                std::vector<int> inds);
DistanceIndexPairs af_lp_distances(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp,
                                   std::vector<int> inds);
DistanceIndexPairs wasserstein_distances(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp,
                                         std::vector<int> inds);
