#pragma once

#include "common.h"

DistanceIndexPairs lp_distances(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp,
                                std::vector<int> inds);
DistanceIndexPairs wasserstein_distances(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp,
                                         std::vector<int> inds);

DistanceIndexPairsOnGPU afLPDistances(const int numPredictions, const Options& opts,
                                      const ManifoldOnGPU& M, const ManifoldOnGPU& Mp,
                                      const af::array& metricOpts);
DistanceIndexPairsOnGPU afWassersteinDistances(int Mp_i, const Options& opts,
                                               const Manifold& hostM, const Manifold& hostMp,
                                               const ManifoldOnGPU& M, const ManifoldOnGPU& Mp,
                                               const af::array& inds, const af::array& metricOpts);
