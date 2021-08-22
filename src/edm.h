#pragma once

#include "common.h"

std::vector<std::future<Prediction>> launch_task_group(
  const ManifoldGenerator& generator, const Options& opts, const std::vector<int>& Es,
  const std::vector<int>& libraries, int k, int numReps, int crossfold, bool explore, bool full,
  bool saveFinalPredictions, bool saveSMAPCoeffs, bool copredictMode, const std::vector<bool>& usable,
  const std::vector<bool>& coTrainingRows, const std::vector<bool>& coPredictionRows, const std::string& rngState,
  double nextRV, IO* io, bool keep_going(), void all_tasks_finished());

std::future<Prediction> launch_edm_task(const ManifoldGenerator& generator, Options opts, int taskNum, int E, int k,
                                        bool savePrediction, bool saveSMAPCoeffs, const std::vector<bool>& trainingRows,
                                        const std::vector<bool>& predictionRows, IO* io, bool keep_going(),
                                        void all_tasks_finished());

Prediction edm_task(const Options opts, const Manifold M, const Manifold Mp, const std::vector<bool> predictionRows,
                    IO* io, bool keep_going(), void all_tasks_finished());

void make_prediction(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp,
                     Eigen::Map<Eigen::MatrixXd> ystar, Eigen::Map<Eigen::MatrixXi> rc,
                     Eigen::Map<Eigen::MatrixXd> coeffs, int* kUsed, bool keep_going());

std::vector<int> potential_neighbour_indices(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp);

DistanceIndexPairs kNearestNeighbours(const DistanceIndexPairs& potentialNeighbours, int k);

void simplex_prediction(int Mp_i, int t, const Options& opts, const Manifold& M, const std::vector<double>& dists,
                        const std::vector<int>& kNNInds, Eigen::Map<Eigen::MatrixXd> ystar,
                        Eigen::Map<Eigen::MatrixXi> rc, int* kUsed);

void smap_prediction(int Mp_i, int t, const Options& opts, const Manifold& M, const Manifold& Mp,
                     const std::vector<double>& dists, const std::vector<int>& kNNInds,
                     Eigen::Map<Eigen::MatrixXd> ystar, Eigen::Map<Eigen::MatrixXd> coeffs,
                     Eigen::Map<Eigen::MatrixXi> rc, int* kUsed);

std::vector<int> af_potential_neighbour_indices(int Mp_i, const Options& opts,
                                                const Manifold& M, const Manifold& Mp);

DistanceIndexPairs af_kNearestNeighbours(const DistanceIndexPairs& potentialNeighbours, int k);

void af_simplex_prediction(int Mp_i, const Options& opts, const Manifold& M, const std::vector<double>& dists,
                           const std::vector<int>& kNNInds, Eigen::Map<Eigen::MatrixXd> ystar,
                           Eigen::Map<Eigen::MatrixXi> rc, int* kUsed);

void af_smap_prediction(int Mp_i, const Options& opts, const Manifold& M, const Manifold& Mp,
                        const std::vector<double>& dists, const std::vector<int>& kNNInds,
                        Eigen::Map<Eigen::MatrixXd> ystar, Eigen::Map<Eigen::MatrixXd> coeffs,
                        Eigen::Map<Eigen::MatrixXi> rc, int* kUsed);

void af_make_prediction(const int numPredictions, const Options& opts,
                        const Manifold& hostM, const Manifold& hostMp,
                        const ManifoldOnGPU& M, const ManifoldOnGPU& Mp, const af::array& metricOpts,
                        Eigen::Map<Eigen::MatrixXd> ystar, Eigen::Map<Eigen::MatrixXi> rc,
                        Eigen::Map<Eigen::MatrixXd> coeffs,
                        std::vector<int>& kUseds, bool keep_going());
