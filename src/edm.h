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

void simplex_prediction(int Mp_i, int t, const Options& opts, const Manifold& M, int k,
                        const std::vector<double>& dists, const std::vector<int>& kNNInds,
                        Eigen::Map<Eigen::MatrixXd> ystar, Eigen::Map<Eigen::MatrixXi> rc, int* kUsed);

void smap_prediction(int Mp_i, int t, const Options& opts, const Manifold& M, const Manifold& Mp, int k,
                     const std::vector<double>& dists, const std::vector<int>& kNNInds,
                     Eigen::Map<Eigen::MatrixXd> ystar, Eigen::Map<Eigen::MatrixXd> coeffs,
                     Eigen::Map<Eigen::MatrixXi> rc, int* kUsed);

std::vector<int> kNearestNeighboursIndices(const std::vector<double>& dists, int k, std::vector<int> idx);
