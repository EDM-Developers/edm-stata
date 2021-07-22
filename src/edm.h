#pragma once

#include "common.h"

void edm_task(Options opts, ManifoldGenerator generator, int E, std::vector<bool> trainingRows,
              std::vector<bool> predictionRows, IO* io, Prediction* pred, bool keep_going() = nullptr,
              void all_tasks_finished(void) = nullptr);

std::future<void> edm_task_async(Options opts, ManifoldGenerator generator, int E, std::vector<bool> trainingRows,
                                 std::vector<bool> predictionRows, IO* io, Prediction* pred,
                                 bool keep_going() = nullptr, void all_tasks_finished(void) = nullptr);

int launch_task_group(ManifoldGenerator generator, const Options& opts, std::vector<int> Es, std::vector<int> libraries,
                      int k, int numReps, int crossfold, bool explore, bool full, bool saveFinalPredictions,
                      bool saveSMAPCoeffs, bool copredictMode, std::vector<bool> usable, std::vector<double> co_x,
                      std::vector<bool> coTrainingRows, std::vector<bool> coPredictionRows, std::string rngState,
                      double nextRV, IO* io, bool keep_going(), void all_tasks_finished(void));

std::queue<Prediction>& get_results();