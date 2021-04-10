#pragma once

#include "common.h"

std::future<void> edm_async(Options opts, ManifoldGenerator generator, size_t E, std::vector<bool> trainingRows,
                            std::vector<bool> predictionRows, IO* io, Prediction* pred, bool keep_going() = nullptr,
                            void all_tasks_finished(void) = nullptr);

void edm_task(Options opts, ManifoldGenerator generator, size_t E, std::vector<bool> trainingRows,
              std::vector<bool> predictionRows, IO* io, Prediction* pred, bool keep_going() = nullptr,
              void all_tasks_finished(void) = nullptr, bool serial = false);