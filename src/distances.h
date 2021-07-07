#pragma once

#include "manifold.h"

std::unique_ptr<double[]> cost_matrix(const Manifold& M, const Manifold& Mp, int i, int j, double gamma,
                                      double missingDistance, int& len_i, int& len_j);

double wasserstein(double* C, int len_i, int len_j);