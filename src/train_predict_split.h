#pragma once

#include <vector>

#include "mersennetwister.h"
#include "stats.h"

class TrainPredictSplitter
{
private:
  bool _explore, _full;
  int _crossfold;
  std::vector<bool> _usable;
  std::vector<int> _crossfoldURank;
  MtRng64 rng;

public:
  TrainPredictSplitter() {}
  TrainPredictSplitter(bool explore, bool full, int crossfold, std::vector<bool> usable)
    : _explore(explore)
    , _full(full)
    , _crossfold(crossfold)
    , _usable(usable)
    , rng(1)
  {}

  void add_crossfold_rvs(std::vector<double> crossfoldU) { _crossfoldURank = rank(remove_value(crossfoldU, MISSING)); }

  bool requiresRandomNumbersEachTask() { return (_crossfold == 0) && !_full; }
  bool requiresCrossFoldRandomNumbers() { return _crossfold > 0; }
  bool requiresRandomNumbers() { return requiresRandomNumbersEachTask() || requiresCrossFoldRandomNumbers(); }

  void set_rng_state(std::string rngState, double nextRV)
  {
    unsigned long long state[312];

    // Set up the rng at the beginning on this batch (given by the 'state' array)
    for (int i = 0; i < 312; i++) {
      state[i] = std::stoull(rngState.substr(3 + i * 16, 16), nullptr, 16);
      rng.state_[i] = state[i];
    }

    rng.left_ = 312;
    rng.next_ = rng.state_;

    // Go through this batch of rv's and find the closest to the
    // observed 'nextRV'
    int bestInd = -1;
    double minDist = 1.0;

    for (int i = 0; i < 320; i++) {
      double dist = std::abs(rng.getReal2() - nextRV);
      if (dist < minDist) {
        minDist = dist;
        bestInd = i;
      }
    }

    // Reset the state to the beginning on this batch
    for (int i = 0; i < 312; i++) {
      rng.state_[i] = state[i];
    }

    rng.left_ = 312;
    rng.next_ = rng.state_;

    // Burn all the rv's which are already used
    for (int i = 0; i < bestInd; i++) {
      rng.getReal2();
    }
  }

  std::pair<std::vector<bool>, std::vector<bool>> train_predict_split(std::vector<double> uWithMissing, int library,
                                                                      int crossfoldIter)
  {
    if (_explore && _full) {
      return { _usable, _usable };
    }

    std::vector<bool> trainingRows(_usable.size()), predictionRows(_usable.size());

    if (_explore && _crossfold > 0) {
      int obsNum = 0;
      for (int i = 0; i < trainingRows.size(); i++) {
        if (_usable[i]) {
          if (_crossfoldURank[obsNum] % _crossfold == (crossfoldIter - 1)) {
            trainingRows[i] = false;
            predictionRows[i] = true;
          } else {
            trainingRows[i] = true;
            predictionRows[i] = false;
          }
          obsNum += 1;
        } else {
          trainingRows[i] = false;
          predictionRows[i] = false;
        }
      }

      return { trainingRows, predictionRows };
    }

    std::vector<double> uStata = remove_value(uWithMissing, MISSING);
    std::vector<double> u;

    for (int i = 0; i < uStata.size(); i++) {
      u.push_back(rng.getReal2());
    }

    if (_explore) {
      double med = median(u);

      int obsNum = 0;
      for (int i = 0; i < trainingRows.size(); i++) {
        if (_usable[i]) {
          if (u[obsNum] < med) {
            trainingRows[i] = true;
            predictionRows[i] = false;
          } else {
            trainingRows[i] = false;
            predictionRows[i] = true;
          }
          obsNum += 1;
        } else {
          trainingRows[i] = false;
          predictionRows[i] = false;
        }
      }
    } else {
      double uCutoff = 1.0;
      if (library < u.size()) {
        std::vector<double> uCopy(u);
        const auto uCutoffIt = uCopy.begin() + library;
        std::nth_element(uCopy.begin(), uCutoffIt, uCopy.end());
        uCutoff = *uCutoffIt;
      }

      int obsNum = 0;
      for (int i = 0; i < trainingRows.size(); i++) {
        if (_usable[i]) {
          predictionRows[i] = true;
          if (u[obsNum] < uCutoff) {
            trainingRows[i] = true;
          } else {
            trainingRows[i] = false;
          }
          obsNum += 1;
        } else {
          trainingRows[i] = false;
          predictionRows[i] = false;
        }
      }
    }

    return { trainingRows, predictionRows };
  }
};