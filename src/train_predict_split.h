#pragma once

#include <vector>

#include "mersennetwister.h"
#include "stats.h"

class TrainPredictSplitter
{
private:
  bool _explore, _full;
  int _crossfold, _numObsUsable;
  std::vector<bool> _usable;
  std::vector<int> _crossfoldURank;
  MtRng64 _rng;

public:
  TrainPredictSplitter() = default;

  TrainPredictSplitter(bool explore, bool full, int crossfold, std::vector<bool> usable)
    : _explore(explore)
    , _full(full)
    , _crossfold(crossfold)
    , _usable(usable)
    , _rng(1)
  {
    _numObsUsable = std::accumulate(usable.begin(), usable.end(), 0);
  };

  TrainPredictSplitter(bool explore, bool full, int crossfold, std::vector<bool> usable, const std::string& rngState,
                       double nextRV)
    : _explore(explore)
    , _full(full)
    , _crossfold(crossfold)
    , _usable(usable)
  {
    // Sync the local random number generator with Stata's
    set_rng_state(rngState, nextRV);

    _numObsUsable = std::accumulate(usable.begin(), usable.end(), 0);

    if (crossfold > 0) {
      std::vector<double> u;

      for (int i = 0; i < _numObsUsable; i++) {
        u.push_back(_rng.getReal2());
      }

      _crossfoldURank = rank(u);
    }
  }

  bool requiresRandomNumbersEachTask() const { return (_crossfold == 0) && !_full; }
  static bool requiresRandomNumbers(int crossfold, bool full) { return crossfold > 0 || !full; }

  void set_rng_state(const std::string& rngState, double nextRV)
  {
    unsigned long long state[312];

    // Set up the rng at the beginning on this batch (given by the 'state' array)
    for (int i = 0; i < 312; i++) {
      state[i] = std::stoull(rngState.substr(3 + i * 16, 16), nullptr, 16);
      _rng.state_[i] = state[i];
    }

    _rng.left_ = 312;
    _rng.next_ = _rng.state_;

    // Go through this batch of rv's and find the closest to the
    // observed 'nextRV'
    int bestInd = -1;
    double minDist = 1.0;

    for (int i = 0; i < 320; i++) {
      double dist = std::abs(_rng.getReal2() - nextRV);
      if (dist < minDist) {
        minDist = dist;
        bestInd = i;
      }
    }

    // Reset the state to the beginning on this batch
    for (int i = 0; i < 312; i++) {
      _rng.state_[i] = state[i];
    }

    _rng.left_ = 312;
    _rng.next_ = _rng.state_;

    // Burn all the rv's which are already used
    for (int i = 0; i < bestInd; i++) {
      _rng.getReal2();
    }
  }

  // Assuming this is called in explore mode
  int next_training_size(int crossfoldIter) const
  {
    int trainSize = 0;
    if (_crossfold > 0) {
      for (int obsNum = 0; obsNum < _numObsUsable; obsNum++) {
        if ((obsNum + 1) % _crossfold != (crossfoldIter - 1)) {
          trainSize += 1;
        }
      }
      return trainSize;
    } else if (_full) {
      return _numObsUsable;
    } else {
      return _numObsUsable / 2;
    }
  }

  std::pair<std::vector<bool>, std::vector<bool>> train_predict_split(int library, int crossfoldIter)
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

    std::vector<double> u;

    for (int i = 0; i < _numObsUsable; i++) {
      u.push_back(_rng.getReal2());
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
