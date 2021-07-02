#pragma once

#include <vector>

#include "stats.h"

class TrainPredictSplitter
{
private:
  bool _explore, _full;
  int _crossfold;
  std::vector<bool> _usable;
  std::vector<int> _crossfoldURank;

public:
  TrainPredictSplitter() {}
  TrainPredictSplitter(bool explore, bool full, int crossfold, std::vector<bool> usable)
    : _explore(explore)
    , _full(full)
    , _crossfold(crossfold)
    , _usable(usable)
  {}

  void add_crossfold_rvs(std::vector<double> crossfoldU) { _crossfoldURank = rank(remove_value(crossfoldU, MISSING)); }

  bool requiresRandomNumbersEachTask() { return (_crossfold == 0) && !_full; }
  bool requiresCrossFoldRandomNumbers() { return _crossfold > 0; }

  std::pair<std::vector<bool>, std::vector<bool>> train_predict_split(std::vector<double> uWithMissing, int library,
                                                                      int crossfoldIter, MtRng64& rng)
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