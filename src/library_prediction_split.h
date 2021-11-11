#pragma once

#include <vector>

#include "mersennetwister.h"
#include "stats.h"

class LibraryPredictionSetSplitter
{
private:
  bool _explore, _full, _shuffle;
  int _crossfold, _numObsUsable;
  std::vector<bool> _usable;
  std::vector<bool> _libraryRows, _predictionRows;
  std::vector<int> _crossfoldURank;
  MtRng64 _rng;

public:
  LibraryPredictionSetSplitter(bool explore, bool full, bool shuffle, int crossfold, std::vector<bool> usable,
                               const std::string& rngState)
    : _explore(explore)
    , _full(full)
    , _shuffle(shuffle)
    , _crossfold(crossfold)
    , _usable(usable)
  {
    if (!rngState.empty() && shuffle) {
      // Sync the local random number generator with Stata's
      set_rng_state(rngState);
    } else {
      _rng.init((unsigned long long)0);
    }

    _numObsUsable = std::accumulate(usable.begin(), usable.end(), 0);

    if (crossfold > 0) {
      if (shuffle) {
        std::vector<double> u;

        for (int i = 0; i < _numObsUsable; i++) {
          u.push_back(_rng.getReal2());
        }

        _crossfoldURank = rank(u);
      } else {
        for (int i = 0; i < _numObsUsable; i++) {
          _crossfoldURank.push_back(i + 1);
        }
      }
    }
  }

  static bool requiresRandomNumbers(int crossfold, bool full) { return crossfold > 0 || !full; }

  void set_rng_state(const std::string& rngState)
  {
    unsigned long long state[312];

    // Set up the rng at the beginning on this batch (given by the 'state' array)
    for (int i = 0; i < 312; i++) {
      state[i] = std::stoull(rngState.substr(3 + i * 16, 16), nullptr, 16);
      _rng.state_[i] = state[i];
    }

    _rng.left_ = 312;
    _rng.next_ = _rng.state_;

    // Burn all the rv's which are already used
    std::string countStr = rngState.substr(3 + 312 * 16 + 4, 8);
    long long numUsed = std::stoull(countStr, nullptr, 16);

    for (int i = 0; i < numUsed; i++) {
      _rng.getReal2();
    }
  }

  // Assuming this is called in explore mode
  int next_library_size(int crossfoldIter) const
  {
    int librarySize = 0;
    if (_crossfold > 0) {
      for (int obsNum = 0; obsNum < _numObsUsable; obsNum++) {
        if ((obsNum + 1) % _crossfold != (crossfoldIter - 1)) {
          librarySize += 1;
        }
      }
      return librarySize;
    } else if (_full) {
      return _numObsUsable;
    } else {
      return _numObsUsable / 2;
    }
  }

  std::vector<bool> libraryRows() const { return _libraryRows; }
  std::vector<bool> predictionRows() const { return _predictionRows; }

  void update_library_prediction_split(int library, int crossfoldIter)
  {
    if (_explore && _full) {
      _libraryRows = _usable;
      _predictionRows = _usable;
    } else if (_explore && _crossfold > 0) {
      crossfold_split(crossfoldIter);
    } else if (_explore) {
      half_library_prediction_split();
    } else {
      fixed_size_library(library);
    }

    int numInLibrarySet = 0, numInPredictionSet = 0;
    for (int i = 0; i < _libraryRows.size(); i++) {
      if (_libraryRows[i]) {
        numInLibrarySet += 1;
      }
      if (_predictionRows[i]) {
        numInPredictionSet += 1;
      }
    }
    assert(numInLibrarySet > 0);
    assert(numInPredictionSet > 0);
  }

  void crossfold_split(int crossfoldIter)
  {
    int obsNum = 0;
    int sizeOfEachFold = std::round(((float)_numObsUsable) / _crossfold);

    _libraryRows = std::vector<bool>(_usable.size());
    _predictionRows = std::vector<bool>(_usable.size());

    for (int i = 0; i < _libraryRows.size(); i++) {
      if (_usable[i]) {
        if (_shuffle) {
          if (_crossfoldURank[obsNum] % _crossfold == (crossfoldIter - 1)) {
            _libraryRows[i] = false;
            _predictionRows[i] = true;
          } else {
            _libraryRows[i] = true;
            _predictionRows[i] = false;
          }
        } else {
          int foldGroup = obsNum / sizeOfEachFold;
          if ((crossfoldIter - 1) == foldGroup) {
            _libraryRows[i] = false;
            _predictionRows[i] = true;
          } else {
            _libraryRows[i] = true;
            _predictionRows[i] = false;
          }
        }
        obsNum += 1;
      } else {
        _libraryRows[i] = false;
        _predictionRows[i] = false;
      }
    }
  }

  void half_library_prediction_split()
  {
    _libraryRows = std::vector<bool>(_usable.size());
    _predictionRows = std::vector<bool>(_usable.size());

    if (_shuffle) {
      std::vector<double> u;
      for (int i = 0; i < _numObsUsable; i++) {
        u.push_back(_rng.getReal2());
      }

      double med = median(u);

      int obsNum = 0;
      for (int i = 0; i < _libraryRows.size(); i++) {
        if (_usable[i]) {
          if (u[obsNum] < med) {
            _libraryRows[i] = true;
            _predictionRows[i] = false;
          } else {
            _libraryRows[i] = false;
            _predictionRows[i] = true;
          }
          obsNum += 1;
        } else {
          _libraryRows[i] = false;
          _predictionRows[i] = false;
        }
      }
    } else {
      int librarySize = _numObsUsable / 2;

      int obsNum = 0;
      for (int i = 0; i < _libraryRows.size(); i++) {
        if (_usable[i]) {
          if (obsNum < librarySize) {
            _libraryRows[i] = true;
            _predictionRows[i] = false;
          } else {
            _libraryRows[i] = false;
            _predictionRows[i] = true;
          }
          obsNum += 1;
        } else {
          _libraryRows[i] = false;
          _predictionRows[i] = false;
        }
      }
    }
  }

  void fixed_size_library(int library)
  {
    _libraryRows = std::vector<bool>(_usable.size());
    _predictionRows = _usable;

    if (_shuffle) {
      std::vector<double> u;
      for (int i = 0; i < _numObsUsable; i++) {
        u.push_back(_rng.getReal2());
      }

      double uCutoff = 1.0;
      if (library < u.size()) {
        std::vector<double> uCopy(u);
        const auto uCutoffIt = uCopy.begin() + library;
        std::nth_element(uCopy.begin(), uCutoffIt, uCopy.end());
        uCutoff = *uCutoffIt;
      }

      int obsNum = 0;
      for (int i = 0; i < _libraryRows.size(); i++) {
        if (_usable[i]) {
          _predictionRows[i] = true;
          if (u[obsNum] < uCutoff) {
            _libraryRows[i] = true;
          } else {
            _libraryRows[i] = false;
          }
          obsNum += 1;
        } else {
          _libraryRows[i] = false;
        }
      }
    } else {
      int obsNum = 0;
      for (int i = 0; i < _libraryRows.size(); i++) {
        if (_usable[i]) {
          if (obsNum < library) {
            _libraryRows[i] = true;
          } else {
            _libraryRows[i] = false;
          }
          obsNum += 1;
        } else {
          _libraryRows[i] = false;
          ;
        }
      }
    }

    int numInLibrary = 0;
    for (int i = 0; i < _libraryRows.size(); i++) {
      if (_libraryRows[i]) {
        numInLibrary += 1;
      }
    }

    if (library < _numObsUsable) {
      assert(numInLibrary == library);
    } else {
      assert(numInLibrary == _numObsUsable);
    }
  }
};
