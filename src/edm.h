#pragma once

#ifdef _MSC_VER
#define DLL extern __declspec(dllexport)
#else
#define DLL
#endif

#define SUCCESS 0
#define TOO_FEW_VARIABLES 102
#define TOO_MANY_VARIABLES 103
#define INVALID_ALGORITHM 400
#define INSUFFICIENT_UNIQUE 503
#define NOT_IMPLEMENTED 908
#define UNKNOWN_ERROR 8000

/* global variable placeholder for missing values */
#define MISSING 1.0e+100

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

typedef int retcode;

template<class T>
void ignore(const T&)
{}

class Manifold
{
private:
  bool _use_dt;
  double _dtweight;
  size_t _nobs, _E_x, _E_dt, _E_extras, _E_actual;
  std::vector<double> _x;
  std::vector<int> _t;
  std::vector<std::vector<double>> _extras;
  std::unordered_map<size_t, int> _obsNumToTime;
  std::unordered_map<int, size_t> _timeToIndex;

public:
  Manifold(std::vector<double> x, std::vector<bool> useRow, size_t E, std::vector<int> t, double dtweight,
           std::vector<std::vector<double>> extras)
    : _x(x)
    , _t(t)
    , _dtweight(dtweight)
    , _extras(extras)
  {
    size_t obsNum = 0;
    for (size_t i = 0; i < useRow.size(); i++) {
      if (useRow[i]) {
        _obsNumToTime[obsNum] = t[i];
        obsNum += 1;
      }
      _timeToIndex[t[i]] = i;
    }

    _nobs = obsNum;

    _use_dt = (dtweight > 0);
    _E_x = E;
    _E_dt = (_use_dt) * (E - 1);
    _E_extras = extras.size();
    _E_actual = _E_x + _E_dt + _E_extras;
  }

  double operator()(size_t i, size_t j) const
  {
    try {
      int referenceTime = _obsNumToTime.at(i);
      size_t index;

      if (j < _E_x) {

        if (_use_dt) {
          index = _timeToIndex.at(referenceTime) - j;
        } else {
          index = _timeToIndex.at(referenceTime - j);
        }
        return _x.at(index);
      } else if (j < _E_x + _E_dt) {
        j -= _E_x;
        index = _timeToIndex.at(referenceTime) - j;
        return _dtweight * (_t.at(index) - _t.at(index - 1));
      } else {
        j -= (_E_x + _E_dt);
        index = _timeToIndex.at(referenceTime);
        return _extras.at(j).at(index);
      }

    } catch (const std::out_of_range& e) {
      ignore(e);
      return MISSING;
    }
  }

  bool any_missing(size_t obsNum) const
  {
    bool missing = false;
    for (size_t j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) == MISSING) {
        missing = true;
        break;
      }
    }
    return missing;
  }

  size_t nobs() const { return _nobs; }
  size_t E() const { return _E_x; }
  size_t E_actual() const { return _E_actual; }
};

struct smap_opts_t
{
  bool force_compute, save_mode;
  int l, varssv;
  double theta, missingdistance;
  std::string algorithm;
  bool distributeThreads = false;
};

class IO
{
public:
  int verbosity = 0;

  virtual void print(std::string s) const
  {
    if (verbosity > 0) {
      out(s.c_str());
    }
  }

  virtual void print_async(std::string s) const
  {
    if (verbosity > 0) {
      out_async(s.c_str());
    }
  }

  virtual void progress_bar(double progress) const
  {
    if (progress == 0.0) {
      print_async("Percent complete: 0");
      nextMessage = 1.0 / 40;
      dots = 0;
      tens = 0;
      return;
    }

    while (progress >= nextMessage) {
      if (dots < 3) {
        print_async(".");
        dots += 1;
      } else {
        tens += 1;
        print_async(std::to_string(tens * 10));
        dots = 0;
      }
      nextMessage += 1.0 / 40;
    }
  }
  mutable int dots, tens;
  mutable double nextMessage;

  // Actual implementation of IO functions are in the subclasses
  virtual void out(const char*) const = 0;
  virtual void out_async(const char*) const = 0;
  virtual void error(const char*) const = 0;
  virtual void flush() const = 0;
};

typedef struct
{
  retcode rc;
  std::vector<double> ystar;
  std::optional<std::vector<double>> flat_Bi_map;
} smap_res_t;

DLL smap_res_t mf_smap_loop(smap_opts_t opts, const std::vector<double>& y, const Manifold& M, const Manifold& Mp,
                            int nthreads, const IO& io, bool keep_going() = nullptr, void finished() = nullptr);
