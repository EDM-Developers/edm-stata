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
#include <vector>

typedef int retcode;

typedef struct
{
  std::vector<double> flat;
  int rows, cols;
} manifold_t;

struct BManifold
{
  std::vector<double> flat;
  size_t _rows, _cols;

  double operator()(size_t i, size_t j) const { return flat[i * _cols + j]; }

};


template<class T> void ignore( const T& ) { }

class Manifold
{
private:
  std::vector<double> _data, _dt;
  std::vector<size_t> _ind;
  std::vector<std::vector<double>> _extras;
  size_t _nobs, _E, _E_actual;
  bool _use_dt;

public:
  Manifold(std::vector<double> data, std::vector<bool> useRow, size_t E, std::vector<double> dt, std::vector<std::vector<double>> extras) : _data(data), _dt(dt), _E(E), _extras(extras) {
    _nobs = 0;
    for (int row = 0; row < useRow.size(); row++) {
      if (useRow[row]) {
        _ind.push_back(row);
        _nobs += 1;
      }
    }
    
    _use_dt = (dt.size() > 0);
    _E_actual = E + _use_dt*(E-1) + extras.size();
  }

  double operator()(size_t i, size_t j) const
  {
    try {
    if (j < _E) {
      return _data.at(_ind[i] - j);
    } else if (_use_dt && j < 2*_E-1) {
      return _dt.at(_ind[i] - (j - _E));
    } else {
      return _extras.at(j-_E-_use_dt*_E).at(i);
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
  size_t E() const { return _E; }
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

DLL smap_res_t mf_smap_loop(smap_opts_t opts, const std::vector<double>& y, const manifold_t& M, const manifold_t& Mp,
                            int nthreads, const IO& io, bool keep_going() = nullptr, void finished() = nullptr);
