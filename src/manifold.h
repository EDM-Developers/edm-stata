#pragma once

#include <memory>
#include <vector>

// #include <boost/dynamic_bitset.hpp>
#include <bitset>

const size_t MAX_MANIFOLD_SIZE = 20;

class Manifold
{

  std::vector<double> _y;
  size_t _nobs, _E_x, _E_dt, _E_extras, _E_actual;
  double _missing;

public:
  std::unique_ptr<double[]> _flat = nullptr;
  // std::unique_ptr<boost::dynamic_bitset<>[]> _missingMasks = nullptr;
  std::unique_ptr<std::bitset<MAX_MANIFOLD_SIZE>[]> _missingMasks = nullptr;

  Manifold(){};

  // Manifold(std::unique_ptr<double[]>& flat, std::unique_ptr<boost::dynamic_bitset<>[]>& missingMasks,
  Manifold(std::unique_ptr<double[]>& flat, std::unique_ptr<std::bitset<MAX_MANIFOLD_SIZE>[]>& missingMasks,
           std::vector<double> y, size_t nobs, size_t E_x, size_t E_dt, size_t E_extras, size_t E_actual,
           double missing)
    : _flat(std::move(flat))
    , _missingMasks(std::move(missingMasks))
    , _y(y)
    , _nobs(nobs)
    , _E_x(E_x)
    , _E_dt(E_dt)
    , _E_extras(E_extras)
    , _E_actual(E_actual)
    , _missing(missing)
  {}

  double operator()(size_t i, size_t j) const { return _flat[i * _E_actual + j]; }

  double x(size_t i, size_t j) const { return _flat[i * _E_actual + j]; }
  double dt(size_t i, size_t j) const { return _flat[i * _E_actual + _E_x + j]; }
  double extras(size_t i, size_t j) const { return _flat[i * _E_actual + _E_x + _E_dt + j]; }
  bool any_missing(size_t obsNum) const { return _missingMasks[obsNum].any(); }

  bool any_not_missing(size_t obsNum) const { return !_missingMasks[obsNum].all(); }

  // boost::dynamic_bitset<> get_missing_mask(size_t obsNum) const { return _missingMasks[obsNum]; }
  std::bitset<MAX_MANIFOLD_SIZE> get_missing_mask(size_t obsNum) const { return _missingMasks[obsNum]; }

  // friend boost::dynamic_bitset<> either_missing(const Manifold & M, const Manifold & Mp, size_t i, size_t j) {
  friend std::bitset<MAX_MANIFOLD_SIZE> either_missing(const Manifold& M, const Manifold& Mp, size_t i, size_t j)
  {
    return M._missingMasks[i] | Mp._missingMasks[j];
  }

  double y(size_t i) const { return _y[i]; }
  size_t ySize() const { return _y.size(); }

  size_t nobs() const { return _nobs; }
  size_t E() const { return _E_x; }
  size_t E_dt() const { return _E_dt; }
  size_t E_extra() const { return _E_extras; }
  size_t E_actual() const { return _E_actual; }
};

class ManifoldGenerator
{
private:
  bool _copredict = false;
  bool _use_dt = false;
  bool _add_dt0 = false;
  int _tau;
  double _missing;
  size_t _nobs;
  size_t _num_extras, _num_extras_varying;

  double lagged(const std::vector<double>& vec, const std::vector<size_t>& inds, size_t i, size_t j) const;
  double find_dt(const std::vector<size_t>& inds, size_t i, size_t j) const;

  // The following variables are normally private, but in the dev mode builds
  // they are made public so we can more easily save them to a dump file.
#if defined(DUMP_INPUT) || defined(DRIVER_MODE)
public:
#endif
  double _dtWeight;
  std::vector<double> _x, _y, _co_x, _t;
  std::vector<std::vector<double>> _extras;
  std::vector<bool> _extrasEVarying;

public:
  ManifoldGenerator(){};

  ManifoldGenerator(const std::vector<double>& x, const std::vector<double>& y,
                    const std::vector<std::vector<double>>& extras, const std::vector<bool>& extrasEVarying,
                    double missing, size_t tau)
    : _x(x)
    , _y(y)
    , _extras(extras)
    , _extrasEVarying(extrasEVarying)
    , _missing(missing)
    , _tau(tau)
  {
    _num_extras = extras.size();
    _num_extras_varying = 0;
    for (const bool& eVarying : extrasEVarying) {
      _num_extras_varying += eVarying;
    }
  }

  void add_coprediction_data(const std::vector<double>& co_x)
  {
    _co_x = co_x;
    _copredict = true;
  }

  void add_dt_data(const std::vector<double>& t, double dtWeight, bool dt0)
  {
    _t = t;
    _dtWeight = dtWeight;
    _use_dt = true;
    _add_dt0 = dt0;
  }

  Manifold create_manifold(size_t E, const std::vector<bool>& filter, bool prediction) const;

  size_t E_dt(size_t E) const { return (_use_dt) * (E - 1 + _add_dt0); }
  size_t E_extras(size_t E) const { return _num_extras + _num_extras_varying * (E - 1); }
  size_t E_actual(size_t E) const { return E + E_dt(E) + E_extras(E); }
};
