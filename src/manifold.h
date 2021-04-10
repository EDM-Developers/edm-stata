#pragma once

#include <memory>
#include <vector>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

class Manifold
{
  std::unique_ptr<double[]> _flat = nullptr;
  std::vector<double> _y;
  size_t _nobs, _E_x, _E_dt, _E_extras, _E_actual;
  double _missing;

public:
  Manifold(){};

  Manifold(std::unique_ptr<double[]>& flat, std::vector<double> y, size_t nobs, size_t E_x, size_t E_dt,
           size_t E_extras, size_t E_actual, double missing)
    : _flat(std::move(flat))
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
  bool any_missing(size_t obsNum) const
  {
    for (size_t j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) == _missing) {
        return true;
      }
    }
    return false;
  }

  bool any_not_missing(size_t obsNum) const
  {
    for (size_t j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) != _missing) {
        return true;
      }
    }
    return false;
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

  double _dtWeight;
  std::vector<double> _x, _y, _co_x, _t;
  std::vector<std::vector<double>> _extras;
  std::vector<bool> _extrasEVarying;

public:
  friend void to_json(json& j, const ManifoldGenerator& g);
  friend void from_json(const json& j, ManifoldGenerator& g);

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
