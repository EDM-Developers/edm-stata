#pragma once

#include <memory>
#include <vector>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

class Manifold
{
  std::unique_ptr<double[]> _flat = nullptr;
  std::vector<double> _y;
  int _nobs, _E_x, _E_dt, _E_extras, _E_actual;
  double _missing;

public:
  Manifold(){};

  Manifold(std::unique_ptr<double[]>& flat, std::vector<double> y, int nobs, int E_x, int E_dt, int E_extras,
           int E_actual, double missing)
    : _flat(std::move(flat))
    , _y(y)
    , _nobs(nobs)
    , _E_x(E_x)
    , _E_dt(E_dt)
    , _E_extras(E_extras)
    , _E_actual(E_actual)
    , _missing(missing)
  {}

  double operator()(int i, int j) const { return _flat[i * _E_actual + j]; }
  double* obs(size_t i) const { return &_flat[i * _E_actual]; }

  double x(int i, int j) const { return _flat[i * _E_actual + j]; }
  double dt(int i, int j) const { return _flat[i * _E_actual + _E_x + j]; }
  double extras(int i, int j) const { return _flat[i * _E_actual + _E_x + _E_dt + j]; }
  bool any_missing(int obsNum) const
  {
    for (int j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) == _missing) {
        return true;
      }
    }
    return false;
  }

  bool any_not_missing(int obsNum) const
  {
    for (int j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) != _missing) {
        return true;
      }
    }
    return false;
  }

  double y(int i) const { return _y[i]; }
  int ySize() const { return (int)_y.size(); }

  int nobs() const { return _nobs; }
  int E() const { return _E_x; }
  int E_dt() const { return _E_dt; }
  int E_extra() const { return _E_extras; }
  int E_actual() const { return _E_actual; }
};

class ManifoldGenerator
{
private:
  bool _copredict = false;
  bool _use_dt = false;
  bool _add_dt0 = false;
  int _tau;
  double _missing;
  int _num_extras, _num_extras_varying;

  double lagged(const std::vector<double>& vec, const std::vector<int>& inds, int i, int j) const;
  double find_dt(const std::vector<int>& inds, int i, int j) const;

  double _dtWeight = 0.0;
  std::vector<double> _x, _y, _co_x, _t;
  std::vector<std::vector<double>> _extras;
  std::vector<bool> _extrasEVarying;

public:
  friend void to_json(json& j, const ManifoldGenerator& g);
  friend void from_json(const json& j, ManifoldGenerator& g);

  ManifoldGenerator(){};

  ManifoldGenerator(const std::vector<double>& x, const std::vector<double>& y,
                    const std::vector<std::vector<double>>& extras, const std::vector<bool>& extrasEVarying,
                    double missing, int tau)
    : _x(x)
    , _y(y)
    , _extras(extras)
    , _extrasEVarying(extrasEVarying)
    , _missing(missing)
    , _tau(tau)
  {
    _num_extras = (int)extras.size();
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

  Manifold create_manifold(int E, const std::vector<bool>& filter, bool prediction) const;

  int E_dt(int E) const { return (_use_dt) * (E - 1 + _add_dt0); }
  int E_extras(int E) const { return _num_extras + _num_extras_varying * (E - 1); }
  int E_actual(int E) const { return E + E_dt(E) + E_extras(E); }
};
