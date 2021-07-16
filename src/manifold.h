#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>
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

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map() const
  {
    return { _flat.get(), _nobs, _E_actual };
  }
  Eigen::Map<const Eigen::VectorXd> yMap() const { return { &(_y[0]), _nobs }; }

  double x(int i, int j) const { return _flat[i * _E_actual + j]; }
  double dt(int i, int j) const { return _E_dt ? _flat[i * _E_actual + _E_x + j] : _missing; }
  double extras(int i, int j) const { return _E_extras ? _flat[i * _E_actual + _E_x + _E_dt + j] : _missing; }

  double range() const
  {
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();

    for (int i = 0; i < _nobs * _E_actual; i++) {
      if (_flat[i] != _missing) {
        if (_flat[i] < min) {
          min = _flat[i];
        }
        if (_flat[i] > max) {
          max = _flat[i];
        }
      }
    }
    return max - min;
  }

  double missing() const { return _missing; }

  double time_range() const { return _nobs - 1; }

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

  int num_not_missing(int obsNum) const
  {
    int count = 0;
    for (int j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) != _missing) {
        count += 1;
      }
    }
    return count;
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
  double _dtWeight = 0.0;
  std::vector<double> _x, _y, _co_x, _t;
  std::vector<std::vector<double>> _extras;
  std::vector<bool> _extrasEVarying;

  double lagged(const std::vector<double>& vec, const std::vector<int>& inds, int i, int j) const;
  double find_dt(const std::vector<int>& inds, int i, int j) const;

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

  ManifoldGenerator(const ManifoldGenerator& obj)
  {
    _copredict = obj._copredict;
    _use_dt = obj._use_dt;
    _add_dt0 = obj._add_dt0;
    _tau = obj._tau;
    _missing = obj._missing;

    _num_extras_varying = obj._num_extras_varying;
    _num_extras = obj._num_extras;

    _dtWeight = obj._dtWeight;

    _x = obj._x;
    _y = obj._y;
    _co_x = obj._co_x;
    _t = obj._t;

    _extras.clear();
    for (auto& extra : obj._extras) {
      _extras.push_back(extra);
    }
    _extrasEVarying = obj._extrasEVarying;
  }

  ManifoldGenerator& operator=(const ManifoldGenerator& obj)
  {
    _copredict = obj._copredict;
    _use_dt = obj._use_dt;
    _add_dt0 = obj._add_dt0;
    _tau = obj._tau;
    _missing = obj._missing;

    _num_extras_varying = obj._num_extras_varying;
    _num_extras = obj._num_extras;

    _dtWeight = obj._dtWeight;

    _x = obj._x;
    _y = obj._y;
    _co_x = obj._co_x;
    _t = obj._t;

    _extras.clear();
    for (auto& extra : obj._extras) {
      _extras.push_back(extra);
    }
    _extrasEVarying = obj._extrasEVarying;
    return *this;
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

  std::vector<int> E_extras_counts(int E) const
  {
    std::vector<int> E_counts;

    for (const bool& eVarying : _extrasEVarying) {
      E_counts.push_back(eVarying ? E : 1);
    }
    return E_counts;
  };
};
