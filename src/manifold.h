#pragma once

#include <memory>
#include <vector>

class Manifold
{
  std::unique_ptr<double[]> _flat;
  std::vector<double> _y;
  size_t _nobs, _E_x, _E_dt, _E_extras, _E_actual;
  double _missing;

public:
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
  double dt(size_t i, size_t j) const { return _flat[i * _E_actual + (j - _E_x)]; }
  double extras(size_t i, size_t j) const { return _flat[i * _E_actual + (j - _E_x - _E_dt)]; }
  bool any_missing(size_t obsNum) const
  {
    if (_y[obsNum] == _missing) {
      return true;
    }

    bool missing = false;
    for (size_t j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) == _missing) {
        missing = true;
        break;
      }
    }
    return missing;
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
  bool _copredict, _use_dt;
  int _tau;
  double _dtWeight, _missing;
  size_t _nobs, _E_x, _E_dt, _E_extras, _E_actual;

  std::vector<double> _x, _y, _co_x, _t;
  std::vector<std::vector<double>> _extras;

  double find_x(std::vector<size_t> inds, size_t i, size_t j) const;
  double find_co_x(std::vector<size_t> inds, size_t i, size_t j) const;
  double find_dt(std::vector<size_t> inds, size_t i, size_t j) const;
  double find_extras(std::vector<size_t> inds, size_t i, size_t j) const;

public:
  ManifoldGenerator(std::vector<double> x, std::vector<double> y, std::vector<double> co_x,
                    std::vector<std::vector<double>> extras, std::vector<double> t, size_t E, double dtWeight,
                    double missing, size_t tau = 1);

  Manifold create_manifold(std::vector<bool> filter, bool prediction);
  size_t E_actual() const { return _E_actual; }
};