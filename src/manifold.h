#pragma once

/* global variable placeholder for missing values */
const double MISSING = 1.0e+100;

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

class Manifold
{
  std::shared_ptr<double[]> _flat = nullptr;
  std::vector<double> _y;
  std::vector<int> _panel_ids;
  int _nobs, _E_x, _E_dt, _E_extras, _E_lagged_extras, _E_actual;

public:
  Manifold(std::unique_ptr<double[]>& flat, std::vector<double> y, std::vector<int> panelIDs, int nobs, int E_x,
           int E_dt, int E_extras, int E_lagged_extras, int E_actual)
    : _flat(std::move(flat))
    , _y(y)
    , _panel_ids(panelIDs)
    , _nobs(nobs)
    , _E_x(E_x)
    , _E_dt(E_dt)
    , _E_extras(E_extras)
    , _E_lagged_extras(E_lagged_extras)
    , _E_actual(E_actual)
  {}

  double operator()(int i, int j) const { return _flat[i * _E_actual + j]; }

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map() const
  {
    return { _flat.get(), _nobs, _E_actual };
  }

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> laggedObsMap(
    int obsNum) const
  {
    int numLaggedExtras = _E_lagged_extras / _E_x;
    return { &(_flat[obsNum * _E_actual]), 1 + (_E_dt > 0) + numLaggedExtras, _E_x };
  }

  Eigen::Map<const Eigen::VectorXd> yMap() const { return { &(_y[0]), _nobs }; }

  double x(int i, int j) const { return _flat[i * _E_actual + j]; }
  double dt(int i, int j) const { return _flat[i * _E_actual + _E_x + j]; }
  double extras(int i, int j) const { return _flat[i * _E_actual + _E_x + _E_dt + j]; }
  int panel(int i) const { return _panel_ids[i]; }

  double unlagged_extras(int obsNum, int varNum) const
  {
    int ind = obsNum * _E_actual + _E_x + _E_dt + _E_lagged_extras + varNum;
    return _flat[ind];
  }

  double range() const
  {
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();

    for (int i = 0; i < _nobs * _E_actual; i++) {
      if (_flat[i] != MISSING) {
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

  double missing() const { return MISSING; }

  bool any_missing(int obsNum) const
  {
    for (int j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) == MISSING) {
        return true;
      }
    }
    return false;
  }

  bool any_not_missing(int obsNum) const
  {
    for (int j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) != MISSING) {
        return true;
      }
    }
    return false;
  }

  int num_not_missing(int obsNum) const
  {
    int count = 0;
    for (int j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) != MISSING) {
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
  int E_lagged_extras() const { return _E_lagged_extras; }
  int E_extras() const { return _E_extras; }
  int E_actual() const { return _E_actual; }
};

class ManifoldGenerator
{
private:
  bool _use_dt;
  bool _add_dt0;
  bool _cumulative_dt;
  bool _panel_mode;
  bool _allow_missing;
  int _tau;
  int _p;
  int _num_extras, _num_extras_lagged;
  double _dtWeight = 0.0;
  std::vector<double> _x, _y, _co_x, _t;
  std::vector<std::vector<double>> _extras;
  std::vector<int> _panel_ids;

  std::vector<int> _observation_number;

  double lagged(const std::vector<double>& vec, const std::vector<int>& inds, int i, int j) const;
  double find_dt(const std::vector<int>& inds, int i, int j) const;

  bool find_observation_num(int target, int& k, int direction, int panel) const;
  std::vector<int> get_lagged_indices(int i, int startIndex, int E, int panel) const;

public:
  double calculate_time_increment() const;
  int get_observation_num(int i);

  friend void to_json(json& j, const ManifoldGenerator& g);
  friend void from_json(const json& j, ManifoldGenerator& g);

  ManifoldGenerator() = default;

  ManifoldGenerator(const std::vector<double>& t, const std::vector<double>& x, const std::vector<double>& y, int tau,
                    int p, const std::vector<double>& co_x = {}, const std::vector<int>& panelIDs = {},
                    const std::vector<std::vector<double>>& extras = {}, int numExtrasLagged = 0, double dtWeight = 0.0,
                    bool dt0 = false, bool cumulativeDT = false, bool allowMissing = false)
    : _t(t)
    , _x(x)
    , _y(y)
    , _tau(tau)
    , _p(p)
    , _co_x(co_x)
    , _panel_ids(panelIDs)
    , _extras(extras)
    , _num_extras((int)extras.size())
    , _num_extras_lagged(numExtrasLagged)
    , _dtWeight(dtWeight)
    , _add_dt0(dt0)
    , _cumulative_dt(cumulativeDT)
    , _allow_missing(allowMissing)
  {

    if (panelIDs.size() > 0) {
      _panel_mode = true;
    }

    if (dtWeight == 0.0) {
      _use_dt = false;

      double unit = calculate_time_increment();
      double minT = *std::min_element(t.begin(), t.end());

      // Create a time index which is a discrete count of the number of 'unit' time units.
      for (int i = 0; i < t.size(); i++) {
        if (t[i] != MISSING) {
          _observation_number.push_back(std::round((t[i] - minT) / unit));
        } else {
          _observation_number.push_back(-1);
        }
      }
    } else {
      _use_dt = true;

      int countUp = 0;
      for (int i = 0; i < _t.size(); i++) {
        if (_t[i] != MISSING && (allowMissing || (_x[i] != MISSING))) {
          _observation_number.push_back(countUp);
          countUp += 1;
        } else {
          _observation_number.push_back(-1);
        }
      }
    }
  }

  Manifold create_manifold(int E, const std::vector<bool>& filter, bool copredict, bool prediction,
                           bool skipMissing = false) const;

  std::vector<bool> generate_usable(const std::vector<bool>& touse, int maxE) const;

  int E_dt(int E) const { return (_use_dt) * (E - 1 + _add_dt0); }
  int E_extras(int E) const { return _num_extras + _num_extras_lagged * (E - 1); }
  int E_actual(int E) const { return E + E_dt(E) + E_extras(E); }

  int numExtrasLagged() const { return _num_extras_lagged; }
  int numExtras() const { return _num_extras; }
};