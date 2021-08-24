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

  double* data() const { return _flat.get(); };
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
  bool _xmap_mode;
  bool _allow_missing;
  int _tau;
  int _p;
  int _num_extras, _num_extras_lagged;
  std::vector<double> _x, _xmap, _co_x, _t;
  std::vector<std::vector<double>> _extras;
  std::vector<int> _panel_ids;

  std::vector<int> _observation_number;

  void setup_observation_numbers();
  void fill_in_point(int i, int E, bool copredict, bool prediction, double dtWeight, double* point,
                     double& target) const;

  bool find_observation_num(int target, int& k, int direction, int panel) const;
  std::vector<int> get_lagged_indices(int startIndex, int E, int panel) const;

public:
  double calculate_time_increment() const;
  int get_observation_num(int i) const { return _observation_number[i]; }

  friend void to_json(json& j, const ManifoldGenerator& g);
  friend void from_json(const json& j, ManifoldGenerator& g);

  ManifoldGenerator() = default;

  ManifoldGenerator(const std::vector<double>& t, const std::vector<double>& x, int tau, int p,
                    const std::vector<double>& xmap = {}, const std::vector<double>& co_x = {},
                    const std::vector<int>& panelIDs = {}, const std::vector<std::vector<double>>& extras = {},
                    int numExtrasLagged = 0, bool dtMode = false, bool dt0 = false, bool cumulativeDT = false,
                    bool allowMissing = false)
    : _t(t)
    , _x(x)
    , _tau(tau)
    , _p(p)
    , _xmap(xmap)
    , _co_x(co_x)
    , _panel_ids(panelIDs)
    , _extras(extras)
    , _num_extras((int)extras.size())
    , _num_extras_lagged(numExtrasLagged)
    , _use_dt(dtMode)
    , _add_dt0(dt0)
    , _cumulative_dt(cumulativeDT)
    , _allow_missing(allowMissing)
  {
    _panel_mode = (panelIDs.size() > 0);
    _xmap_mode = (xmap.size() > 0);
    setup_observation_numbers();
  }

  Manifold create_manifold(int E, const std::vector<bool>& filter, bool copredict, bool prediction,
                           double dtWeight = 0.0, bool skipMissing = false) const;

  std::vector<bool> generate_usable(int maxE) const;

  int E_dt(int E) const { return (_use_dt) * (E - 1 + _add_dt0); }
  int E_extras(int E) const { return _num_extras + _num_extras_lagged * (E - 1); }
  int E_actual(int E) const { return E + E_dt(E) + E_extras(E); }

  int numExtrasLagged() const { return _num_extras_lagged; }
  int numExtras() const { return _num_extras; }
};
