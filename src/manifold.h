#pragma once

/* global variable placeholder for missing values */
const double MISSING_D = 1.0e+100;
const float MISSING_F = 1.0e+30;

#include <memory>
#include <utility>
#include <vector>

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>

#include <nlohmann/json.hpp>
#if defined(WITH_ARRAYFIRE)
#include <arrayfire.h>
#endif

using json = nlohmann::json;

#if defined(WITH_ARRAYFIRE)
struct ManifoldOnGPU
{
  af::array mdata;   // shape [_E_actual _numPoints 1 1] - manifold
  af::array targets; // Shape [_numPoints 1 1 1]
  af::array panel;   // Shape [_numPoints 1 1 1] - panel ids
  int numPoints, E_x, E_dt, E_extras, E_lagged_extras, E_actual;
  double missing;
};
#endif
class Manifold
{
public:
  Manifold() = default;

  virtual double operator()(int i, int j) const = 0;

  virtual double dt(int i, int j) const = 0;

  virtual int panel(int i) const = 0;
  // void fill_in_point(int i, int E, bool copredictionMode, bool predictionSet, double dtWeight, double* point,
  //                     double& target) const;
  virtual double unlagged_extras(int obsNum, int varNum) const = 0;

  virtual Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map() const = 0;
  virtual Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> laggedObsMap(
    int obsNum) const = 0;
  virtual Eigen::Map<const Eigen::VectorXd> targetsMap() const = 0;

  virtual double range() const = 0;
  virtual double missing() const = 0;

  virtual double target(int i) const = 0;
  virtual int numTargets() const = 0;
  virtual const std::vector<double>& targets() const = 0;

  virtual double* data() const = 0;
  virtual int numPoints() const = 0;
  virtual int E() const = 0;
  virtual int E_dt() const = 0;
  virtual int E_lagged_extras() const = 0;
  virtual int E_extras() const = 0;
  virtual int E_actual() const = 0;
};

class KeenManifold : public Manifold
{
  std::shared_ptr<double[]> _flat = nullptr;
  std::vector<double> _targets;
  std::vector<int> _panelIDs;
  int _numPoints, _E_x, _E_dt, _E_extras, _E_lagged_extras, _E_actual;

public:
  KeenManifold() = default;

  KeenManifold(std::shared_ptr<double[]>& flat, std::vector<double> targets, std::vector<int> panelIDs, int numPoints,
               int E_x, int E_dt, int E_extras, int E_lagged_extras, int E_actual)
    : _flat(flat)
    , _targets(targets)
    , _panelIDs(panelIDs)
    , _numPoints(numPoints)
    , _E_x(E_x)
    , _E_dt(E_dt)
    , _E_extras(E_extras)
    , _E_lagged_extras(E_lagged_extras)
    , _E_actual(E_actual)
  {}

  virtual double operator()(int i, int j) const { return _flat[i * _E_actual + j]; }

  virtual Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map() const
  {
    return { _flat.get(), _numPoints, _E_actual };
  }

  virtual Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> laggedObsMap(
    int obsNum) const
  {
    int numLaggedExtras = _E_lagged_extras / _E_x;
    return { &(_flat[obsNum * _E_actual]), 1 + (_E_dt > 0) + numLaggedExtras, _E_x };
  }

  virtual Eigen::Map<const Eigen::VectorXd> targetsMap() const { return { &(_targets[0]), _numPoints }; }

  double x(int i, int j) const { return _flat[i * _E_actual + j]; }
  virtual double dt(int i, int j) const { return _flat[i * _E_actual + _E_x + j]; }
  double extras(int i, int j) const { return _flat[i * _E_actual + _E_x + _E_dt + j]; }
  virtual int panel(int i) const { return _panelIDs[i]; }

  virtual double unlagged_extras(int obsNum, int varNum) const
  {
    int ind = obsNum * _E_actual + _E_x + _E_dt + _E_lagged_extras + varNum;
    return _flat[ind];
  }

  virtual double range() const
  {
    double min = std::numeric_limits<double>::max();
    double max = std::numeric_limits<double>::min();

    for (int i = 0; i < _numPoints * _E_actual; i++) {
      if (_flat[i] != MISSING_D) {
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

  virtual double missing() const { return MISSING_D; }

  bool any_missing(int obsNum) const
  {
    for (int j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) == MISSING_D) {
        return true;
      }
    }
    return false;
  }

  bool any_not_missing(int obsNum) const
  {
    for (int j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) != MISSING_D) {
        return true;
      }
    }
    return false;
  }

  int num_not_missing(int obsNum) const
  {
    int count = 0;
    for (int j = 0; j < _E_actual; j++) {
      if (operator()(obsNum, j) != MISSING_D) {
        count += 1;
      }
    }
    return count;
  }

  virtual double target(int i) const { return _targets[i]; }
  virtual int numTargets() const { return (int)_targets.size(); }
  virtual const std::vector<double>& targets() const { return _targets; }

  virtual double* data() const { return _flat.get(); };
  virtual int numPoints() const { return _numPoints; }
  virtual int E() const { return _E_x; }
  virtual int E_dt() const { return _E_dt; }
  virtual int E_lagged_extras() const { return _E_lagged_extras; }
  virtual int E_extras() const { return _E_extras; }
  virtual int E_actual() const { return _E_actual; }
  const std::vector<int>& panelIDs() const { return _panelIDs; }
  std::shared_ptr<double[]> flatf64() const { return _flat; }
  std::shared_ptr<double[]> laggedObsMapf64(int obsNum) const
  {
    return std::shared_ptr<double[]>(_flat, _flat.get() + obsNum * _E_actual);
  }

#if defined(WITH_ARRAYFIRE)
  ManifoldOnGPU toGPU(const bool useFloat = false) const;
#endif
};

class ManifoldGenerator
{
private:
  bool _dt;
  bool _reldt;
  bool _panel_mode;
  bool _xmap_mode;
  bool _allow_missing;
  int _tau;
  int _p;
  int _num_extras, _num_extras_lagged;
  std::vector<double> _x, _xmap, _co_x, _t;
  std::vector<std::vector<double>> _extras;
  std::vector<int> _panelIDs;

  std::vector<int> _observation_number;

  void setup_observation_numbers();
  void fill_in_point(int i, int E, bool copredictionMode, bool predictionSet, double dtWeight, double* point,
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
                    int numExtrasLagged = 0, bool dt = false, bool reldt = false, bool allowMissing = false)
    : _t(t)
    , _x(x)
    , _tau(tau)
    , _p(p)
    , _xmap(xmap)
    , _co_x(co_x)
    , _panelIDs(panelIDs)
    , _extras(extras)
    , _num_extras((int)extras.size())
    , _num_extras_lagged(numExtrasLagged)
    , _dt(dt)
    , _reldt(reldt)
    , _allow_missing(allowMissing)
  {
    _panel_mode = (panelIDs.size() > 0);
    _xmap_mode = (xmap.size() > 0);
    setup_observation_numbers();
  }

  KeenManifold create_manifold(int E, const std::vector<bool>& filter, bool predictionSet, double dtWeight = 0.0,
                               bool copredictMode = false, bool skipMissing = false) const;

  std::vector<bool> generate_usable(int maxE, bool copredictionMode = false) const;

  int E_dt(int E) const { return _dt * E; }
  int E_extras(int E) const { return _num_extras + _num_extras_lagged * (E - 1); }
  int E_actual(int E) const { return E + E_dt(E) + E_extras(E); }

  int numExtrasLagged() const { return _num_extras_lagged; }
  int numExtras() const { return _num_extras; }

  const std::vector<int>& panelIDs() const { return _panelIDs; }
};
