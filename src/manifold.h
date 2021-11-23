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
  std::vector<double> _x, _xmap, _co_x;
  std::vector<std::vector<double>> _extras;

  std::vector<int> _observation_number;

  void setup_observation_numbers();

  bool find_observation_num(int target, int& k, int direction, int panel) const;
  std::vector<int> get_lagged_indices(int startIndex, int E, int panel) const;

public:
  std::vector<double> _t;
  std::vector<int> _panelIDs;
  void fill_in_point(int i, int E, bool copredictionMode, bool predictionSet, double dtWeight, double* point,
                     double& target) const;
  double get_target(int i, bool copredictionMode, bool predictionSet, int& targetIndex) const;

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

  std::vector<bool> generate_usable(int maxE, bool copredictionMode = false) const;

  int E_dt(int E) const { return _dt * E; }
  int E_extras(int E) const { return _num_extras + _num_extras_lagged * (E - 1); }
  int E_actual(int E) const { return E + E_dt(E) + E_extras(E); }

  int numExtrasLagged() const { return _num_extras_lagged; }
  int numExtras() const { return _num_extras; }

  const std::vector<int>& panelIDs() const { return _panelIDs; }
};

class Manifold
{
  std::shared_ptr<double[]> _flat = nullptr;
  std::vector<double> _targets;
  std::vector<int> _panelIDs;
  int _numPoints, _E_x, _E_dt, _E_extras, _E_lagged_extras, _E_actual;

public:
  Manifold(const std::shared_ptr<ManifoldGenerator> gen, int E, const std::vector<bool>& filter, bool predictionSet,
           double dtWeight = 0.0, bool copredictMode = false)
  {
    _init(*gen, E, filter, predictionSet, dtWeight, copredictMode);
  }

  Manifold(const ManifoldGenerator& gen, int E, const std::vector<bool>& filter, bool predictionSet,
           double dtWeight = 0.0, bool copredictMode = false) {
    _init(gen, E, filter, predictionSet, dtWeight, copredictMode);
  }

    void _init(const ManifoldGenerator& gen, int E, const std::vector<bool>& filter, bool predictionSet,
             double dtWeight, bool copredictMode) {
    _E_x = E;
    _E_dt = gen.E_dt(E);
    _E_extras = gen.E_extras(E);
    _E_lagged_extras = gen.numExtrasLagged() * E;
    _E_actual = gen.E_actual(E);

    bool takeEveryPoint = filter.size() == 0;

    std::vector<int> pointNumToStartIndex;
    std::vector<double> targetTimes;

    bool panelMode = gen._panelIDs.size() > 0;

    for (int i = 0; i < gen._t.size(); i++) {
      if (takeEveryPoint || filter[i]) {

        // Throwing away library set points whose targets are missing.
        int targetIndex = i;
        double target = gen.get_target(i, copredictMode, predictionSet, targetIndex);
        if (!predictionSet && (target == MISSING_D)) {
          continue;
        }

        _targets.push_back(target);
        // targetTimes.push_back(gen._t[targetIndex]);

        if (panelMode) {
          _panelIDs.push_back(gen._panelIDs[i]);
        }
        pointNumToStartIndex.push_back(i);
      }
    }

    _numPoints = pointNumToStartIndex.size();

    // Fill in the manifold row-by-row (point-by-point)
    _flat = std::shared_ptr<double[]>(new double[_numPoints * _E_actual], std::default_delete<double[]>());

    for (int i = 0; i < _numPoints; i++) {
      double target;
      double* point = &(_flat[i * _E_actual]);
      gen.fill_in_point(pointNumToStartIndex[i], E, copredictMode, predictionSet, dtWeight, point, target);
    }
  }

  double operator()(int i, int j) const { return _flat[i * _E_actual + j]; }

  void fill_in_point(int i, double* point) const
  {
    //    gen->fill_in_point(int i, int E, bool copredictionMode, bool predictionSet, double dtWeight, double* point,
    //                       double& target) const;

    for (int j = 0; j < _E_actual; j++) {
      point[j] = _flat[i * _E_actual + j];
    }
  }

  Eigen::Map<const Eigen::VectorXd> targetsMap() const { return { &(_targets[0]), _numPoints }; }

  double x(int i, int j) const { return _flat[i * _E_actual + j]; }
  double dt(int i, int j) const { return _flat[i * _E_actual + _E_x + j]; }
  double extras(int i, int j) const { return _flat[i * _E_actual + _E_x + _E_dt + j]; }
  int panel(int i) const { return _panelIDs[i]; }

  double missing() const { return MISSING_D; }

  double target(int i) const { return _targets[i]; }
  int numTargets() const { return (int)_targets.size(); }
  const std::vector<double>& targets() const { return _targets; }

  double* data() const { return _flat.get(); };
  int numPoints() const { return _numPoints; }
  int E() const { return _E_x; }
  int E_dt() const { return _E_dt; }
  int E_lagged_extras() const { return _E_lagged_extras; }
  int E_extras() const { return _E_extras; }
  int E_actual() const { return _E_actual; }
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
