#pragma warning(disable : 4018)

#include "manifold.h"

ManifoldOnGPU Manifold::toGPU(const bool useFloat) const
{
  using af::array;

  if (useFloat) {
    return ManifoldOnGPU {
      array(_E_actual, _nobs, _flat.get()).as(f32),
      (_y.size() > 0 ? array(_nobs, _y.data()) : array()).as(f32),
      (_panel_ids.size() > 0 ? array(_nobs, _panel_ids.data()) : array()),
      _nobs, _E_x, _E_dt, _E_extras, _E_lagged_extras, _E_actual, _missing
    };
  } else {
    return ManifoldOnGPU {
      array(_E_actual, _nobs, _flat.get()),
      (_y.size() > 0 ? array(_nobs, _y.data()) : array()),
      (_panel_ids.size() > 0 ? array(_nobs, _panel_ids.data()) : array()),
      _nobs, _E_x, _E_dt, _E_extras, _E_lagged_extras, _E_actual, _missing
    };
  }
}

Manifold ManifoldGenerator::create_manifold(int E, const std::vector<bool>& filter, bool copredict,
                                            bool prediction) const
{
  bool panelMode = _panel_ids.size() > 0;

  std::vector<int> inds, panelIDs;
  std::vector<double> y;

  int nobs = 0;
  for (int i = 0; i < filter.size(); i++) {
    if (filter[i]) {
      inds.push_back(i);
      y.push_back(_y[i]);
      if (panelMode) {
        panelIDs.push_back(_panel_ids[i]);
      }
      nobs += 1;
    }
  }

  auto flat = std::make_unique<double[]>(nobs * E_actual(E));

  // Fill in the lagged embedding of x (or co_x) in the first columns
  for (int i = 0; i < nobs; i++) {
    for (int j = 0; j < E; j++) {
      if (prediction && copredict) {
        flat[i * E_actual(E) + j] = lagged(_co_x, inds, i, j);
      } else {
        flat[i * E_actual(E) + j] = lagged(_x, inds, i, j);
      }
    }
  }

  // Put the lagged embedding of dt in the next columns
  for (int i = 0; i < nobs; i++) {
    for (int j = 0; j < E_dt(E); j++) {
      flat[i * E_actual(E) + E + j] = find_dt(inds, i, j);
    }
  }

  // Finally put the extras in the last columns
  for (int i = 0; i < nobs; i++) {
    int offset = 0;
    for (int k = 0; k < _num_extras; k++) {
      int numLags = (k < _num_extras_lagged) ? E : 1;
      for (int j = 0; j < numLags; j++) {
        flat[i * E_actual(E) + E + E_dt(E) + offset + j] = lagged(_extras[k], inds, i, j);
      }
      offset += numLags;
    }
  }

  return { flat, y, panelIDs, nobs, E, E_dt(E), E_extras(E), E * numExtrasLagged(), E_actual(E), _missing };
}

double ManifoldGenerator::lagged(const std::vector<double>& vec, const std::vector<int>& inds, int i, int j) const
{
  int index = inds.at(i) - j * _tau;
  if (index < 0) {
    return _missing;
  }
  return vec[index];
}

double ManifoldGenerator::find_dt(const std::vector<int>& inds, int i, int j) const
{
  int ind1, ind2;
  if (_cumulative_dt) {
    ind1 = inds.at(i) + _tau;
    ind2 = ind1 - j * _tau;
  } else {
    ind1 = inds.at(i) + _add_dt0 * _tau - j * _tau;
    ind2 = ind1 - _tau;
  }

  if ((ind1 >= _t.size()) || (ind2 < 0) || (_t[ind1] == _missing) || (_t[ind2] == _missing) || (_t[ind1] < _t[ind2])) {
    return _missing;
  }
  return _dtWeight * (_t[ind1] - _t[ind2]);
}

void to_json(json& j, const ManifoldGenerator& g)
{
  j = json{ { "_use_dt", g._use_dt },
            { "_add_dt0", g._add_dt0 },
            { "_cumulative_dt", g._cumulative_dt },
            { "_tau", g._tau },
            { "_missing", g._missing },
            { "_num_extras", g._num_extras },
            { "_num_extras_lagged", g._num_extras_lagged },
            { "_dtWeight", g._dtWeight },
            { "_x", g._x },
            { "_y", g._y },
            { "_co_x", g._co_x },
            { "_t", g._t },
            { "_extras", g._extras },
            { "_panel_ids", g._panel_ids } };
}

void from_json(const json& j, ManifoldGenerator& g)
{
  j.at("_use_dt").get_to(g._use_dt);
  j.at("_add_dt0").get_to(g._add_dt0);
  j.at("_cumulative_dt").get_to(g._cumulative_dt);
  j.at("_tau").get_to(g._tau);
  j.at("_missing").get_to(g._missing);
  j.at("_num_extras").get_to(g._num_extras);
  j.at("_num_extras_lagged").get_to(g._num_extras_lagged);
  j.at("_dtWeight").get_to(g._dtWeight);
  j.at("_x").get_to(g._x);
  j.at("_y").get_to(g._y);
  j.at("_co_x").get_to(g._co_x);
  j.at("_t").get_to(g._t);
  j.at("_extras").get_to(g._extras);
  j.at("_panel_ids").get_to(g._panel_ids);
}
