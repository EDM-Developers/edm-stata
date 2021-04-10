#include "manifold.h"

Manifold ManifoldGenerator::create_manifold(size_t E, const std::vector<bool>& filter, bool prediction) const
{
  std::vector<size_t> inds;
  std::vector<double> y;

  size_t nobs = 0;
  for (size_t i = 0; i < filter.size(); i++) {
    if (filter[i]) {
      inds.push_back(i);
      y.push_back(_y[i]);
      nobs += 1;
    }
  }

  auto flat = std::make_unique<double[]>(nobs * E_actual(E));

  // Fill in the lagged embedding of x (or co_x) in the first columns
  for (size_t i = 0; i < nobs; i++) {
    for (size_t j = 0; j < E; j++) {
      if (prediction && _copredict) {
        flat[i * E_actual(E) + j] = lagged(_co_x, inds, i, j);
      } else {
        flat[i * E_actual(E) + j] = lagged(_x, inds, i, j);
      }
    }
  }

  // Put the lagged embedding of dt in the next columns
  for (size_t i = 0; i < nobs; i++) {
    for (size_t j = 0; j < E_dt(E); j++) {
      flat[i * E_actual(E) + E + j] = find_dt(inds, i, j);
    }
  }

  // Finally put the extras in the last columns
  for (size_t i = 0; i < nobs; i++) {
    int offset = 0;
    for (size_t k = 0; k < _num_extras; k++) {
      int numLags = _extrasEVarying[k] ? E : 1;
      for (size_t j = 0; j < numLags; j++) {
        flat[i * E_actual(E) + E + E_dt(E) + offset + j] = lagged(_extras[k], inds, i, j);
      }
      offset += numLags;
    }
  }

  return { flat, y, nobs, E, E_dt(E), E_extras(E), E_actual(E), _missing };
}

double ManifoldGenerator::lagged(const std::vector<double>& vec, const std::vector<size_t>& inds, size_t i,
                                 size_t j) const
{
  int index = inds.at(i) - j * _tau;
  if (index < 0) {
    return _missing;
  }
  return vec[index];
}

double ManifoldGenerator::find_dt(const std::vector<size_t>& inds, size_t i, size_t j) const
{
  int ind1 = inds.at(i) + _add_dt0 * _tau - j * _tau;
  int ind2 = ind1 - _tau;

  if ((ind1 >= _t.size()) || (ind2 < 0) || (_t[ind1] == _missing) || (_t[ind2] == _missing) || (_t[ind1] < _t[ind2])) {
    return _missing;
  }
  return _dtWeight * (_t[ind1] - _t[ind2]);
}

void to_json(json& j, const ManifoldGenerator& g)
{
  j = json{ { "_copredict", g._copredict },
            { "_use_dt", g._use_dt },
            { "_add_dt0", g._add_dt0 },
            { "_tau", g._tau },
            { "_missing", g._missing },
            { "_nobs", g._nobs },
            { "_num_extras", g._num_extras },
            { "_num_extras_varying", g._num_extras_varying },
            { "_dtWeight", g._dtWeight },
            { "_x", g._x },
            { "_y", g._y },
            { "_co_x", g._co_x },
            { "_t", g._t },
            { "_extras", g._extras },
            { "_extrasEVarying", g._extrasEVarying } };
}

void from_json(const json& j, ManifoldGenerator& g)
{
  j.at("_copredict").get_to(g._copredict);
  j.at("_use_dt").get_to(g._use_dt);
  j.at("_add_dt0").get_to(g._add_dt0);
  j.at("_tau").get_to(g._tau);
  j.at("_missing").get_to(g._missing);
  j.at("_nobs").get_to(g._nobs);
  j.at("_num_extras").get_to(g._num_extras);
  j.at("_num_extras_varying").get_to(g._num_extras_varying);
  j.at("_dtWeight").get_to(g._dtWeight);
  j.at("_x").get_to(g._x);
  j.at("_y").get_to(g._y);
  j.at("_co_x").get_to(g._co_x);
  j.at("_t").get_to(g._t);
  j.at("_extras").get_to(g._extras);
  j.at("_extrasEVarying").get_to(g._extrasEVarying);
}
