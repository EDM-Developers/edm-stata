#include "manifold.h"

ManifoldGenerator::ManifoldGenerator(std::vector<double> x, std::vector<double> y, std::vector<double> co_x,
                                     std::vector<std::vector<double>> extras, std::vector<double> t, size_t E,
                                     double dtWeight, double missing, size_t tau)
  : _x(x)
  , _y(y)
  , _co_x(co_x)
  , _extras(extras)
  , _t(t)
  , _dtWeight(dtWeight)
  , _missing(missing)
{
  // TODO: Add 'tau' != 1 support
  _copredict = (co_x.size() > 0);
  _use_dt = (dtWeight > 0);
  _E_x = E;
  _E_dt = (_use_dt) * (E - 1);
  _E_extras = extras.size();
  _E_actual = _E_x + _E_dt + _E_extras;
}

Manifold ManifoldGenerator::create_manifold(std::vector<bool> filter, bool prediction)
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

  auto flat = std::make_unique<double[]>(nobs * _E_actual);

  // Fill in the lagged embedding of x (or co_x) in the first columns
  for (size_t i = 0; i < nobs; i++) {
    for (size_t j = 0; j < _E_x; j++) {
      if (prediction && _copredict) {
        flat[i * _E_actual + j] = find_co_x(inds, i, j);
      } else {
        flat[i * _E_actual + j] = find_x(inds, i, j);
      }
    }
  }

  // Put the lagged embedding of dt in the next columns
  for (size_t i = 0; i < nobs; i++) {
    for (size_t j = 0; j < _E_dt; j++) {
      flat[i * _E_actual + _E_x + j] = find_dt(inds, i, j);
    }
  }

  // Finally put the unlagged extras in the last columns
  for (size_t i = 0; i < nobs; i++) {
    for (size_t j = 0; j < _E_extras; j++) {
      flat[i * _E_actual + _E_x + _E_dt + j] = find_extras(inds, i, j);
    }
  }

  return { flat, y, nobs, _E_x, _E_dt, _E_extras, _E_actual, _missing };
}

double ManifoldGenerator::find_x(std::vector<size_t> inds, size_t i, size_t j) const
{
  size_t index = inds.at(i);
  if (index < j) {
    return _missing;
  }
  return _x[index - j];
}

double ManifoldGenerator::find_co_x(std::vector<size_t> inds, size_t i, size_t j) const
{
  size_t index = inds.at(i);
  if (index < j) {
    return _missing;
  }
  return _co_x[index - j];
}

double ManifoldGenerator::find_dt(std::vector<size_t> inds, size_t i, size_t j) const
{
  size_t index = inds.at(i);
  if (index < j + 1 || _t[index - 1] == _missing || _t[index] == _missing) {
    return _missing;
  }
  index -= j;
  return _dtWeight * (_t[index] - _t[index - 1]);
}

double ManifoldGenerator::find_extras(std::vector<size_t> inds, size_t i, size_t j) const
{
  size_t index = inds.at(i);
  return _extras.at(j).at(index);
}